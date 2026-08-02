"""
winding_cnn_v260625.py  [STAGE 2 of the per-cell winding-field experiment]

Train the 3-way CNN on per-cell TOPOLOGICAL fields and compare recall-vs-f to the
full-config baseline. Tests whether a per-cell winding field (Kind-2, global) is
sufficient to predict fate, and whether adding vortex info helps.

Input representations (--rep):
  winding       : 2 channels = per-cell (wx, wy) of the cell's color-cluster
                  (componentwise max-abs signed winding; 0 for non-wrapping clusters)
  winding_vortex: 4 channels = winding (2) + signed vortex-core field (2)
                  (channel +1-cores, channel -1-cores; painted on core blobs)
  full          : 4 channels = cell-state one-hot (the standard full-config baseline;
                  consistency check, should reproduce the known 3-way curve)

Fields from winding_fields_v260625/fields_Ttrue_f{f}{,_val,_test}.npz
  (keys: winding (N,2,196) int8, vortex (N,2,196) int8), row-aligned with split X.

3-way (chance 0.333), N up to 850k, seeds [0,1,2], same SmallCNN-style net + training
as the other ablations (conv channels adapt to input channel count). MPS-safe loss.

Usage:
    python winding_cnn_v260625.py --rep winding
    python winding_cnn_v260625.py --rep winding_vortex
    python winding_cnn_v260625.py --rep full
"""

import os
import json
import time
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = "<PATH_TO_OUTPUT>"
SPLITS_ROOT = os.path.join(ROOT, "ml_splits_v260529")
FIELD_DIR = os.path.join(ROOT, "winding_fields_v260625")
OUT_ROOT = os.path.join(ROOT, "winding_cnn_results_v260625")
CLOCK = "Ttrue"
FRACTIONS = [0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
N_TRAIN = 850_000
SEEDS = [0, 1, 2]
BATCH_SIZE = 1024
MAX_EPOCHS = 25
PATIENCE = 5
LR = 3e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 1.0
FATE_NAMES = {1: "static", 2: "spiral", 3: "rectilinear"}
REP_CHANNELS = {"winding": 2, "winding_vortex": 4, "full": 4}


def log(m):
    print(m, flush=True)


def split_dir(frac):
    return os.path.join(SPLITS_ROOT, f"ml_splits_v260529_{CLOCK}_f{frac:.2f}")


def load_fate(p):
    with np.load(p) as z:
        return np.asarray(z["final_pattern_type"]).astype(np.int64)


def load_fields(frac, suffix):
    p = os.path.join(FIELD_DIR, f"fields_{CLOCK}_f{frac:.2f}{suffix}.npz")
    with np.load(p) as z:
        return z["winding"].astype(np.float32), z["vortex"].astype(np.float32)


def build_input(X_flat_int, rep, winding, vortex):
    """Return (N, C, 14, 14) float32 input for the representation."""
    N = X_flat_int.shape[0]
    if rep == "full":
        X = X_flat_int.astype(np.int64) - 1
        out = np.zeros((N, 4, 14, 14), dtype=np.float32)
        for k in range(4):
            out[:, k, :, :] = (X == k).reshape(N, 14, 14)
        return out
    if rep == "winding":
        return winding.reshape(N, 2, 14, 14)
    if rep == "winding_vortex":
        out = np.zeros((N, 4, 14, 14), dtype=np.float32)
        out[:, 0:2, :, :] = winding.reshape(N, 2, 14, 14)
        out[:, 2:4, :, :] = vortex.reshape(N, 2, 14, 14)
        return out
    raise ValueError(rep)


class SmallCNN3(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, 3)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


@torch.no_grad()
def predict_labels(model, X4, device, batch=4096):
    model.eval()
    preds = []
    for i in range(0, X4.shape[0], batch):
        xb = torch.from_numpy(X4[i:i + batch]).to(device=device, dtype=torch.float32)
        preds.append(model(xb).argmax(1).cpu().numpy())
    return np.concatenate(preds)


def train_one(X4_tr, y_tr, X4_val, y_val, device, seed, class_w, in_ch):
    torch.manual_seed(seed); np.random.seed(seed)
    model = SmallCNN3(in_ch).to(device)
    cw = torch.tensor(class_w, device=device, dtype=torch.float32)

    def loss_fn(logits, target):
        return F.nll_loss(F.log_softmax(logits, dim=1), target, weight=cw)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best_bal, best_state, bad = -1.0, None, 0
    N = X4_tr.shape[0]
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        perm = np.random.permutation(N)
        for i in range(0, N, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            xb = torch.from_numpy(X4_tr[idx]).to(device=device, dtype=torch.float32)
            yb = torch.from_numpy(y_tr[idx]).to(device=device, dtype=torch.long)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            if not torch.isfinite(loss):
                return None, float("nan")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            opt.step()
        bal = balanced_accuracy_score(y_val, predict_labels(model, X4_val, device))
        if bal > best_bal:
            best_bal, bad = bal, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, float(best_bal)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep", choices=["winding", "winding_vortex", "full"], required=True)
    ap.add_argument("--n-train", type=int, default=N_TRAIN)
    args = ap.parse_args()
    rep = args.rep; in_ch = REP_CHANNELS[rep]
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    os.makedirs(OUT_ROOT, exist_ok=True)
    csv_path = os.path.join(OUT_ROOT, f"winding_cnn_recall_{rep}.csv")
    log(f"Device: {device}  rep={rep} (in_ch={in_ch})  n_train={args.n_train}")

    cols = ["rep", "fraction", "N_train", "seed", "true_fate", "n_true",
            "recall", "val_balacc", "test_balacc", "status"]
    rows = []
    t0 = time.time()

    for frac in FRACTIONS:
        sd = split_dir(frac)
        if not os.path.isdir(sd):
            log(f"[f={frac:.2f}] MISSING split"); continue
        log(f"\n[f={frac:.2f}] loading")
        Xtr_pool = np.load(os.path.join(sd, "X_train_pool.npy"))
        Xval = np.load(os.path.join(sd, "X_val.npy"))
        Xtest = np.load(os.path.join(sd, "X_test.npy"))
        ytr_pool = load_fate(os.path.join(sd, "meta_train_pool.npz"))
        yval = (load_fate(os.path.join(sd, "meta_val.npz")) - 1).astype(np.int64)
        ytest = (load_fate(os.path.join(sd, "meta_test.npz")) - 1).astype(np.int64)
        if rep == "full":
            w_tr = v_tr = w_val = v_val = w_test = v_test = None
        else:
            w_tr, v_tr = load_fields(frac, "")
            w_val, v_val = load_fields(frac, "_val")
            w_test, v_test = load_fields(frac, "_test")
        n = min(args.n_train, Xtr_pool.shape[0])

        X4_val = build_input(Xval, rep, w_val, v_val)
        X4_test = build_input(Xtest, rep, w_test, v_test)
        for seed in SEEDS:
            t1 = time.time()
            idx = (np.random.default_rng(seed).permutation(n) if n == Xtr_pool.shape[0]
                   else np.random.default_rng(seed).choice(Xtr_pool.shape[0], n, replace=False))
            w_tr_i = None if w_tr is None else w_tr[idx]
            v_tr_i = None if v_tr is None else v_tr[idx]
            X4_tr = build_input(Xtr_pool[idx], rep, w_tr_i, v_tr_i)
            ytr3 = (ytr_pool[idx] - 1).astype(np.int64)
            counts = np.array([np.sum(ytr3 == k) for k in (0, 1, 2)], dtype=np.float64)
            class_w = counts.sum() / (3.0 * np.maximum(counts, 1.0))
            model, val_bal = train_one(X4_tr, ytr3, X4_val, yval, device, seed, class_w, in_ch)
            if model is None:
                for k in (1, 2, 3):
                    rows.append([rep, frac, n, seed, k, 0, float("nan"),
                                 float("nan"), float("nan"), "failed_nan"])
                continue
            pred = predict_labels(model, X4_test, device)
            test_bal = balanced_accuracy_score(ytest, pred)
            for k in (1, 2, 3):
                mask = (ytest == (k - 1)); n_true = int(mask.sum())
                recall = float(np.mean(pred[mask] == (k - 1))) if n_true else float("nan")
                rows.append([rep, frac, n, seed, k, n_true, recall, val_bal, test_bal, "ok"])
            pd.DataFrame(rows, columns=cols).to_csv(csv_path, index=False)
            log(f"  [f={frac:.2f} seed{seed}] ({time.time()-t1:.0f}s) val={val_bal:.3f} test={test_bal:.3f}")
            del X4_tr
        del X4_val, X4_test, Xtr_pool, Xval, Xtest

    df = pd.DataFrame(rows, columns=cols)
    ok = df[df["status"] == "ok"]
    if len(ok):
        agg = ok.groupby("fraction", as_index=False)["test_balacc"].mean()
        agg.to_csv(os.path.join(OUT_ROOT, f"winding_cnn_balacc_agg_{rep}.csv"), index=False)
        log("\n" + "=" * 60)
        log(f"3-way test balanced accuracy vs f  (rep={rep})")
        log("=" * 60)
        log("  " + "  ".join(f"{r.fraction:.1f}:{r.test_balacc:.2f}" for r in agg.itertuples()))
    with open(os.path.join(OUT_ROOT, f"settings_{rep}.json"), "w") as f:
        json.dump(dict(rep=rep, in_ch=in_ch, n_train=args.n_train, seeds=SEEDS,
                       device=device, torch=torch.__version__), f, indent=2)
    log(f"\nTotal elapsed: {(time.time()-t0)/3600:.2f} h")


if __name__ == "__main__":
    main()
