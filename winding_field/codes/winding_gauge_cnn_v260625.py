"""
winding_gauge_cnn_v260625.py

Gauge / magnitude test for the per-cell winding field. Retrains the SAME 3-way CNN
(same architecture/training as winding_cnn_v260625.py) on transformed winding inputs,
to decompose what fate-relevant content the winding field carries:

  --rep signed     : (wx, wy) as saved            [2 ch]  BASELINE (= winding_cnn 'winding')
  --rep abs        : (|wx|, |wy|)                 [2 ch]  drops ALL sign info
  --rep canonical  : global-flip gauge-fixed      [2 ch]  fix sign so first nonzero comp > 0;
                                                          keeps relative sign, removes gauge
  --rep mask       : 1 if (wx,wy)!=(0,0) else 0   [1 ch]  binary wrap-mask (magnitude gone too)

Logic:
  - if abs ~= signed          -> individual signs don't matter
  - if canonical ~= signed > abs -> relative sign (diagonal / vs \) matters
  - if mask ~= signed         -> only "which cells wrap" matters (magnitude irrelevant)

Fields from winding_fields_v260625/fields_Ttrue_f{f}{,_val,_test}.npz (key: winding (N,2,196) int8).
3-way balanced accuracy, N up to 850k, seeds [0,1,2].

Usage:
    python winding_gauge_cnn_v260625.py --rep signed
    python winding_gauge_cnn_v260625.py --rep abs
    python winding_gauge_cnn_v260625.py --rep canonical
    python winding_gauge_cnn_v260625.py --rep mask
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
OUT_ROOT = os.path.join(ROOT, "winding_gauge_results_v260625")
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
REP_CHANNELS = {"signed": 2, "abs": 2, "canonical": 2, "mask": 1}


def log(m):
    print(m, flush=True)


def split_dir(frac):
    return os.path.join(SPLITS_ROOT, f"ml_splits_v260529_{CLOCK}_f{frac:.2f}")


def load_fate(p):
    with np.load(p) as z:
        return np.asarray(z["final_pattern_type"]).astype(np.int64)


def load_winding(frac, suffix):
    p = os.path.join(FIELD_DIR, f"fields_{CLOCK}_f{frac:.2f}{suffix}.npz")
    with np.load(p) as z:
        return z["winding"].astype(np.float32)   # (N,2,196)


def transform(winding, rep):
    """winding: (N,2,196) signed. Return (N, C, 14, 14) float32 per rep."""
    N = winding.shape[0]
    wx = winding[:, 0, :]  # (N,196)
    wy = winding[:, 1, :]
    if rep == "signed":
        out = winding.reshape(N, 2, 14, 14)
    elif rep == "abs":
        a = np.abs(winding)
        out = a.reshape(N, 2, 14, 14)
    elif rep == "canonical":
        # global flip per cell: make the first nonzero component positive.
        # sign s = sign(wx) if wx!=0 else sign(wy); multiply both by s.
        s = np.sign(wx)
        s_wy = np.sign(wy)
        s = np.where(wx != 0, s, s_wy)     # if wx==0 use sign(wy); if both 0, s=0
        s = np.where(s == 0, 1.0, s)       # non-wrapping cells: s=1 (leaves (0,0) unchanged)
        cx = (wx * s).reshape(N, 14, 14)
        cy = (wy * s).reshape(N, 14, 14)
        out = np.stack([cx, cy], axis=1).astype(np.float32)
    elif rep == "mask":
        m = ((wx != 0) | (wy != 0)).astype(np.float32)  # (N,196)
        out = m.reshape(N, 1, 14, 14)
    else:
        raise ValueError(rep)
    return np.ascontiguousarray(out.astype(np.float32))


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
    Ntr = X4_tr.shape[0]
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        perm = np.random.permutation(Ntr)
        for i in range(0, Ntr, BATCH_SIZE):
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
    ap.add_argument("--rep", choices=["signed", "abs", "canonical", "mask"], required=True)
    ap.add_argument("--n-train", type=int, default=N_TRAIN)
    args = ap.parse_args()
    rep = args.rep
    in_ch = REP_CHANNELS[rep]
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    os.makedirs(OUT_ROOT, exist_ok=True)
    csv_path = os.path.join(OUT_ROOT, f"winding_gauge_recall_{rep}.csv")
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
        Xtr_pool = np.load(os.path.join(sd, "X_train_pool.npy"))  # only for count/row-alignment
        n = min(args.n_train, Xtr_pool.shape[0])
        ntot = Xtr_pool.shape[0]
        del Xtr_pool
        ytr_pool = load_fate(os.path.join(sd, "meta_train_pool.npz"))
        yval = (load_fate(os.path.join(sd, "meta_val.npz")) - 1).astype(np.int64)
        ytest = (load_fate(os.path.join(sd, "meta_test.npz")) - 1).astype(np.int64)

        w_tr = load_winding(frac, "")
        w_val = load_winding(frac, "_val")
        w_test = load_winding(frac, "_test")
        X4_val = transform(w_val, rep)
        X4_test = transform(w_test, rep)

        for seed in SEEDS:
            t1 = time.time()
            rng = np.random.default_rng(seed)
            idx = (rng.permutation(n) if n == ntot else rng.choice(ntot, n, replace=False))
            X4_tr = transform(w_tr[idx], rep)
            ytr3 = (ytr_pool[idx] - 1).astype(np.int64)
            counts = np.array([np.sum(ytr3 == k) for k in (0, 1, 2)], dtype=np.float64)
            class_w = counts.sum() / (3.0 * np.maximum(counts, 1.0))
            model, val_bal = train_one(X4_tr, ytr3, X4_val, yval, device, seed, class_w, in_ch)
            if model is None:
                for k in (1, 2, 3):
                    rows.append([rep, frac, n, seed, k, 0, float("nan"), float("nan"), float("nan"), "failed_nan"])
                continue
            pred = predict_labels(model, X4_test, device)
            test_bal = balanced_accuracy_score(ytest, pred)
            for k in (1, 2, 3):
                mask_k = (ytest == (k - 1)); n_true = int(mask_k.sum())
                recall = float(np.mean(pred[mask_k] == (k - 1))) if n_true else float("nan")
                rows.append([rep, frac, n, seed, k, n_true, recall, val_bal, test_bal, "ok"])
            pd.DataFrame(rows, columns=cols).to_csv(csv_path, index=False)
            log(f"  [f={frac:.2f} seed{seed}] ({time.time()-t1:.0f}s) val={val_bal:.3f} test={test_bal:.3f}")
            del X4_tr
        del w_tr, w_val, w_test, X4_val, X4_test

    df = pd.DataFrame(rows, columns=cols)
    ok = df[df["status"] == "ok"]
    if len(ok):
        agg = ok.groupby("fraction", as_index=False)["test_balacc"].mean()
        agg.to_csv(os.path.join(OUT_ROOT, f"winding_gauge_balacc_agg_{rep}.csv"), index=False)
        log("\n" + "=" * 60)
        log(f"3-way test balanced accuracy vs f  (rep={rep})")
        log("  " + "  ".join(f"{r.fraction:.1f}:{r.test_balacc:.3f}" for r in agg.itertuples()))
    with open(os.path.join(OUT_ROOT, f"settings_{rep}.json"), "w") as f:
        json.dump(dict(rep=rep, in_ch=in_ch, n_train=args.n_train, seeds=SEEDS,
                       device=device, torch=torch.__version__), f, indent=2)
    log(f"\nTotal elapsed: {(time.time()-t0)/3600:.2f} h")


if __name__ == "__main__":
    main()
