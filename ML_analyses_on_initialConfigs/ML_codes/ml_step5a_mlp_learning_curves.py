#!/usr/bin/env python3
"""
ml_step5a_mlp_learning_curves.py

Aggressive MLP (PyTorch) learning curves for predicting static vs dynamic fate
from initial 14x14 (196) 4-state lattice configuration.

- Inputs: integer states in {1,2,3,4} of shape (N,196)
- Label: 1 = static, 0 = dynamic
- Train pool: X_train_pool.npy / y_train_pool.npy
- Val:        X_val.npy        / y_val.npy
- Test:       X_test.npy       / y_test.npy (frozen)

Protocol:
- Train sizes: [10k, 30k, 100k, 300k, 850k] (configurable)
- Seeds: [0,1,2]
- For each (N,seed): sample N without replacement from train pool
- Train an aggressive MLP, early-stop on VAL AUC
- Choose threshold on VAL that maximizes balanced accuracy
- Report TEST AUC + (acc, bal_acc) at that threshold
- Save CSVs + plots (PNG/PDF) + settings.json
"""

import os
import json
import time
import math
import platform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, balanced_accuracy_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# -------------------------
# Config
# -------------------------
SPLIT_DIR = "ml_splits_v251209"

OUT_DIR = "ml_results_step5_mlp"
FIG_DIR = os.path.join(OUT_DIR, "figs")
os.makedirs(FIG_DIR, exist_ok=True)

# Match Step 4 sizes by default
TRAIN_SIZES = [10_000, 30_000, 100_000, 300_000, 850_000]
SEEDS = [0, 1, 2]

# Training hyperparams (aggressive but stable)
MAX_EPOCHS = 50
BATCH_SIZE = 2048
LR = 1e-3
WEIGHT_DECAY = 1e-5

# Early stopping on VAL AUC
EARLY_STOPPING_PATIENCE = 6   # epochs without improvement
MIN_DELTA_AUC = 1e-4          # improvement threshold

# Model (aggressive)
HIDDEN_DIMS = [2048, 2048, 1024, 512]
DROPOUT_P = 0.25



# Reproducibility
TORCH_DETERMINISTIC = False  # deterministic mode is disabled for performance; MPS backend does not fully support it



# -------------------------
# Device
# -------------------------
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# -------------------------
# Data utils
# -------------------------
def load_arrays():
    X_train_pool = np.load(os.path.join(SPLIT_DIR, "X_train_pool.npy"), mmap_mode="r")
    y_train_pool = np.load(os.path.join(SPLIT_DIR, "y_train_pool.npy"), mmap_mode="r")

    X_val = np.load(os.path.join(SPLIT_DIR, "X_val.npy"), mmap_mode="r")
    y_val = np.load(os.path.join(SPLIT_DIR, "y_val.npy"), mmap_mode="r")

    X_test = np.load(os.path.join(SPLIT_DIR, "X_test.npy"), mmap_mode="r")
    y_test = np.load(os.path.join(SPLIT_DIR, "y_test.npy"), mmap_mode="r")
    return X_train_pool, y_train_pool, X_val, y_val, X_test, y_test


def subsample_indices(n_total, n, seed):
    rng = np.random.default_rng(seed)
    if n > n_total:
        raise ValueError(f"Requested n={n} > available={n_total}")
    return rng.choice(n_total, size=n, replace=False)


class NumpyIndexDataset(Dataset):
    """
    Wraps memmapped numpy arrays with an index list.
    Returns:
      x: int64 tensor of shape (196,), values in {1,2,3,4}
      y: float32 tensor scalar in {0,1}
    """
    def __init__(self, X_mmap, y_mmap, indices):
        self.X = X_mmap
        self.y = y_mmap
        self.idx = np.asarray(indices)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = int(self.idx[i])
        x = np.asarray(self.X[j], dtype=np.int64)    # (196,)
        y = float(self.y[j])
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)


# -------------------------
# Metrics / threshold selection
# -------------------------
def best_threshold_balanced_accuracy(y_true, scores):
    """
    Choose threshold maximizing balanced accuracy using ROC thresholds.
    """
    fpr, tpr, thr = roc_curve(y_true, scores)
    bal = 0.5 * (tpr + (1.0 - fpr))
    k = int(np.argmax(bal))
    return float(thr[k]), float(bal[k])


def eval_at_threshold(y_true, scores, thr):
    y_pred = (scores >= thr).astype(np.int8)
    acc = accuracy_score(y_true, y_pred)
    bal = balanced_accuracy_score(y_true, y_pred)
    frac_pos = float(np.mean(y_pred))
    return float(acc), float(bal), frac_pos


# -------------------------
# Model
# -------------------------
class AggressiveMLP(nn.Module):
    """
    One-hot inside the model:
      x: (B,196) int64 values in {1,2,3,4}
      -> one_hot(x-1, 4): (B,196,4)
      -> flatten to (B,784)
      -> deep MLP
    """
    def __init__(self, hidden_dims, dropout_p=0.25):
        super().__init__()
        in_dim = 196 * 4
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=dropout_p))
            prev = h
        layers.append(nn.Linear(prev, 1))  # logits
        self.net = nn.Sequential(*layers)

    def forward(self, x_int):
        # x_int: (B,196), values in {1,2,3,4}
        x_oh = F.one_hot((x_int - 1).clamp(min=0, max=3), num_classes=4).float()
        x = x_oh.view(x_oh.size(0), -1)  # (B,784)
        logits = self.net(x).squeeze(1)
        return logits


# -------------------------
# Train / eval loops
# -------------------------
@torch.no_grad()
def predict_scores(model, loader, device):
    model.eval()
    all_scores = []
    all_y = []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        scores = torch.sigmoid(logits).detach().cpu().numpy()
        all_scores.append(scores)
        all_y.append(y.numpy())
    return np.concatenate(all_y), np.concatenate(all_scores)


def train_one_run(X_train_pool, y_train_pool, X_val, y_val, X_test, y_test, N, seed, device):
    # Repro
    torch.manual_seed(seed)
    np.random.seed(seed)
    if TORCH_DETERMINISTIC:
        torch.use_deterministic_algorithms(True)

    # Indices and datasets
    tr_idx = subsample_indices(len(y_train_pool), N, seed)
    ds_tr = NumpyIndexDataset(X_train_pool, y_train_pool, tr_idx)

    # Fixed val/test datasets (full)
    ds_val = NumpyIndexDataset(X_val, y_val, np.arange(len(y_val)))
    ds_test = NumpyIndexDataset(X_test, y_test, np.arange(len(y_test)))

    # Loaders
    dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=False)
    dl_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    dl_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Compute pos_weight from *this* training subset for BCEWithLogitsLoss
    y_tr_np = np.asarray(y_train_pool)[tr_idx].astype(np.int8)
    pos = float(np.sum(y_tr_np == 1))
    neg = float(np.sum(y_tr_np == 0))
    pos_weight = (neg / pos) if pos > 0 else 1.0  # >1 increases weight on positive class
    pos_weight_t = torch.tensor([pos_weight], dtype=torch.float32, device=device)

    # Model
    model = AggressiveMLP(HIDDEN_DIMS, dropout_p=DROPOUT_P).to(device)

    # Loss / opt
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_t)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val_auc = -np.inf
    best_state = None
    epochs_no_improve = 0

    t0 = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        n_seen = 0

        for x, y in dl_tr:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            bs = x.size(0)
            running_loss += float(loss.item()) * bs
            n_seen += bs

        train_loss = running_loss / max(1, n_seen)

        # Evaluate VAL AUC
        yv, sv = predict_scores(model, dl_val, device)
        try:
            val_auc = roc_auc_score(yv, sv)
        except ValueError:
            val_auc = float("nan")

        # Early stopping
        improved = val_auc > (best_val_auc + MIN_DELTA_AUC)
        if improved:
            best_val_auc = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        print(f"    epoch {epoch:02d}: train_loss={train_loss:.4f}  val_auc={val_auc:.4f}"
              f"{'  *' if improved else ''}")

        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final VAL scores (for threshold)
    yv, sv = predict_scores(model, dl_val, device)
    yt, st = predict_scores(model, dl_test, device)

    # Metrics
    auc_val = roc_auc_score(yv, sv)
    auc_test = roc_auc_score(yt, st)

    thr, bal_val_at_thr = best_threshold_balanced_accuracy(yv, sv)
    acc_test, bal_test, frac_pos_test = eval_at_threshold(yt, st, thr)

    dt = time.time() - t0

    return dict(
        model="mlp_onehot_aggressive",
        N_train=int(N),
        seed=int(seed),
        frac_static_train=float(np.mean(y_tr_np)),
        pos_weight=float(pos_weight),

        auc_val=float(auc_val),
        thr_val_bestbal=float(thr),
        balacc_val_at_thr=float(bal_val_at_thr),

        auc_test=float(auc_test),
        acc_test_at_thr=float(acc_test),
        balacc_test_at_thr=float(bal_test),
        frac_pred_pos_test_at_thr=float(frac_pos_test),

        fit_eval_seconds=float(dt),
        best_val_auc=float(best_val_auc),
        epochs_run=int(epoch),
    )


# -------------------------
# Plotting
# -------------------------
def save_plot(df_agg, metric, ylabel, outbase):
    plt.figure()
    x = np.log10(df_agg["N_train"].values.astype(float))
    y = df_agg[f"mean_{metric}"].values
    yerr = df_agg[f"std_{metric}"].values
    plt.errorbar(x, y, yerr=yerr, fmt="o-", capsize=3)
    plt.xlabel("log10(N_train)")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outbase + ".png", dpi=300)
    plt.savefig(outbase + ".pdf")
    plt.close()


# -------------------------
# Main
# -------------------------
def main():
    device = get_device()
    print("Device:", device)

    X_train_pool, y_train_pool, X_val, y_val, X_test, y_test = load_arrays()

    print("Loaded arrays:")
    print("  train_pool:", X_train_pool.shape, y_train_pool.shape, "frac_static=", float(np.mean(y_train_pool)))
    print("  val       :", X_val.shape, y_val.shape, "frac_static=", float(np.mean(y_val)))
    print("  test      :", X_test.shape, y_test.shape, "frac_static=", float(np.mean(y_test)))

    rows = []
    t_global0 = time.time()

    for N in TRAIN_SIZES:
        for seed in SEEDS:
            print(f"\nN={N} seed={seed}")
            row = train_one_run(X_train_pool, y_train_pool, X_val, y_val, X_test, y_test,
                                N=N, seed=seed, device=device)
            print(f"  AUC(val)={row['auc_val']:.4f} thr*={row['thr_val_bestbal']:.4f} bal(val@thr*)={row['balacc_val_at_thr']:.4f}")
            print(f"  AUC(test)={row['auc_test']:.4f} acc(test@thr*)={row['acc_test_at_thr']:.4f} "
                  f"bal(test@thr*)={row['balacc_test_at_thr']:.4f} frac_pred_pos={row['frac_pred_pos_test_at_thr']:.4f} "
                  f"epochs={row['epochs_run']} ({row['fit_eval_seconds']:.1f}s)")
            rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    out_csv = os.path.join(OUT_DIR, "learning_curves_mlp.csv")
    df.to_csv(out_csv, index=False)
    print("\nSaved:", out_csv)

    # Aggregate
    agg = df.groupby(["model", "N_train"], as_index=False).agg(
        mean_auc_test=("auc_test", "mean"),
        std_auc_test=("auc_test", "std"),
        mean_balacc_test=("balacc_test_at_thr", "mean"),
        std_balacc_test=("balacc_test_at_thr", "std"),
        mean_acc_test=("acc_test_at_thr", "mean"),
        std_acc_test=("acc_test_at_thr", "std"),
        mean_fit_eval_seconds=("fit_eval_seconds", "mean"),
    )

    out_agg = os.path.join(OUT_DIR, "learning_curves_mlp_agg.csv")
    agg.to_csv(out_agg, index=False)
    print("Saved:", out_agg)

    # Plots (PNG+PDF)
    save_plot(agg, "auc_test", "ROC-AUC on TEST", os.path.join(FIG_DIR, "mlp_learning_curve_auc_test"))
    save_plot(agg, "balacc_test", "Balanced accuracy on TEST (thr from VAL)", os.path.join(FIG_DIR, "mlp_learning_curve_balacc_test"))
    save_plot(agg, "acc_test", "Accuracy on TEST (thr from VAL)", os.path.join(FIG_DIR, "mlp_learning_curve_acc_test"))

    # Settings for SI reproducibility
    settings = dict(
        split_dir=SPLIT_DIR,
        train_sizes=TRAIN_SIZES,
        seeds=SEEDS,
        max_epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        min_delta_auc=MIN_DELTA_AUC,
        hidden_dims=HIDDEN_DIMS,
        dropout_p=DROPOUT_P,
        device=str(device),
        python_version=platform.python_version(),
        platform=platform.platform(),
        torch_version=torch.__version__,
        numpy_version=np.__version__,
        sklearn_version=None,  # omitted; version can be retrieved via importlib.metadata if needed
    )
    out_settings = os.path.join(OUT_DIR, "settings.json")
    with open(out_settings, "w") as f:
        json.dump(settings, f, indent=2)
    print("Saved:", out_settings)

    print("\nSaved figures to:", FIG_DIR)
    print(f"Total elapsed: {(time.time() - t_global0)/60:.1f} min")


if __name__ == "__main__":
    main()