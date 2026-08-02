import os
import json
import time
import math
import platform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, balanced_accuracy_score


# -------------------------
# Config
# -------------------------
SPLIT_DIR = "ml_splits_v251209"
OUT_DIR = "ml_results_step5b_cnn"
FIG_DIR = os.path.join(OUT_DIR, "figs")
os.makedirs(FIG_DIR, exist_ok=True)

# Results are saved incrementally after each run; if a large training size fails numerically,
# previously completed results are preserved.
TRAIN_SIZES = [10_000, 30_000, 100_000, 300_000, 850_000]  

SEEDS = [0, 1, 2]

# Stability-focused settings: gradient clipping and conservative LR prevent numerical divergence
BATCH_SIZE = 1024
MAX_EPOCHS = 25
PATIENCE = 5

LR = 3e-4              # conservative learning rate; higher values caused numerical instability
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 1.0   # crucial for preventing NaNs

# For additional stability, reduce LR to 1e-4 or reduce model width.


RANDOM_STATE = 0


# -------------------------
# Utilities
# -------------------------
def load_arrays():
    X_train_pool = np.load(os.path.join(SPLIT_DIR, "X_train_pool.npy"), mmap_mode="r")
    y_train_pool = np.load(os.path.join(SPLIT_DIR, "y_train_pool.npy"), mmap_mode="r")
    X_val = np.load(os.path.join(SPLIT_DIR, "X_val.npy"), mmap_mode="r")
    y_val = np.load(os.path.join(SPLIT_DIR, "y_val.npy"), mmap_mode="r")
    X_test = np.load(os.path.join(SPLIT_DIR, "X_test.npy"), mmap_mode="r")
    y_test = np.load(os.path.join(SPLIT_DIR, "y_test.npy"), mmap_mode="r")
    return X_train_pool, y_train_pool, X_val, y_val, X_test, y_test


def subsample_train(X, y, n, seed):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y), size=n, replace=False)
    return np.asarray(X[idx]), np.asarray(y[idx])


def to_4ch_onehot_14x14(X_flat_int):
    """
    X_flat_int: (N,196) with values in {1,2,3,4}
    returns: (N,4,14,14) float32
    """
    X = X_flat_int.astype(np.int64) - 1  # -> {0,1,2,3}
    N = X.shape[0]
    X = X.reshape(N, 14, 14)
    out = np.zeros((N, 4, 14, 14), dtype=np.float32)
    for k in range(4):
        out[:, k, :, :] = (X == k).astype(np.float32)
    return out


def best_threshold_balanced_accuracy(y_true, scores):
    fpr, tpr, thr = roc_curve(y_true, scores)
    balacc = 0.5 * (tpr + (1.0 - fpr))
    k = int(np.argmax(balacc))
    return float(thr[k]), float(balacc[k])


def eval_at_threshold(y_true, scores, thr):
    y_pred = (scores >= thr).astype(np.int8)
    return (
        float(accuracy_score(y_true, y_pred)),
        float(balanced_accuracy_score(y_true, y_pred)),
        float(np.mean(y_pred)),
    )


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


def dump_progress(rows):
    """Write incremental CSV so we never lose work if a long run crashes."""
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "learning_curves_cnn.csv"), index=False)

    # Also write an agg file (on whatever exists so far)
    if len(df) > 0:
        agg = df.groupby(["model", "N_train"], as_index=False).agg(
            mean_auc_test=("auc_test", "mean"),
            std_auc_test=("auc_test", "std"),
            mean_balacc_test=("balacc_test_at_thr", "mean"),
            std_balacc_test=("balacc_test_at_thr", "std"),
            mean_acc_test=("acc_test_at_thr", "mean"),
            std_acc_test=("acc_test_at_thr", "std"),
            mean_fit_eval_seconds=("fit_eval_seconds", "mean"),
        )
        agg.to_csv(os.path.join(OUT_DIR, "learning_curves_cnn_agg.csv"), index=False)


# -------------------------
# Model
# -------------------------
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Three convolutional layers followed by two fully connected layers.
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)  # 14->7

        self.fc1 = nn.Linear(64 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        logits = self.fc2(x).squeeze(1)
        return logits


@torch.no_grad()
def predict_scores(model, X, device, batch_size=4096):
    model.eval()
    scores = []
    N = X.shape[0]
    for i in range(0, N, batch_size):
        xb = torch.from_numpy(X[i:i+batch_size]).to(device=device, dtype=torch.float32)
        logits = model(xb)
        probs = torch.sigmoid(logits)
        scores.append(probs.detach().cpu().numpy())
    out = np.concatenate(scores, axis=0)
    return out


def train_one_run(X_tr, y_tr, X_val, y_val, device, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = SmallCNN().to(device)
    # pos_weight for imbalance: weight positives (static=1)
    pos = float(np.sum(y_tr == 1))
    neg = float(np.sum(y_tr == 0))
    pos_weight = torch.tensor([neg / max(pos, 1.0)], device=device, dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val_auc = -1.0
    best_epoch = -1
    best_state = None
    bad = 0

    # Make shuffled indices for batching
    N = X_tr.shape[0]
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        perm = np.random.permutation(N)
        total_loss = 0.0
        n_batches = 0

        for i in range(0, N, BATCH_SIZE):
            idx = perm[i:i+BATCH_SIZE]
            xb = torch.from_numpy(X_tr[idx]).to(device=device, dtype=torch.float32)
            yb = torch.from_numpy(y_tr[idx]).to(device=device, dtype=torch.float32)

            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)

            # NaN/inf guard
            if not torch.isfinite(loss):
                return None, float("nan"), -1  # signal failure

            loss.backward()
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            opt.step()

            total_loss += float(loss.detach().cpu().item())
            n_batches += 1

        train_loss = total_loss / max(n_batches, 1)

        # Validation AUC
        s_val = predict_scores(model, X_val, device=device)
        if not np.isfinite(s_val).all():
            return None, float("nan"), -1

        auc_val = roc_auc_score(y_val, s_val)
        star = "  *" if auc_val > best_val_auc else ""
        print(f"    epoch {epoch:02d}: train_loss={train_loss:.4f}  val_auc={auc_val:.4f}{star}")

        if auc_val > best_val_auc:
            best_val_auc = auc_val
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break

    # restore best
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, float(best_val_auc), int(best_epoch)


# -------------------------
# Main
# -------------------------
def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("Device:", device)

    X_train_pool, y_train_pool, X_val, y_val, X_test, y_test = load_arrays()

    print("Loaded arrays:")
    print("  train_pool:", X_train_pool.shape, y_train_pool.shape, "frac_static=", float(np.mean(y_train_pool)))
    print("  val       :", X_val.shape, y_val.shape, "frac_static=", float(np.mean(y_val)))
    print("  test      :", X_test.shape, y_test.shape, "frac_static=", float(np.mean(y_test)))

    # Precompute 4-channel encodings for VAL/TEST once
    X_val_4ch = to_4ch_onehot_14x14(np.asarray(X_val))
    X_test_4ch = to_4ch_onehot_14x14(np.asarray(X_test))
    y_val_np = np.asarray(y_val).astype(np.int8)
    y_test_np = np.asarray(y_test).astype(np.int8)

    rows = []
    t0_global = time.time()

    for N in TRAIN_SIZES:
        for seed in SEEDS:
            print(f"\nN={N} seed={seed}")
            t0 = time.time()

            X_tr, y_tr = subsample_train(X_train_pool, y_train_pool, N, seed=seed)
            y_tr = y_tr.astype(np.int8)
            X_tr_4ch = to_4ch_onehot_14x14(X_tr)

            model, best_val_auc, best_epoch = train_one_run(
                X_tr_4ch, y_tr, X_val_4ch, y_val_np, device=device, seed=seed
            )

            # If training failed (NaN), record and continue
            if model is None or (not np.isfinite(best_val_auc)):
                dt = time.time() - t0
                print("  TRAIN FAILED (NaN/inf encountered). Recording failure and continuing.")
                rows.append(dict(
                    model="cnn_4ch",
                    N_train=int(N),
                    seed=int(seed),
                    auc_val=float("nan"),
                    auc_test=float("nan"),
                    thr_val_bestbal=float("nan"),
                    balacc_val_at_thr=float("nan"),
                    acc_test_at_thr=float("nan"),
                    balacc_test_at_thr=float("nan"),
                    frac_pred_pos_test_at_thr=float("nan"),
                    best_epoch=int(best_epoch),
                    fit_eval_seconds=float(dt),
                    status="failed_nan",
                ))
                dump_progress(rows)
                continue

            # Scores
            s_val = predict_scores(model, X_val_4ch, device=device)
            s_test = predict_scores(model, X_test_4ch, device=device)

            auc_val = roc_auc_score(y_val_np, s_val)
            auc_test = roc_auc_score(y_test_np, s_test)

            thr, bal_val_at_thr = best_threshold_balanced_accuracy(y_val_np, s_val)
            acc_test, bal_test, frac_pos = eval_at_threshold(y_test_np, s_test, thr)

            dt = time.time() - t0
            print(f"  AUC(val)={auc_val:.4f} thr*={thr:.4f} bal(val@thr*)={bal_val_at_thr:.4f}")
            print(f"  AUC(test)={auc_test:.4f} acc(test@thr*)={acc_test:.4f} bal(test@thr*)={bal_test:.4f} "
                  f"frac_pred_pos={frac_pos:.4f} epochs={best_epoch} ({dt:.1f}s)")

            rows.append(dict(
                model="cnn_4ch",
                N_train=int(N),
                seed=int(seed),
                auc_val=float(auc_val),
                auc_test=float(auc_test),
                thr_val_bestbal=float(thr),
                balacc_val_at_thr=float(bal_val_at_thr),
                acc_test_at_thr=float(acc_test),
                balacc_test_at_thr=float(bal_test),
                frac_pred_pos_test_at_thr=float(frac_pos),
                best_epoch=int(best_epoch),
                fit_eval_seconds=float(dt),
                status="ok",
            ))

            dump_progress(rows)  # write incrementally after each run to preserve results

    # Final saves + plots (for completed runs)
    df = pd.DataFrame(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(os.path.join(OUT_DIR, "learning_curves_cnn.csv"), index=False)

    df_ok = df[df["status"] == "ok"].copy()
    if len(df_ok) > 0:
        agg = df_ok.groupby(["model", "N_train"], as_index=False).agg(
            mean_auc_test=("auc_test", "mean"),
            std_auc_test=("auc_test", "std"),
            mean_balacc_test=("balacc_test_at_thr", "mean"),
            std_balacc_test=("balacc_test_at_thr", "std"),
            mean_acc_test=("acc_test_at_thr", "mean"),
            std_acc_test=("acc_test_at_thr", "std"),
            mean_fit_eval_seconds=("fit_eval_seconds", "mean"),
        )
        agg.to_csv(os.path.join(OUT_DIR, "learning_curves_cnn_agg.csv"), index=False)

        save_plot(agg, "auc_test", "ROC-AUC on TEST", os.path.join(FIG_DIR, "cnn_learning_curve_auc_test"))
        save_plot(agg, "balacc_test", "Balanced accuracy on TEST (thr from VAL)", os.path.join(FIG_DIR, "cnn_learning_curve_balacc_test"))
        save_plot(agg, "acc_test", "Accuracy on TEST (thr from VAL)", os.path.join(FIG_DIR, "cnn_learning_curve_acc_test"))

    settings = dict(
        split_dir=SPLIT_DIR,
        train_sizes=TRAIN_SIZES,
        seeds=SEEDS,
        batch_size=BATCH_SIZE,
        max_epochs=MAX_EPOCHS,
        patience=PATIENCE,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        grad_clip_norm=GRAD_CLIP_NORM,
        device=device,
        python_version=platform.python_version(),
        platform=platform.platform(),
        torch_version=torch.__version__,
        numpy_version=np.__version__,
    )
    with open(os.path.join(OUT_DIR, "settings.json"), "w") as f:
        json.dump(settings, f, indent=2)

    print("\nSaved:", os.path.join(OUT_DIR, "learning_curves_cnn.csv"))
    print("Saved:", os.path.join(OUT_DIR, "learning_curves_cnn_agg.csv"))
    print("Saved:", os.path.join(OUT_DIR, "settings.json"))
    print("Saved figures to:", FIG_DIR)
    print(f"Total elapsed: {(time.time() - t0_global)/60:.1f} min")


if __name__ == "__main__":
    main()