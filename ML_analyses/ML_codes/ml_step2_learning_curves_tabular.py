import os
import json
import time
import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score


# -----------------------------
# User settings
# -----------------------------
SPLIT_DIR = "ml_splits_v251209"
CHUNK_DIR = "labeled_data_2025_parallel"

OUT_DIR = "ml_results_tabular"
os.makedirs(OUT_DIR, exist_ok=True)

# Learning-curve training sizes (must be <= train pool size)
TRAIN_SIZES = [1_000, 3_000, 10_000, 30_000, 100_000, 300_000, 850_000]

# Repeat each point with these seeds (controls subsampling + model randomness)
SEEDS = [0, 1, 2]

# Optional: downsample the majority class to balance training data.
# Set to True to enable; False retains the natural class distribution.
DOWNSAMPLE_TO_BALANCE = False


# -----------------------------
# Utilities: load rows from chunked dataset using global indices
# -----------------------------
def load_split_metadata():
    with open(os.path.join(SPLIT_DIR, "split_metadata.json"), "r") as f:
        return json.load(f)

def build_offsets(chunk_sizes):
    offsets = np.zeros(len(chunk_sizes) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(np.array(chunk_sizes, dtype=np.int64))
    return offsets

def chunk_row_from_global(global_indices, offsets):
    chunk_ids = np.searchsorted(offsets, global_indices, side="right") - 1
    row_ids = global_indices - offsets[chunk_ids]
    return chunk_ids.astype(np.int64), row_ids.astype(np.int64)

def load_rows_from_chunks(global_indices, chunk_files, chunk_sizes, num_cells):
    offsets = build_offsets(chunk_sizes)
    chunk_ids, row_ids = chunk_row_from_global(global_indices, offsets)

    X = np.empty((len(global_indices), num_cells), dtype=np.int8)
    y = np.empty((len(global_indices),), dtype=np.int8)

    order = np.argsort(chunk_ids)
    chunk_ids_sorted = chunk_ids[order]
    row_ids_sorted = row_ids[order]

    start = 0
    n = len(global_indices)
    while start < n:
        c = int(chunk_ids_sorted[start])
        end = start
        while end < n and int(chunk_ids_sorted[end]) == c:
            end += 1

        rows = row_ids_sorted[start:end]
        path = os.path.join(CHUNK_DIR, chunk_files[c])

        with np.load(path) as z:
            data = z["data"]
            labels = z["labels"]
            X_block = data[rows]
            y_block = labels[rows]

        orig_pos = order[start:end]
        X[orig_pos, :] = X_block.astype(np.int8)
        y[orig_pos] = y_block.astype(np.int8)

        start = end

    return X, y

def maybe_balance(X, y, rng):
    if not DOWNSAMPLE_TO_BALANCE:
        return X, y
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    n = min(len(idx0), len(idx1))
    keep0 = rng.choice(idx0, size=n, replace=False)
    keep1 = rng.choice(idx1, size=n, replace=False)
    keep = np.concatenate([keep0, keep1])
    rng.shuffle(keep)
    return X[keep], y[keep]


# -----------------------------
# Models
# -----------------------------
def make_models(num_cells: int):
    models = {}

    # Logistic regression with one-hot encoding of integer states 1..4
    # We shift to 0..3 before passing to the encoder.
    models["logreg_onehot"] = Pipeline([
        ("onehot", OneHotEncoder(
            categories=[np.array([0, 1, 2, 3])] * num_cells,
            sparse_output=True,
            handle_unknown="ignore"
        )),
        ("clf", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs"
        ))
    ])

    # ExtraTrees: 'auto' no longer accepted in recent sklearn; use 'sqrt' for classifier defaults.
    models["extratrees"] = ExtraTreesClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        n_jobs=-1,
        random_state=0,
        class_weight="balanced_subsample"
    )

    # Built-in gradient boosting (strong tabular baseline, no external deps)
    models["hgb"] = HistGradientBoostingClassifier(
        max_depth=10,
        learning_rate=0.05,
        max_iter=500,
        random_state=0
    )

    return models


# -----------------------------
# Main
# -----------------------------
def main():
    meta = load_split_metadata()
    chunk_files = meta["chunk_files"]
    chunk_sizes = meta["chunk_sizes"]
    num_cells = meta["num_cells"]

    train_indices = np.load(os.path.join(SPLIT_DIR, "train_indices.npy"))
    X_test = np.load(os.path.join(SPLIT_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(SPLIT_DIR, "y_test.npy"))

    # For logreg_onehot: shift to 0..3
    X_test_shift = (X_test - 1).astype(np.int8)

    print(f"Train pool size: {len(train_indices)}")
    print(f"Test size:       {len(y_test)}  frac_static={y_test.mean():.4f}")

    models = make_models(num_cells)

    results = []

    settings = {
        "TRAIN_SIZES": TRAIN_SIZES,
        "SEEDS": SEEDS,
        "DOWNSAMPLE_TO_BALANCE": DOWNSAMPLE_TO_BALANCE,
        "models": list(models.keys()),
        "test_frac_static": float(y_test.mean()),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(os.path.join(OUT_DIR, "settings.json"), "w") as f:
        json.dump(settings, f, indent=2)

    for N in TRAIN_SIZES:
        if N > len(train_indices):
            print(f"Skipping N={N} (larger than train pool).")
            continue

        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            subset = rng.choice(train_indices, size=N, replace=False)

            t_load0 = time.time()
            X_train, y_train = load_rows_from_chunks(subset, chunk_files, chunk_sizes, num_cells)
            t_load = time.time() - t_load0

            # shift for one-hot logreg
            X_train_shift = (X_train - 1).astype(np.int8)

            # optional balancing
            X_train_use, y_train_use = maybe_balance(X_train, y_train, rng)
            X_train_shift_use = (X_train_use - 1).astype(np.int8)

            print(f"\nN={N} seed={seed}  loaded in {t_load:.1f}s  frac_static(train)={y_train_use.mean():.4f}")

            for name, model in models.items():
                # set randomness where applicable
                if hasattr(model, "random_state"):
                    try:
                        model.set_params(random_state=seed)
                    except Exception:
                        pass
                if name == "extratrees":
                    model.set_params(random_state=seed)
                if name == "hgb":
                    model.set_params(random_state=seed)

                t0 = time.time()

                if name == "logreg_onehot":
                    model.fit(X_train_shift_use, y_train_use)
                    p = model.predict_proba(X_test_shift)[:, 1]
                    y_pred = (p >= 0.5).astype(np.int8)
                else:
                    model.fit(X_train_use, y_train_use)
                    if hasattr(model, "predict_proba"):
                        p = model.predict_proba(X_test)[:, 1]
                        y_pred = (p >= 0.5).astype(np.int8)
                    else:
                        y_pred = model.predict(X_test).astype(np.int8)
                        p = None

                dt = time.time() - t0

                acc = accuracy_score(y_test, y_pred)
                bacc = balanced_accuracy_score(y_test, y_pred)

                auc = np.nan
                if p is not None:
                    auc = roc_auc_score(y_test, p)

                results.append({
                    "model": name,
                    "N_train": int(N),
                    "seed": int(seed),
                    "train_frac_static": float(y_train_use.mean()),
                    "test_acc": float(acc),
                    "test_bal_acc": float(bacc),
                    "test_auc": float(auc),
                    "train_load_s": float(t_load),
                    "fit_eval_s": float(dt)
                })

                print(f"  {name:14s}  acc={acc:.4f}  bal_acc={bacc:.4f}  auc={auc:.4f}  (fit+eval {dt:.1f}s)")

    df = pd.DataFrame(results)
    out_csv = os.path.join(OUT_DIR, "learning_curves_tabular.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved results to: {out_csv}")

    agg = df.groupby(["model", "N_train"]).agg(
        mean_acc=("test_acc", "mean"),
        sem_acc=("test_acc", lambda x: x.std(ddof=1)/np.sqrt(len(x))),
        mean_bal_acc=("test_bal_acc", "mean"),
        sem_bal_acc=("test_bal_acc", lambda x: x.std(ddof=1)/np.sqrt(len(x))),
        mean_auc=("test_auc", "mean"),
        sem_auc=("test_auc", lambda x: x.std(ddof=1)/np.sqrt(len(x))),
    ).reset_index()

    out_csv2 = os.path.join(OUT_DIR, "learning_curves_tabular_agg.csv")
    agg.to_csv(out_csv2, index=False)
    print(f"Saved aggregated results to: {out_csv2}")


if __name__ == "__main__":
    main()