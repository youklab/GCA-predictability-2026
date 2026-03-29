import os
import json
import time
import platform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_selection import mutual_info_classif

# -------------------------
# Config
# -------------------------
SPLIT_DIR = "ml_splits_v251209"

OUT_DIR = "ml_results_stepN_nmi"
FIG_DIR = os.path.join(OUT_DIR, "figs")
os.makedirs(FIG_DIR, exist_ok=True)

# Training sizes match those used in Steps 4 and 5 for direct comparability.
TRAIN_SIZES = [10_000, 30_000, 100_000, 300_000, 850_000]
SEEDS = [0, 1, 2]

# Include a shuffled-label control (y permuted) to establish the noise floor.
DO_SHUFFLE_CONTROL = True

# MI estimator randomness
MI_RANDOM_STATE_BASE = 12345


# -------------------------
# Utilities
# -------------------------
def load_arrays():
    X_train_pool = np.load(os.path.join(SPLIT_DIR, "X_train_pool.npy"), mmap_mode="r")
    y_train_pool = np.load(os.path.join(SPLIT_DIR, "y_train_pool.npy"), mmap_mode="r")

    # We don't strictly need val/test for this statistic
    return X_train_pool, y_train_pool


def subsample_train(X, y, n, seed):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y), size=n, replace=False)
    # Make dense in RAM for sklearn MI routine
    return np.asarray(X[idx]), np.asarray(y[idx])


def binary_entropy(p):
    # H in bits
    eps = 1e-12
    p = np.clip(p, eps, 1 - eps)
    return float(-(p * np.log2(p) + (1 - p) * np.log2(1 - p)))


def save_errorbar_plot(df_agg, ycol_mean, ycol_std, ylabel, outbase):
    plt.figure()
    x = np.log10(df_agg["N_train"].values.astype(float))
    y = df_agg[ycol_mean].values
    yerr = df_agg[ycol_std].values
    plt.errorbar(x, y, yerr=yerr, fmt="o-", capsize=3)
    plt.xlabel("log10(N_train)")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outbase + ".png", dpi=300)
    plt.savefig(outbase + ".pdf")
    plt.close()


def compute_featurewise_nmi(X, y, mi_seed):
    """
    Returns:
      H_y_bits
      mi_bits_per_feature: (196,) array, MI in bits
      nmi_per_feature: (196,) array, MI/H(y)
    """
    y = y.astype(np.int8)

    # Convert feature values {1,2,3,4} to {0,1,2,3} (optional but tidy)
    X0 = X.astype(np.int16) - 1

    p = float(np.mean(y))
    H_y = binary_entropy(p)  # bits

    # mutual_info_classif returns MI in nats; convert to bits
    mi_nats = mutual_info_classif(
        X0,
        y,
        discrete_features=True,
        random_state=mi_seed
    )
    mi_bits = mi_nats / np.log(2.0)

    # Normalize by H(y) in bits
    if H_y <= 0:
        nmi = np.zeros_like(mi_bits)
    else:
        nmi = mi_bits / H_y

    return H_y, mi_bits, nmi


# -------------------------
# Main
# -------------------------
def main():
    X_train_pool, y_train_pool = load_arrays()

    print("Loaded arrays:")
    print("  train_pool:", X_train_pool.shape, y_train_pool.shape, "frac_static=", float(np.mean(y_train_pool)))

    os.makedirs(OUT_DIR, exist_ok=True)

    rows = []
    t_global0 = time.time()

    for N in TRAIN_SIZES:
        for seed in SEEDS:
            print(f"\nN={N} seed={seed}")

            t0 = time.time()
            X_tr, y_tr = subsample_train(X_train_pool, y_train_pool, N, seed=seed)
            y_tr = y_tr.astype(np.int8)

            H_y, mi_bits, nmi = compute_featurewise_nmi(
                X_tr, y_tr, mi_seed=MI_RANDOM_STATE_BASE + seed
            )

            # Summaries across the 196 features
            mean_nmi = float(np.mean(nmi))
            max_nmi = float(np.max(nmi))
            p95_nmi = float(np.quantile(nmi, 0.95))

            dt = time.time() - t0
            print(f"  H(y)={H_y:.4f} bits; mean_nMI={mean_nmi:.6f}; p95_nMI={p95_nmi:.6f}; max_nMI={max_nmi:.6f}  ({dt:.1f}s)")

            rows.append(dict(
                condition="real",
                N_train=int(N),
                seed=int(seed),
                frac_static=float(np.mean(y_tr)),
                Hy_bits=float(H_y),
                mean_nmi=float(mean_nmi),
                p95_nmi=float(p95_nmi),
                max_nmi=float(max_nmi),
                wall_seconds=float(dt),
            ))

            if DO_SHUFFLE_CONTROL:
                rng = np.random.default_rng(seed)
                y_shuf = y_tr.copy()
                rng.shuffle(y_shuf)

                H_y_s, mi_bits_s, nmi_s = compute_featurewise_nmi(
                    X_tr, y_shuf, mi_seed=MI_RANDOM_STATE_BASE + 1000 + seed
                )

                mean_nmi_s = float(np.mean(nmi_s))
                max_nmi_s = float(np.max(nmi_s))
                p95_nmi_s = float(np.quantile(nmi_s, 0.95))

                print(f"  [shuffle] mean_nMI={mean_nmi_s:.6f}; p95_nMI={p95_nmi_s:.6f}; max_nMI={max_nmi_s:.6f}")

                rows.append(dict(
                    condition="shuffle_y",
                    N_train=int(N),
                    seed=int(seed),
                    frac_static=float(np.mean(y_shuf)),
                    Hy_bits=float(H_y_s),
                    mean_nmi=float(mean_nmi_s),
                    p95_nmi=float(p95_nmi_s),
                    max_nmi=float(max_nmi_s),
                    wall_seconds=np.nan,
                ))

    df = pd.DataFrame(rows)
    out_csv = os.path.join(OUT_DIR, "featurewise_nmi_learning_curve.csv")
    df.to_csv(out_csv, index=False)
    print("\nSaved:", out_csv)

    # Aggregate across seeds
    agg = df.groupby(["condition", "N_train"], as_index=False).agg(
        mean_mean_nmi=("mean_nmi", "mean"),
        std_mean_nmi=("mean_nmi", "std"),
        mean_p95_nmi=("p95_nmi", "mean"),
        std_p95_nmi=("p95_nmi", "std"),
        mean_max_nmi=("max_nmi", "mean"),
        std_max_nmi=("max_nmi", "std"),
        mean_Hy_bits=("Hy_bits", "mean"),
    )
    out_agg = os.path.join(OUT_DIR, "featurewise_nmi_learning_curve_agg.csv")
    agg.to_csv(out_agg, index=False)
    print("Saved:", out_agg)

    # Plots (real vs shuffle)
    for metric, ylabel in [
        ("mean_nmi", "mean per-cell nMI = mean_j I(X_j;y)/H(y)"),
        ("p95_nmi", "95th percentile per-cell nMI"),
        ("max_nmi", "max per-cell nMI"),
    ]:
        # pivot into separate frames per condition
        for condition in ["real", "shuffle_y"] if DO_SHUFFLE_CONTROL else ["real"]:
            sub = agg[agg["condition"] == condition].copy()
            if len(sub) == 0:
                continue
            save_errorbar_plot(
                sub,
                ycol_mean=(
    "mean_mean_nmi" if metric == "mean_nmi" else
    "mean_p95_nmi"  if metric == "p95_nmi"  else
    "mean_max_nmi"
),
ycol_std=(
    "std_mean_nmi" if metric == "mean_nmi" else
    "std_p95_nmi"  if metric == "p95_nmi"  else
    "std_max_nmi"
),
                ylabel=f"{ylabel}  [{condition}]",
                outbase=os.path.join(FIG_DIR, f"nmi_{metric}_{condition}")
            )

    # Settings dump
    settings = dict(
        split_dir=SPLIT_DIR,
        train_sizes=TRAIN_SIZES,
        seeds=SEEDS,
        do_shuffle_control=DO_SHUFFLE_CONTROL,
        estimator="sklearn.feature_selection.mutual_info_classif(discrete_features=True)",
        normalization="divide by H(y) in bits",
        python_version=platform.python_version(),
        platform=platform.platform(),
        numpy_version=np.__version__,
    )
    out_settings = os.path.join(OUT_DIR, "settings.json")
    with open(out_settings, "w") as f:
        json.dump(settings, f, indent=2)
    print("Saved:", out_settings)

    print("\nSaved figures to:", FIG_DIR)
    print(f"Total elapsed: {(time.time() - t_global0)/60:.1f} min")


if __name__ == "__main__":
    main()