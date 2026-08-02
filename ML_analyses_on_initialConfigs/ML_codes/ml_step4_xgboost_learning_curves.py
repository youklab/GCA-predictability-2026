# ml_step4_xgboost_learning_curves.py
# XGBoost learning curves (one-hot encoded lattice states) using xgb.train + DMatrix
# Compatible with older xgboost sklearn APIs that lack early_stopping_rounds/callbacks in XGBClassifier.fit().
#
# Outputs:
#   ml_results_step4_xgb/learning_curves_xgb.csv
#   ml_results_step4_xgb/learning_curves_xgb_agg.csv
#   ml_results_step4_xgb/figs/*.png + *.pdf
#   ml_results_step4_xgb/settings.json

import os
import json
import time
import platform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, balanced_accuracy_score

try:
    import xgboost as xgb
except ImportError as e:
    raise SystemExit(
        "xgboost is not installed in this venv.\n"
        "Install with:  pip install xgboost\n"
        "Then re-run this script."
    ) from e


# -------------------------
# Config
# -------------------------
SPLIT_DIR = "ml_splits_v251209"

OUT_DIR = "ml_results_step4_xgb"
FIG_DIR = os.path.join(OUT_DIR, "figs")
os.makedirs(FIG_DIR, exist_ok=True)

TRAIN_SIZES = [10_000, 30_000, 100_000, 300_000, 850_000]
SEEDS = [0, 1, 2]

# Large boosting budget; early stopping on validation AUC will terminate training before this limit in practice.

NUM_BOOST_ROUND = 8000
EARLY_STOPPING_ROUNDS = 200

# XGBoost core params (binary classification)
XGB_PARAMS = dict(
    objective="binary:logistic",
    eval_metric="auc",
    learning_rate=0.03,
    max_depth=6,
    min_child_weight=1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    reg_alpha=0.0,
    gamma=0.0,
    tree_method="hist",      # CPU-friendly
    nthread=-1,              # core API uses nthread rather than n_jobs
)

# One-hot encoder yields 196*4 = 784 sparse features
ENCODER = OneHotEncoder(handle_unknown="ignore", sparse_output=True)


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
    return np.asarray(X[idx]), np.asarray(y[idx]).astype(np.int8)


def best_threshold_balanced_accuracy(y_true, scores):
    fpr, tpr, thr = roc_curve(y_true, scores)
    balacc = 0.5 * (tpr + (1.0 - fpr))
    k = int(np.argmax(balacc))
    return float(thr[k]), float(balacc[k])


def eval_at_threshold(y_true, scores, thr):
    y_pred = (scores >= thr).astype(np.int8)
    acc = accuracy_score(y_true, y_pred)
    bal = balanced_accuracy_score(y_true, y_pred)
    frac_pos = float(np.mean(y_pred))
    return float(acc), float(bal), frac_pos


def save_plot(df_agg, metric, ylabel, outbase):
    """
    df_agg must contain:
      - N_train
      - mean_<metric>
      - std_<metric>
    """
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


def train_xgb_with_early_stopping(X_tr_oh, y_tr, dval, seed, scale_pos_weight):
    """
    Train booster with early stopping using xgb.train.
    Returns (booster, best_iteration_int_or_None).
    """
    params = dict(XGB_PARAMS)
    params["seed"] = int(seed)
    params["scale_pos_weight"] = float(scale_pos_weight)

    dtrain = xgb.DMatrix(X_tr_oh, label=y_tr)

    evals = [(dval, "val")]
    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        evals=evals,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose_eval=False
    )

    best_iter = None
    # Depending on version, booster has best_iteration or best_ntree_limit
    if hasattr(booster, "best_iteration") and booster.best_iteration is not None:
        best_iter = int(booster.best_iteration)
    elif hasattr(booster, "best_ntree_limit") and booster.best_ntree_limit is not None:
        try:
            best_iter = int(booster.best_ntree_limit)
        except Exception:
            best_iter = None

    return booster, best_iter


def predict_proba_booster(booster, X_oh):
    d = xgb.DMatrix(X_oh)
    # For binary:logistic, prediction is P(y=1)
    return booster.predict(d)


# -------------------------
# Main
# -------------------------
def main():
    X_train_pool, y_train_pool, X_val, y_val, X_test, y_test = load_arrays()

    print("Loaded arrays:")
    print("  train_pool:", X_train_pool.shape, y_train_pool.shape, "frac_static=", float(np.mean(y_train_pool)))
    print("  val       :", X_val.shape, y_val.shape, "frac_static=", float(np.mean(y_val)))
    print("  test      :", X_test.shape, y_test.shape, "frac_static=", float(np.mean(y_test)))

    # Fit encoder on TRAIN ONLY (small slice is sufficient: values are {1,2,3,4})
    fit_slice = np.asarray(X_train_pool[:50_000])
    ENCODER.fit(fit_slice)

    # Transform VAL/TEST once
    X_val_oh = ENCODER.transform(np.asarray(X_val))
    X_test_oh = ENCODER.transform(np.asarray(X_test))
    y_val_np = np.asarray(y_val).astype(np.int8)
    y_test_np = np.asarray(y_test).astype(np.int8)

    # DMatrix for val (reused across runs)
    dval = xgb.DMatrix(X_val_oh, label=y_val_np)

    rows = []
    t_global0 = time.time()

    for N in TRAIN_SIZES:
        for seed in SEEDS:
            print(f"\nN={N} seed={seed}")
            t0 = time.time()

            X_tr, y_tr = subsample_train(X_train_pool, y_train_pool, N, seed=seed)
            X_tr_oh = ENCODER.transform(X_tr)

            pos = float(np.sum(y_tr == 1))
            neg = float(np.sum(y_tr == 0))
            scale_pos_weight = (neg / pos) if pos > 0 else 1.0

            booster, best_iter = train_xgb_with_early_stopping(
                X_tr_oh, y_tr, dval, seed=seed, scale_pos_weight=scale_pos_weight
            )

            s_val = predict_proba_booster(booster, X_val_oh)
            s_test = predict_proba_booster(booster, X_test_oh)

            thr, bal_val_at_thr = best_threshold_balanced_accuracy(y_val_np, s_val)

            auc_val = roc_auc_score(y_val_np, s_val)
            auc_test = roc_auc_score(y_test_np, s_test)

            acc_test, bal_test, frac_pos_test = eval_at_threshold(y_test_np, s_test, thr)

            dt = time.time() - t0
            print(f"  AUC(val)={auc_val:.4f}  thr*={thr:.4f}  bal(val@thr*)={bal_val_at_thr:.4f}")
            print(f"  AUC(test)={auc_test:.4f} acc(test@thr*)={acc_test:.4f} bal(test@thr*)={bal_test:.4f} "
                  f"frac_pred_pos={frac_pos_test:.4f}  best_iter={best_iter}  ({dt:.1f}s)")

            rows.append(dict(
                model="xgboost_onehot",
                N_train=int(N),
                seed=int(seed),
                frac_static_train=float(np.mean(y_tr)),
                scale_pos_weight=float(scale_pos_weight),
                best_iteration=None if best_iter is None else int(best_iter),

                auc_val=float(auc_val),
                thr_val_bestbal=float(thr),
                balacc_val_at_thr=float(bal_val_at_thr),

                auc_test=float(auc_test),
                acc_test_at_thr=float(acc_test),
                balacc_test_at_thr=float(bal_test),
                frac_pred_pos_test_at_thr=float(frac_pos_test),

                fit_eval_seconds=float(dt),
            ))

    df = pd.DataFrame(rows)
    os.makedirs(OUT_DIR, exist_ok=True)

    out_csv = os.path.join(OUT_DIR, "learning_curves_xgb.csv")
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

    out_agg = os.path.join(OUT_DIR, "learning_curves_xgb_agg.csv")
    agg.to_csv(out_agg, index=False)
    print("Saved:", out_agg)

    # Plots (PNG+PDF)
    save_plot(agg, "auc_test", "ROC-AUC on TEST", os.path.join(FIG_DIR, "xgb_learning_curve_auc_test"))
    save_plot(agg, "balacc_test", "Balanced accuracy on TEST (thr from VAL)", os.path.join(FIG_DIR, "xgb_learning_curve_balacc_test"))
    save_plot(agg, "acc_test", "Accuracy on TEST (thr from VAL)", os.path.join(FIG_DIR, "xgb_learning_curve_acc_test"))

    # Settings dump
    settings = dict(
        split_dir=SPLIT_DIR,
        train_sizes=TRAIN_SIZES,
        seeds=SEEDS,
        xgb_params=XGB_PARAMS,
        num_boost_round=NUM_BOOST_ROUND,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        encoder="OneHotEncoder(handle_unknown='ignore', sparse_output=True) fit on train_pool[:50k]",
        python_version=platform.python_version(),
        platform=platform.platform(),
        xgboost_version=getattr(xgb, "__version__", "unknown"),
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