import os
import json
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    confusion_matrix, accuracy_score, balanced_accuracy_score
)
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import HistGradientBoostingClassifier


# -------------------------
# Config
# -------------------------
SPLIT_DIR = "ml_splits_v251209"
OUT_DIR = "ml_results_step3"
FIG_DIR = os.path.join(OUT_DIR, "figs")
os.makedirs(FIG_DIR, exist_ok=True)

RANDOM_STATE = 0
N_TRAIN = 850_000  # full training pool

# Save both PNG and PDF for every figure
SAVE_PNG = True
SAVE_PDF = True
PNG_DPI = 300



# -------------------------
# Utility functions
# -------------------------
def load_arrays():
    X_train_pool = np.load(os.path.join(SPLIT_DIR, "X_train_pool.npy"), mmap_mode="r")
    y_train_pool = np.load(os.path.join(SPLIT_DIR, "y_train_pool.npy"), mmap_mode="r")

    X_val = np.load(os.path.join(SPLIT_DIR, "X_val.npy"), mmap_mode="r")
    y_val = np.load(os.path.join(SPLIT_DIR, "y_val.npy"), mmap_mode="r")

    X_test = np.load(os.path.join(SPLIT_DIR, "X_test.npy"), mmap_mode="r")
    y_test = np.load(os.path.join(SPLIT_DIR, "y_test.npy"), mmap_mode="r")
    return X_train_pool, y_train_pool, X_val, y_val, X_test, y_test


def subsample_train(X, y, n, seed=0):
    rng = np.random.default_rng(seed)
    if n > len(y):
        raise ValueError(f"Requested n={n} > train_pool={len(y)}")
    idx = rng.choice(len(y), size=n, replace=False)
    return np.asarray(X[idx]), np.asarray(y[idx])


def best_threshold_balanced_accuracy(y_true, scores):
    """
    Choose threshold that maximizes balanced accuracy on y_true using scores.
    Returns: (thr_best, balacc_best, thr_best_by_J, balacc_at_thrJ)
    """
    fpr, tpr, thr = roc_curve(y_true, scores)
    balacc = 0.5 * (tpr + (1.0 - fpr))
    j = tpr - fpr  # Youden's J; same argmax as balacc
    k_bal = int(np.argmax(balacc))
    k_j = int(np.argmax(j))
    return float(thr[k_bal]), float(balacc[k_bal]), float(thr[k_j]), float(balacc[k_j])


def evaluate_with_threshold(y_true, scores, thr):
    y_pred = (scores >= thr).astype(np.int8)
    acc = accuracy_score(y_true, y_pred)
    bal = balanced_accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)  # [[TN FP],[FN TP]]
    return acc, bal, cm, y_pred


def _save_figure_both(outpath_base_no_ext: str):
    """
    Save current figure as PNG and/or PDF using the same base path.
    Example: base=".../extratrees_roc_test" -> saves .png and .pdf
    """
    if SAVE_PNG:
        plt.savefig(outpath_base_no_ext + ".png", dpi=PNG_DPI, bbox_inches="tight")
    if SAVE_PDF:
        plt.savefig(outpath_base_no_ext + ".pdf", bbox_inches="tight")


def plot_roc(y_true, scores, title, outpath_base_no_ext):
    fpr, tpr, _ = roc_curve(y_true, scores)
    auc = roc_auc_score(y_true, scores)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(title)
    plt.legend(frameon=False)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    _save_figure_both(outpath_base_no_ext)
    plt.close()


def plot_pr(y_true, scores, title, outpath_base_no_ext):
    precision, recall, _ = precision_recall_curve(y_true, scores)
    ap = average_precision_score(y_true, scores)

    plt.figure()
    plt.plot(recall, precision, label=f"AP={ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend(frameon=False)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    _save_figure_both(outpath_base_no_ext)
    plt.close()


def plot_score_hist(y_true, scores, title, outpath_base_no_ext):
    plt.figure()
    s0 = scores[y_true == 0]
    s1 = scores[y_true == 1]
    plt.hist(s0, bins=50, alpha=0.7, label="dynamic (0)")
    plt.hist(s1, bins=50, alpha=0.7, label="static (1)")
    plt.xlabel("Predicted probability for class 1 (static)")
    plt.ylabel("Count")
    plt.title(title)
    plt.legend(frameon=False)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    _save_figure_both(outpath_base_no_ext)
    plt.close()


def save_confusion_matrix(cm, title, outpath_base_no_ext):
    plt.figure()
    plt.imshow(cm)
    plt.title(title)
    plt.xticks([0, 1], ["pred 0", "pred 1"])
    plt.yticks([0, 1], ["true 0", "true 1"])
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")
    plt.xlabel("Prediction")
    plt.ylabel("Truth")
    plt.tight_layout()

    _save_figure_both(outpath_base_no_ext)
    plt.close()


# -------------------------
# Model builders
# -------------------------
def make_models():
    logreg = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ("clf", LogisticRegression(
            max_iter=200,
            solver="lbfgs",
            random_state=RANDOM_STATE,
        ))
    ])

    extratrees = ExtraTreesClassifier(
        n_estimators=400,
        max_depth=None,
        max_features="sqrt",
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    hgb = HistGradientBoostingClassifier(
        max_depth=None,
        learning_rate=0.1,
        max_iter=300,
        random_state=RANDOM_STATE,
    )

    return {
        "logreg_onehot": logreg,
        "extratrees": extratrees,
        "hgb": hgb,
    }


# -------------------------
# Main
# -------------------------
def main():
    X_train_pool, y_train_pool, X_val, y_val, X_test, y_test = load_arrays()

    print("Loaded arrays:")
    print("  train_pool:", X_train_pool.shape, y_train_pool.shape, "frac_static=", float(np.mean(y_train_pool)))
    print("  val       :", X_val.shape, y_val.shape, "frac_static=", float(np.mean(y_val)))
    print("  test      :", X_test.shape, y_test.shape, "frac_static=", float(np.mean(y_test)))

# Use full training pool, or a subsample if N_TRAIN was reduced above.
    if N_TRAIN == len(y_train_pool):
        X_train = np.asarray(X_train_pool)
        y_train = np.asarray(y_train_pool)
    else:
        X_train, y_train = subsample_train(X_train_pool, y_train_pool, N_TRAIN, seed=RANDOM_STATE)

    models = make_models()
    summary = {
        "N_train": int(len(y_train)),
        "val_size": int(len(y_val)),
        "test_size": int(len(y_test)),
        "frac_static_val": float(np.mean(y_val)),
        "frac_static_test": float(np.mean(y_test)),
        "models": {}
    }

    for name, model in models.items():
        print("\n==============================")
        print("Model:", name)

        # Fit
        model.fit(X_train, y_train)

        # Scores (probabilities for class 1)
        if hasattr(model, "predict_proba"):
            s_val = model.predict_proba(X_val)[:, 1]
            s_test = model.predict_proba(X_test)[:, 1]
        else:
            s_val = model.decision_function(X_val)
            s_test = model.decision_function(X_test)

        # Threshold selection on VAL
        thr_best, bal_best, thr_j, bal_j = best_threshold_balanced_accuracy(y_val, s_val)

        # Evaluate on TEST with that threshold
        acc_test, bal_test, cm_test, ypred_test = evaluate_with_threshold(y_test, s_test, thr_best)

        # Threshold-independent metrics
        auc_val = roc_auc_score(y_val, s_val)
        auc_test = roc_auc_score(y_test, s_test)
        ap_val = average_precision_score(y_val, s_val)
        ap_test = average_precision_score(y_test, s_test)

        frac_pos_test = float(np.mean(ypred_test))

        print(f"VAL : AUC={auc_val:.4f}  AP={ap_val:.4f}  best_thr(bal)={thr_best:.4f}  best_bal={bal_best:.4f}")
        print(f"TEST: AUC={auc_test:.4f} AP={ap_test:.4f}  acc={acc_test:.4f}  bal_acc={bal_test:.4f}  frac_pred_pos={frac_pos_test:.4f}")
        print("Confusion matrix TEST [[TN FP],[FN TP]]:")
        print(cm_test)

        # Save figs (both PNG and PDF)
        plot_roc(y_val, s_val, f"{name} ROC (VAL)", os.path.join(FIG_DIR, f"{name}_roc_val"))
        plot_roc(y_test, s_test, f"{name} ROC (TEST)", os.path.join(FIG_DIR, f"{name}_roc_test"))

        plot_pr(y_val, s_val, f"{name} PR (VAL)", os.path.join(FIG_DIR, f"{name}_pr_val"))
        plot_pr(y_test, s_test, f"{name} PR (TEST)", os.path.join(FIG_DIR, f"{name}_pr_test"))

        plot_score_hist(y_val, s_val, f"{name} score hist (VAL)", os.path.join(FIG_DIR, f"{name}_scorehist_val"))
        plot_score_hist(y_test, s_test, f"{name} score hist (TEST)", os.path.join(FIG_DIR, f"{name}_scorehist_test"))

        save_confusion_matrix(cm_test, f"{name} confusion (TEST, thr from VAL)", os.path.join(FIG_DIR, f"{name}_cm_test"))

        summary["models"][name] = {
            "auc_val": float(auc_val),
            "auc_test": float(auc_test),
            "ap_val": float(ap_val),
            "ap_test": float(ap_test),
            "thr_best_val_balacc": float(thr_best),
            "balacc_val_at_thr": float(bal_best),
            "thr_best_val_youdenJ": float(thr_j),
            "balacc_val_at_thr_youdenJ": float(bal_j),
            "acc_test_at_thr": float(acc_test),
            "balacc_test_at_thr": float(bal_test),
            "frac_pred_pos_test_at_thr": float(frac_pos_test),
            "cm_test": cm_test.tolist(),
        }

    out_json = os.path.join(OUT_DIR, "threshold_calibration_summary.json")
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print("\nSaved summary JSON:", out_json)
    print("Saved figures to:", FIG_DIR)
    print(f"Saved formats: {'PNG ' if SAVE_PNG else ''}{'PDF' if SAVE_PDF else ''}".strip())


if __name__ == "__main__":
    main()