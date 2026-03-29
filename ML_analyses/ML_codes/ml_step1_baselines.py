import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, roc_auc_score,
    confusion_matrix, classification_report
)

SPLIT_DIR = "ml_splits_v251209"

X_test = np.load(f"{SPLIT_DIR}/X_test.npy")
y_test = np.load(f"{SPLIT_DIR}/y_test.npy")
X_val  = np.load(f"{SPLIT_DIR}/X_val.npy")
y_val  = np.load(f"{SPLIT_DIR}/y_val.npy")

def summarize_split(name, y):
    frac1 = y.mean()
    print(f"{name}: n={len(y)}  frac_static(label=1)={frac1:.4f}  frac_dynamic(label=0)={1-frac1:.4f}")

def eval_predictions(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n{name}")
    print(f"  accuracy         = {acc:.4f}")
    print(f"  balanced_accuracy= {bacc:.4f}")
    print("  confusion matrix [ [TN FP], [FN TP] ]:")
    print(cm)

print("Shapes / dtypes:")
print("X_test:", X_test.shape, X_test.dtype, "y_test:", y_test.shape, y_test.dtype)
print("X_val :", X_val.shape,  X_val.dtype,  "y_val :", y_val.shape,  y_val.dtype)

summarize_split("VAL ", y_val)
summarize_split("TEST", y_test)

# Baseline 1: always predict dynamic (0)
y_pred_all0_val  = np.zeros_like(y_val)
y_pred_all0_test = np.zeros_like(y_test)

eval_predictions("Baseline: always predict dynamic (0) on VAL",  y_val,  y_pred_all0_val)
eval_predictions("Baseline: always predict dynamic (0) on TEST", y_test, y_pred_all0_test)

# Baseline 2: always predict static (1)
y_pred_all1_val  = np.ones_like(y_val)
y_pred_all1_test = np.ones_like(y_test)

eval_predictions("Baseline: always predict static (1) on VAL",  y_val,  y_pred_all1_val)
eval_predictions("Baseline: always predict static (1) on TEST", y_test, y_pred_all1_test)

# Baseline 3: random guess with correct class prior from VAL
p1 = y_val.mean()
rng = np.random.default_rng(0)
y_pred_rand_val  = (rng.random(len(y_val))  < p1).astype(y_val.dtype)
y_pred_rand_test = (rng.random(len(y_test)) < p1).astype(y_test.dtype)

eval_predictions("Baseline: random guess with prior p1(VAL) on VAL",  y_val,  y_pred_rand_val)
eval_predictions("Baseline: random guess with prior p1(VAL) on TEST", y_test, y_pred_rand_test)

