"""
ncl_predictor_v260625.py

The NCL-count-only fate predictor: the analog of the density-only predictor,
but the features are the per-config NCL counts (pooled + 4 colors) instead of
cell densities.

For each fraction, loads ncl_features_v260625/ncl_features_f{frac}.npz (counts
(N,5), fate (N,)), splits 80/20, trains a class-balanced 3-way logistic
regression and a small gradient-boosted tree (to allow nonlinear/non-monotone
use of the counts), and reports per-fate recall and 3-way balanced accuracy vs f.

This quantifies how much fate information the NCL counts ALONE carry, to compare
against:
  - density-only (~chance, density_only_results_v260619)
  - single-color CNN recall and full-config CNN recall (the structure the CNN reads)

Writes ncl_predictor_results_v260625/ncl_predictor_recall.csv and prints a table.

Usage:
    python ncl_predictor_v260625.py
"""

import os
import csv
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score

LOCAL_ROOT = "<PATH_TO_OUTPUT>"
FEAT_DIR = os.path.join(LOCAL_ROOT, "ncl_features_v260625")
OUT_DIR = os.path.join(LOCAL_ROOT, "ncl_predictor_results_v260625")

FRACTIONS = [0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
FATE_NAMES = {1: "static", 2: "spiral", 3: "rectilinear"}
SEED = 0
TEST_FRAC = 0.2
FEATURE_COLS = ["pooled", "c1", "c2", "c3", "c4"]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    print(f"{'f':>5s} | {'model':>8s} | {'balacc':>7s} | static spiral rect (recall)")
    for frac in FRACTIONS:
        fp = os.path.join(FEAT_DIR, f"ncl_features_f{frac:.2f}.npz")
        if not os.path.isfile(fp):
            print(f"[f={frac:.2f}] MISSING {fp}"); continue
        with np.load(fp) as z:
            Xc = z["counts"].astype(np.float64)
            fate = z["fate"].astype(int)
        rng = np.random.default_rng(SEED)
        n = Xc.shape[0]
        perm = rng.permutation(n)
        ntest = int(n * TEST_FRAC)
        te, tr = perm[:ntest], perm[ntest:]
        Xtr, ytr = Xc[tr], fate[tr]
        Xte, yte = Xc[te], fate[te]

        for name, clf in [
            ("logit", LogisticRegression(max_iter=2000, class_weight="balanced")),
            ("gbtree", HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1,
                                                      max_depth=4, random_state=SEED)),
        ]:
            clf.fit(Xtr, ytr)
            pred = clf.predict(Xte)
            bal = balanced_accuracy_score(yte, pred)
            recalls = {}
            for k in (1, 2, 3):
                m = (yte == k)
                recalls[k] = float(np.mean(pred[m] == k)) if m.any() else float("nan")
                rows.append([frac, name, k, FATE_NAMES[k], recalls[k], bal])
            print(f"{frac:>5.2f} | {name:>8s} | {bal:>7.3f} | "
                  f"{recalls[1]:.2f}  {recalls[2]:.2f}  {recalls[3]:.2f}")

    csv_path = os.path.join(OUT_DIR, "ncl_predictor_recall.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["fraction", "model", "fate", "fate_name", "recall", "balacc_3way"])
        w.writerows(rows)
    print("\nWrote:", csv_path)


if __name__ == "__main__":
    main()
