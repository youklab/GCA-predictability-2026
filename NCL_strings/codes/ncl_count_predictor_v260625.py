"""
ncl_count_predictor_v260625.py

The NCL-count-only fate predictor: how much of the fate is predictable from the
NCL-string quantities ALONE, as a function of fraction-of-formation f -- the
direct analog of the density-only and single-color predictors, to place NCL
structure on the same ladder.

For each fraction f and each trajectory, extract NCL features at the absolute
frame round(f * Ttrue) (clamped to the trajectory's available range) from the raw
per-frame series in ncl_raw_series.npz. Then train class-balanced 3-way
classifiers (logistic regression + gradient-boosted tree) with a train/test
split, and report per-fate recall + 3-way balanced accuracy vs f.

Two feature sets, as SEPARATE curves:
  (A) abundance only : [ncl_count, total_strings]
  (B) abundance+final: [ncl_count, total_strings, final_pair_on_ncl]

Reads:  ncl_string_ensemble_v260625/ncl_raw_series.npz  (the 2000/fate run)
Writes: ncl_count_predictor_results_v260625/ncl_count_predictor_recall.csv
        + prints balanced-accuracy tables

Usage:
    python ncl_count_predictor_v260625.py
"""

import os
import csv
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score

ROOT = "<PATH_TO_OUTPUT>"
RAW = os.path.join(ROOT, "ncl_string_ensemble_v260625", "ncl_raw_series.npz")
OUT_DIR = os.path.join(ROOT, "ncl_count_predictor_results_v260625")

FRACTIONS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
FATE_NAMES = {1: "static", 2: "spiral", 3: "rectilinear"}
SEED = 0
TEST_FRAC = 0.25

FEATURE_SETS = {
    "abundance":        ["ncl_count", "total_strings"],
    "abundance+final":  ["ncl_count", "total_strings", "finalpair_ncl"],
}


def value_at_fraction(series, T, f):
    """NCL value at absolute frame round(f*T), clamped into the series range.
    series may be empty (e.g. finalpair_ncl absent) -> nan."""
    n = len(series)
    if n == 0 or T <= 0:
        return np.nan
    idx = int(round(f * T))
    idx = max(0, min(idx, n - 1))
    return float(series[idx])


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    z = np.load(RAW, allow_pickle=True)
    fate = z["fate"].astype(int)
    Ttrue = z["Ttrue"].astype(int)
    raw = {
        "ncl_count": z["ncl_count"],
        "total_strings": z["total_strings"],
        "finalpair_ncl": z["finalpair_ncl"],
    }
    n_traj = len(fate)
    print(f"Loaded {n_traj} trajectories: " +
          ", ".join(f"{FATE_NAMES[k]}={int(np.sum(fate==k))}" for k in (1, 2, 3)))

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n_traj)
    ntest = int(n_traj * TEST_FRAC)
    te_idx, tr_idx = perm[:ntest], perm[ntest:]

    rows = []
    for fsname, feats in FEATURE_SETS.items():
        print(f"\n=== feature set: {fsname}  ({', '.join(feats)}) ===")
        print(f"{'f':>5s} | {'model':>8s} | {'balacc':>7s} | static spiral rect")
        for f in FRACTIONS:
            # build feature matrix at this fraction
            X = np.zeros((n_traj, len(feats)), dtype=float)
            for j, feat in enumerate(feats):
                for i in range(n_traj):
                    X[i, j] = value_at_fraction(raw[feat][i], Ttrue[i], f)
            # finalpair_ncl can be nan (series absent); impute with 0 (no final-pair info)
            X = np.nan_to_num(X, nan=0.0)
            Xtr, ytr = X[tr_idx], fate[tr_idx]
            Xte, yte = X[te_idx], fate[te_idx]
            for mname, clf in [
                ("logit", LogisticRegression(max_iter=2000, class_weight="balanced")),
                ("gbtree", HistGradientBoostingClassifier(max_iter=200, max_depth=4,
                                                          learning_rate=0.1, random_state=SEED)),
            ]:
                clf.fit(Xtr, ytr)
                pred = clf.predict(Xte)
                bal = balanced_accuracy_score(yte, pred)
                rec = {}
                for k in (1, 2, 3):
                    m = (yte == k)
                    rec[k] = float(np.mean(pred[m] == k)) if m.any() else float("nan")
                    rows.append([fsname, f, mname, k, FATE_NAMES[k], rec[k], bal])
                print(f"{f:>5.1f} | {mname:>8s} | {bal:>7.3f} | "
                      f"{rec[1]:.2f}  {rec[2]:.2f}  {rec[3]:.2f}")

    csv_path = os.path.join(OUT_DIR, "ncl_count_predictor_recall.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["feature_set", "fraction", "model", "fate", "fate_name", "recall", "balacc_3way"])
        w.writerows(rows)
    print("\nWrote:", csv_path)


if __name__ == "__main__":
    main()
