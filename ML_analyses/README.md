# ML_analyses

This folder contains all code and data associated with the machine-learning analysis of predictability in the generalized cellular automaton (GCA). We sought to determine whether the macroscopic fate of the system — static configurations versus dynamic wave patterns — could be predicted from the initial configuration of cell states.

## The central question

Despite the fully deterministic update rules of the GCA, we discovered that we could not predict which of the three macroscopic outcomes (static configuration, rectilinear wave, or spiral wave) a given initial configuration would produce. To test this rigorously, we subjected the system to a deliberate, conservative ML challenge: reduce the problem to binary classification (static vs. dynamic), generate one million independent simulations, and ask whether any of a broad set of supervised learning models could extract predictive signal from the initial cell states.

The answer, across every model class we tested, was no.

## What is in this folder

```
ML_analyses/
├── ML_codes/          # All Python scripts for the ML and NMI analyses
├── data_for_ML/
│   ├── labeled_data_2025_parallel/   # 100 raw data chunks (.npz), one per batch of simulations
│   └── ML_splits/                    # Frozen train/val/test split arrays and metadata
```

## Design philosophy

The ML analysis was designed around three guiding principles:

1. **Make the prediction task as easy as possible.** We reduced the three-class outcome to binary classification (static = 1, dynamic = 0), which is strictly easier than predicting the full pattern class.

2. **Eliminate data as a confounding factor.** We generated 1,000,000 independent simulations and evaluated learning curves spanning nearly three orders of magnitude in training set size (up to 850,000 examples).

3. **Use a frozen, held-out test set.** All reported metrics were computed on a fixed test set of 100,000 configurations never seen during training or threshold calibration, preventing any form of test leakage.

We found that prediction performance saturated at chance-level balanced accuracy and ROC–AUC (≈ 0.5) across all model classes, regardless of training set size. This result is consistent with and supported by a model-agnostic bound derived from normalized mutual information (NMI) analysis, which showed that individual cell states carry no detectable predictive information about the final macroscopic outcome.

## Dependencies

- Python ≥ 3.9
- NumPy, SciPy, Matplotlib, pandas
- scikit-learn (logistic regression, ExtraTrees, histogram-based gradient boosting)
- XGBoost
- PyTorch (MLP and CNN models)

## Reference

Koopmans, Kay, and Youk (2026).

For full methodological details, see **Supplementary Note 2** of the paper.
