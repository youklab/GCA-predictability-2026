# Machine learning on initial configurations

This folder contains the codes and data for the machine-learning analysis that asked whether the static-vs-dynamic fate (2-way prediction task) could be predicted directly from its initial configuration.

This README covers only the initial-configuration analysis of 2-way prediction task. Analyses of later configurations, three-fate prediction, and the winding-field representation are documented separately.


## The central question

The GCA is fully deterministic: each initial configuration completely fixes whether the system will eventually form a static configuration, rectilinear wave, or spiral wave. We asked whether this predetermined fate could be predicted directly from the initial cell states without evolving the automaton to the final pattern.

To make this test deliberately conservative, we reduced the three fates to a binary classification problem:

- static configuration = 1
- dynamic pattern, comprising rectilinear and spiral waves = 0

Distinguishing static from dynamic outcomes is less demanding than predicting all three pattern types separately. We generated one million independently initialized simulations and tested whether any of several supervised machine-learning models could detect fate-predictive information in the initial configurations.

None of the tested model classes predicted the binary fate more accurately than chance.


## What is in this folder

```
ML_analyses_on_initialConfig/
├── ML_codes/          # Python scripts for the initial-configuration ML and NMI analyses
├── data_for_ML/
│   ├── labeled_data_2025_parallel/   # 100 raw data chunks (.npz), one per batch of simulations
│   └── ML_splits/                    # Frozen train/val/test split arrays and metadata
```

## Data representation and CNN input

In the stored data files, each 14 × 14 lattice configuration is represented as a flattened vector of 196 cell states encoded as integers in {1, 2, 3, 4}. This flattened vector is a storage format, not the CNN input shape.

Before being supplied to the CNN, each vector was reshaped into a 14 × 14 lattice and one-hot encoded as a four-channel image, with one channel for each cellular state. The resulting CNN input therefore had shape 4 × 14 × 14.

The CNN treated the triangular lattice through this square-array embedding. Its local 3 × 3 receptive field contained all six triangular-lattice nearest neighbors of each cell, together with two array positions that were not triangular-lattice nearest neighbors. The embedding therefore preserved the local information while approximating, rather than exactly encoding, the triangular-lattice geometry.



## Design philosophy

The ML analysis was designed around three guiding principles:

1. **Make the prediction task as easy as possible.** We reduced the three-class outcome to binary classification (static = 1, dynamic = 0), which is strictly easier than predicting the full pattern class.

2. **Eliminate data as a confounding factor.** We generated 1,000,000 independent simulations and evaluated learning curves spanning nearly three orders of magnitude in training set size (up to 850,000 examples).

3. **Use a frozen, held-out test set.** All reported metrics were computed on a fixed test set of 100,000 configurations never seen during training or threshold calibration, preventing any form of test leakage.

Across all tested model classes, balanced accuracy and ROC–AUC saturated near their binary chance levels of 0.5, regardless of training-set size. A complementary normalized mutual information (NMI) analysis found no detectable association between the initial state of any individual cell and the final binary fate. Together, these results indicate that the tested models could not extract practically accessible fate-predictive information from the initial configurations.

## Dependencies

- Python ≥ 3.9
- NumPy, SciPy, Matplotlib, pandas
- scikit-learn (logistic regression, ExtraTrees, histogram-based gradient boosting)
- XGBoost
- PyTorch, for the multilayer perceptron and convolutional neural network

## Reference

Koopmans, Kay, and Youk (2026), *Predictability can be dynamically constructed in deterministic systems*

For full methodological details, see **Supplementary Note 2** of the paper.
