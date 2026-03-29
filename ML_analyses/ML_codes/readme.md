# ML_codes

This folder contains all Python scripts used to carry out the machine-learning and mutual-information analyses described in **Supplementary Note 2** of Koopmans, Kay, and Youk (2026). The scripts are numbered in the order they were run. Two upstream scripts handle data preparation before the ML models themselves are trained.

---

## Data preparation scripts

These two scripts must be run first, in order, before any of the `ml_step*` scripts.

### `prepare_splits_from_chunks.py`

**What it does:**
Reads the 100 raw simulation data chunks (`v251209_chunk_0000.npz` through `v251209_chunk_0099.npz`) and constructs a reproducible, frozen train/validation/test split across the full 1,000,000-simulation dataset. The script:

- Inspects all 100 chunk files, verifying consistency of array shapes and data types
- Computes per-chunk class balance (fraction of static outcomes) and saves a diagnostic CSV (`chunk_label_fractions.csv`) and summary JSON (`chunk_label_stats.json`)
- Constructs a global shuffled index space using a fixed random seed (12345), then carves out a **frozen test set** (100,000 simulations) and **validation set** (50,000 simulations) that never change between experiments
- Saves the index arrays (`test_indices.npy`, `val_indices.npy`, `train_indices.npy`) and full split metadata (`split_metadata.json`)
- Materializes the frozen `X_test.npy`, `y_test.npy`, `X_val.npy`, `y_val.npy` arrays on disk

**Why this matters:** We found that fixing a frozen test set from the start was essential to ensure that all model comparisons are evaluated on identical examples and that apparent performance improvements cannot arise from lucky test sampling.

---

### `materialize_train_pool_v3.py`

**What it does:**
Reconstructs the training pool (all simulations not in the test or validation set, up to ~850,000 examples) and materializes it as a single memory-mapped NumPy array (`X_train_pool.npy`, `y_train_pool.npy`). This avoids loading all 100 chunks into RAM at training time and allows the learning-curve scripts to subsample arbitrary training sizes efficiently.

---

## ML analysis scripts

### `ml_step1_baselines.py`

**What it does:**
Establishes trivial classifier baselines before any model fitting: always-predict-dynamic, always-predict-static, and random guessing at the class prior. Reports both raw accuracy and balanced accuracy for each baseline.

**Scientific role:**
We discovered that because the dataset is class-imbalanced (static outcomes are less frequent than dynamic ones), raw accuracy is misleading — a useless classifier can achieve high raw accuracy simply by always predicting the majority class. This script establishes that **balanced accuracy = 0.5 is the correct chance-level reference**, which motivates all subsequent emphasis on balanced accuracy and ROC–AUC as primary metrics.

---

### `ml_step2_learning_curves_tabular.py`

**What it does:**
Generates learning curves for a panel of standard tabular classifiers:
- Logistic regression (with one-hot encoding of the 4-state lattice)
- Extremely Randomized Trees
- Histogram-based gradient boosting

Each model is evaluated at seven training set sizes spanning nearly three orders of magnitude (1,000 to 850,000 examples), with three independent random seeds per training size. All metrics — raw accuracy, balanced accuracy, and ROC–AUC — are computed on the frozen test set.

**What we found:**
We discovered that all three model families produced learning curves that flatlined at chance level across the entire range of training set sizes. No model improved with more data, indicating that the predictive signal was absent rather than merely hard to find with small samples.

---

### `ml_step3_threshold_calibration.py`

**What it does:**
Addresses a potential objection: perhaps the models do carry signal, but the default 0.5 decision threshold is suboptimal for this imbalanced task. This script selects the optimal classification threshold **exclusively on the validation set** (never on the test set), then evaluates the resulting classifier on the frozen test set. It reports threshold-free metrics (ROC–AUC, average precision) and thresholded metrics (accuracy, balanced accuracy, confusion matrix) separately.

**What we found:**
We found that threshold tuning on the validation set provided no rescue: when ROC–AUC is at chance level, no threshold can convert a non-discriminating score distribution into a useful classifier. This step closes a standard objection that an ML-savvy reviewer might raise.

---

### `ml_step4_xgboost_learning_curves.py`

**What it does:**
Repeats the learning-curve analysis using XGBoost — a stronger, nonlinear gradient-boosted tree method that is widely considered a state-of-the-art tabular classifier. One-hot encoding is applied to the 4-state lattice before tree fitting. Early stopping on the validation set is used to prevent overfitting. Class imbalance is handled via `scale_pos_weight`.

**What we found:**
We found that XGBoost, like the simpler tabular models, produced chance-level performance at all training set sizes. Even a well-tuned nonlinear tabular method could not extract predictive signal from the initial configuration.

---

### `ml_step5a_mlp_learning_curves.py`

**What it does:**
Trains a multilayer perceptron (MLP) implemented in PyTorch, using one-hot encoding of the 4-state lattice as input. The MLP is the natural "fully connected deep network" benchmark: it considers all possible nonlinear combinations of cell states but imposes no spatial structure. Training uses `BCEWithLogitsLoss` with `pos_weight` for class imbalance, and early stopping is performed on validation ROC–AUC. Learning curves are evaluated on the frozen test set across increasing training set sizes.

**What we found:**
We discovered that even a high-capacity neural network that can in principle capture arbitrary nonlinear combinations of all 196 cell states could not predict macroscopic fate above chance level.

---

### `ml_step5b_cnn_learning_curves.py`

**What it does:**
Trains a convolutional neural network (CNN) implemented in PyTorch, using a 4-channel one-hot image representation of the 14 × 14 lattice as input. The CNN is given explicit spatial inductive bias — it can detect local patterns and spatial correlations — and is therefore the strongest spatially-aware model in the pipeline.

**Note on geometry:**
The CNN operates on a square 14 × 14 tensor representation rather than an exact triangular-lattice neural architecture; it therefore provides a conservative spatial benchmark rather than a geometry-matched one.

**What we found:**
We discovered that the CNN, despite its spatial inductive bias and high capacity, produced chance-level learning curves indistinguishable from all other model classes.

---

### `ml_stepN_nmi_featurewise_learning_curve.py`

**What it does:**
Computes featurewise normalized mutual information (NMI) between each individual cell's initial state and the binary outcome label (static vs. dynamic). NMI is normalized by the binary entropy of the label distribution, so that NMI = 1 corresponds to perfect predictability and NMI = 0 corresponds to no mutual information. The analysis:

- Evaluates NMI at increasing training set sizes (10,000 to 850,000) to assess whether NMI estimates converge
- Includes a shuffled-label control to establish the noise floor
- Reports the mean and 95th percentile of NMI across all 196 cell positions

**Scientific role:**
Unlike the supervised learning analysis, the NMI computation is **model-agnostic**: it provides an upper bound on how much predictive information any single cell's initial state could contribute. We found that per-cell NMI converged to values indistinguishable from the shuffled control — consistent with and independently confirming the ML learning-curve results.

**Important caveat:**
Featurewise NMI addresses whether individual cell states carry detectable signal; it does not rule out extremely weak or high-order joint statistical dependencies across many cells simultaneously. However, this interpretation is consistent with the overall picture: no model class, regardless of its capacity to combine cell states nonlinearly, extracted signal above chance.

---

## Running the full pipeline

> **Note:** These scripts require that the simulation dataset and train/validation/test splits have already been generated and saved to disk. Run `prepare_splits_from_chunks.py` and `materialize_train_pool_v3.py` first (see sections above). All ML experiments use the same precomputed splits to ensure comparability across model classes.

Run scripts in this order:

```bash
python prepare_splits_from_chunks.py
python materialize_train_pool_v3.py
python ml_step1_baselines.py
python ml_step2_learning_curves_tabular.py
python ml_step3_threshold_calibration.py
python ml_step4_xgboost_learning_curves.py
python ml_step5a_mlp_learning_curves.py
python ml_step5b_cnn_learning_curves.py
python ml_stepN_nmi_featurewise_learning_curve.py
python ml_make_fig_learning_curves.py   # generates final figures
```

The data chunks (`labeled_data_2025_parallel/`) must be present in the working directory before running `prepare_splits_from_chunks.py`.
