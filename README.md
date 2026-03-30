# GCA-predictability-2026

Code repository for the manuscript:

**"Predictability is dynamically constructed by topological collective modes in deterministic systems"**

Lars Koopmans, Elinor M. Kay, and Hyun Youk

---

## Overview

This repository contains the primary codes, data, outputs, and links to repositories of data used in the study. We investigated a "secrete-and-sense" generalized cellular automaton (GCA) on a two-dimensional triangular lattice with four cell states and exponentially decaying diffusion-weighted coupling. Starting from maximally disordered initial conditions, the system self-organizes into one of three macroscopic outcomes: static configurations, rectilinear waves, or spiral waves.

Each folder contains its own README describing the scripts and their purpose.

---

## Repository structure

```
GCA-predictability-2026/
├── README.md                    ← this file
├── GCA_simulations/             ← GCA simulation engine and main.py for running GCA
├── analyze_topologicalModes/    ← analyses of vortices, NCL strings, pattern-formation times, etc.
├── ML_analyses/                 ← machine-learning tests of predictability from initial configurations
├── clustering_trajectory/       ← graph-based clustering and unique-trajectory analysis

```


---

## Simulation (`simulation/`)

This folder contains the core cellular automaton simulation code, written in Python. The CA operates on a triangular lattice with periodic boundary conditions (toroidal topology). Each cell occupies one of four discrete gene-expression states. At each timestep, every cell simultaneously senses the exponentially decaying diffusion-weighted concentrations secreted by all other cells and updates its state according to a fixed regulatory interaction matrix.

Scripts in this folder implement:

- lattice initialization with controlled spatial disorder
- the CA update rule with diffusion-weighted coupling
- simulation of the single-cell perturbation experiments reported in Fig. 1
- generation of the one-million-run dataset used in the machine-learning analyses

---

## Vortex analysis (`vortex_analysis/`)

This folder contains the Python algorithms used to detect, classify, track, and analyze discrete vortices — the topological collective modes that govern pattern formation in the CA.

We discovered that recoding cell states as discrete phase vectors reveals three classes of vortex cores: +1 vortices (counterclockwise winding), −1 vortices (clockwise winding), and 0 vortices (no net helicity). These vortex structures are invisible in the raw cell-state representation but fully determine the system's dynamical trajectory and final fate.

Scripts in this folder implement:

- connected-component vortex core identification on the triangular lattice with periodic boundary conditions
- winding number calculation via contour integration around each vortex core
- vortex charge classification and tracking across timesteps
- verification of global topological charge neutrality (total winding = 0 at all timesteps, proved analytically in Supplementary Note 5)
- the Brownian particle model of vortex diffusion and annihilation (Supplementary Note 4), including Monte Carlo simulations and comparison with CA dynamics
- non-contractible loop (NCL) string detection and charge analysis

---

## Graph-based clustering and trajectory analysis (`clustering/`)

This folder contains the MATLAB code used for graph-based clustering of CA trajectories and the unique-trajectory collapse analysis. Figures were generated in R.

---

## Machine-learning analyses (`ml/`)

This folder contains the Python machine-learning scripts used to test whether the final macroscopic fate of the CA can be predicted directly from the static initial configuration.

### Scientific goal

The goal of these analyses was not leaderboard-style model optimization. Instead, we sought to ask the most forgiving version of the prediction question: can any supervised learning approach extract predictive signal about final fate from the initial configuration, given virtually unlimited training data and increasingly expressive model classes?

To make the prediction task as easy as possible, we reduced it to binary classification:
- **label = 1**: the run ends in a static configuration
- **label = 0**: the run ends in any dynamic spatial pattern (rectilinear or spiral waves)

All models were trained and evaluated on fixed, precomputed dataset splits derived from one million independent CA simulations:
- **Training pool:** 850,000 samples
- **Validation set:** 50,000 samples
- **Frozen test set:** 100,000 samples (never used during training or model selection)

These splits were used consistently across all ML experiments. Approximately 26.6% of runs yielded static outcomes and 73.4% yielded dynamic outcomes.

### Scripts

**`ml_step1_baselines.py`**  
Computes trivial baselines for the imbalanced binary classification task: always predict dynamic, always predict static, and random guessing at the class prior. This script was used as a sanity check to show that raw accuracy is misleading under class imbalance, motivating the use of balanced accuracy and ROC–AUC as the primary evaluation metrics throughout.

**`ml_step2_learning_curves_tabular.py`**  
Generates learning curves for baseline tabular classifiers trained directly on the initial configuration: logistic regression with one-hot encoding of the four cell states, Extremely Randomized Trees, and histogram-based gradient boosting. Models are trained on subsets of increasing size drawn from the training pool (N = 10³, 3×10³, 10⁴, 3×10⁴, 10⁵, 3×10⁵, 8.5×10⁵) and evaluated on the frozen test set. Three independent random subsamples (seeds) are used per training size. This script provides the primary baseline learning-curve analysis (Supplementary Fig. 5).

**`ml_step3_threshold_calibration.py`**  
Tests whether careful decision-threshold selection can improve predictive performance for the baseline classifiers. Each model is trained on the full training pool to produce continuous probability scores. A threshold is selected exclusively on the validation set to maximize balanced accuracy, then applied unchanged to the frozen test set. This script also generates ROC curves, precision-recall curves, score histograms, and confusion matrices. We found that threshold calibration does not rescue predictability when score rankings are already at chance (Supplementary Fig. 6).

**`ml_step4_xgboost_learning_curves.py`**  
Generates learning curves for an aggressively tuned XGBoost classifier trained on one-hot encoded initial configurations, with validation-based early stopping. This script tests whether a stronger nonlinear boosted-tree model uncovers predictive signal missed by the baseline tabular models (Supplementary Fig. 7).

**`ml_step5a_mlp_learning_curves.py`**  
Generates learning curves for a deep multilayer perceptron (MLP) implemented in PyTorch, trained on one-hot encoded initial configurations with validation-based early stopping. This script tests whether a high-capacity fully connected neural network can extract predictive signal from the initial configuration (Supplementary Fig. 8).

**`ml_step5b_cnn_learning_curves.py`**  
Generates learning curves for a convolutional neural network (CNN) implemented in PyTorch. The initial configuration is represented as a four-channel 2D lattice with one channel per cell state. This script tests whether explicitly incorporating spatial inductive bias improves predictability (Supplementary Fig. 9). Note: the CNN uses a square-grid tensor representation of the lattice and therefore provides a conservative spatial benchmark rather than an architecture exactly matched to the triangular-lattice geometry.

**`ml_stepN_nmi_featurewise_learning_curve.py`**  
Computes a model-agnostic featurewise normalized mutual-information (NMI) analysis as an independent sanity check. For each lattice site, the script estimates the mutual information between that site's initial state and the binary fate label, normalized by the outcome entropy H(y). Results are summarized across sites using the mean and 95th-percentile NMI, and compared against shuffled-label controls. This analysis confirms that individual lattice sites carry no measurable predictive dependence on final fate beyond the null baseline, consistent with the supervised-learning learning curves.

### Interpretation

Across all tested static predictors — linear models, tree ensembles, boosted trees, fully connected deep networks, and convolutional neural networks — prediction performance remained indistinguishable from chance when evaluated by balanced accuracy and ROC–AUC. The NMI analysis likewise showed no detectable site-level predictive dependence beyond shuffled-label baselines. These results demonstrate that the failure of prediction is not a limitation of model capacity or training data, but reflects the genuine absence of accessible predictive features in the initial state.

This null result is consistent with the single-cell perturbation experiments reported in the main text, which show that altering a single cell's state out of 196 redirects macroscopic fate approximately 50% of the time — indicating that the decision boundary between fate classes is so finely structured in configuration space that no static classifier operating on the initial configuration should be expected to resolve it.

---

## Dependencies

### Python
Used for CA simulation, vortex analysis, machine-learning analyses, and NMI computation.
- Python ≥ 3.9
- numpy, scipy, pandas, matplotlib
- scikit-learn
- xgboost
- torch (PyTorch) — for MLP and CNN scripts

Install via:
```bash
pip install numpy scipy pandas matplotlib scikit-learn xgboost torch
```

### MATLAB
Used for graph-based clustering and unique-trajectory analysis (`clustering/`).

### R
Used for figure generation.

---

## Citation

If you use this code, please cite the corresponding manuscript (preprint and journal details to be added upon posting).

---

## Acknowledgements

This work was supported by a grant from the National Institutes of Health (NIH-NIGMS R35 Grant, GM147508).
