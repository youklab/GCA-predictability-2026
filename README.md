# GCA-predictability-2026

Code repository for the manuscript:

**"Predictability is dynamically constructed by topological collective modes in deterministic systems"**

Lars Koopmans, Elinor M. Kay, and Hyun Youk (2026)

---

## Overview

This repository contains the simulation engine, analysis scripts, machine-learning codes, and key data used in the study. We investigated a generalized cellular automaton (GCA) of secrete-and-sense cells on a two-dimensional triangular lattice with four discrete gene-expression states and exponentially decaying diffusion-weighted coupling. Starting from maximally disordered initial conditions, the system self-organizes into one of three macroscopic outcomes: static configurations, rectilinear waves, or spiral waves. Although the dynamics are deterministic, the final outcome cannot be reliably inferred from the initial state alone — neither human inspection nor machine-learning models can do so — yet predictability can instead be constructed dynamically, emerging through the behavior of discrete topological objects (vortices and non-contractible loop strings) that arise during self-organization.

Each subfolder contains its own README with descriptions of individual scripts.

---

## Quick start

To run a single CA simulation and watch the self-organization dynamics unfold in real time:

```bash
cd GCA_simulation
python main.py
```

This runs the GCA from a randomly drawn initial configuration using the parameter values reported in the paper, displays a real-time movie of the evolving lattice from the initial disordered state until the final pattern forms, and prints the pattern type and formation time to the terminal. The final frame remains on screen until the window is closed. The outcome — static configuration, rectilinear wave, or spiral wave — will vary between runs, illustrating that the final outcome cannot be reliably inferred from the initial state alone, the central finding of the paper.

Requires: Python ≥ 3.9, NumPy, SciPy, Matplotlib, pandas, imageio, and a Matplotlib backend that supports interactive display (`TkAgg` is set by default; change this in `main.py` if needed for your system).

---

## Repository structure

```
GCA-predictability-2026/
├── README.md                     ← this file
├── GCA_simulation/               ← CA simulation engine and demo script
├── analyze_topologicalModes/     ← vortex and NCL-string analyses
├── ML_analyses/                  ← machine-learning tests of predictability
└── clustering_trajectory/        ← graph-based trajectory clustering
```

---

## GCA_simulation

Contains the core CA simulation engine (`HY_CA_secrete_and_sense_cells.py`) and a minimal demo script (`main.py`). The CA operates on a triangular lattice with periodic boundary conditions. At each synchronous timestep, every cell senses the concentrations of two diffusible molecules secreted by its neighbors and updates its gene-expression state according to a fixed interaction matrix, threshold matrix, and diffusion lengths. The engine handles lattice construction, parameter initialization, simulation, trajectory analysis, vortex detection, and visualization.

Also contains the the one-million-run dataset for the machine-learning analyses (`/labeled_data_2025_parallel`) and the single-cell perturbation experiments (`perturb_initial_state_oneByone.py`, `Analyze_perturb_initial_state_oneByone.py`).

---

## analyze_topologicalModes

Contains Python scripts for analyzing the discrete vortices and non-contractible loop (NCL) strings that govern pattern formation.

Recoding the four cell states as discrete phase vectors reveals three classes of vortex cores: +1 vortices (counterclockwise winding), −1 vortices (clockwise winding), and 0 vortices (no net winding). These structures are invisible in the raw cell-state (color) representation but their dynamics fully determine macroscopic fate. Scripts characterize vortex peak abundances, the descending-staircase structure of vortex counts over time, the timing between last vortex-pair annihilation and final pattern formation, and the dynamics of NCL strings connecting the final surviving vortex pair — which provide a forward-time predictive signature distinguishing spiral-wave runs from non-spiral runs.

---

## ML_analyses

Contains Python scripts testing whether the final macroscopic outcome of the CA can be predicted from the static initial configuration.

The goal was not model optimization but rather asking the most forgiving version of the prediction question: can any supervised learning approach extract predictive signal from the initial configuration, given virtually unlimited training data and increasingly expressive model classes? The task was reduced to binary classification (static vs. dynamic outcome) and tested across logistic regression, Extremely Randomized Trees, gradient boosting, XGBoost, a deep MLP, and a CNN — all trained and evaluated on fixed splits derived from one million independent CA runs (850,000 training / 50,000 validation / 100,000 frozen test).

Across all model classes, prediction performance remained indistinguishable from chance under balanced accuracy and ROC–AUC. A model-agnostic featurewise normalized mutual information (NMI) analysis confirmed that no individual lattice site carries detectable predictive dependence on final fate beyond a shuffled-label baseline. The limitation is not model capacity or data availability: no practically extractable predictive signal exists in the initial configuration under any of the conditions tested. This is consistent with the single-cell perturbation experiments showing that altering one cell out of 196 redirects macroscopic fate approximately 50% of the time — indicating that the features that make the system's fate predictable have not yet taken shape in the initial configuration.

---

## clustering_trajectory

Contains MATLAB scripts for graph-based clustering of CA trajectories and a visualization of how trajectory structure evolves over time. Analyses use a coarse scalar observable — total vortex-core size as a function of backward time — to ask when trajectories destined for different final pattern types begin to separate. An animated example (`graph_evolution.gif`) illustrates the graph-based clustering analysis.

---

## Dependencies

**Python** (CA simulation, vortex analysis, machine learning):
- Python ≥ 3.9
- numpy, scipy, pandas, matplotlib, imageio
- scikit-learn, xgboost
- torch (PyTorch) — for MLP and CNN scripts only
- tqdm — for `ncl_data_collection_script.py` only

```bash
pip install numpy scipy pandas matplotlib imageio scikit-learn xgboost torch tqdm
```

**MATLAB** — for `clustering_trajectory/`


---

## Citation

If you use this code, please cite the corresponding paper (preprint and journal details to be added upon posting).

---

## Acknowledgements

This work was supported by the National Institutes of Health (NIH-NIGMS R35 Grant GM147508).
