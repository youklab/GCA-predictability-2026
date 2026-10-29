# GCA-predictability-2026

Code repository for the paper:

**"Predictability can be dynamically constructed in deterministic systems"**

Lars Koopmans, Elinor M. Kay, and Hyun Youk (2026)

Preprint (April 2026 (v1); Aug. 2026 (v2)): [arXiv:2604.01088v2](https://arxiv.org/abs/2604.01088v2)

Data: [10.5281/zenodo.21728102](https://doi.org/10.5281/zenodo.21728102)
Code archive: [10.5281/zenodo.21754318](https://doi.org/10.5281/zenodo.21754318)

---

## Overview

This repository contains the simulation engine, analysis scripts, machine-learning codes, and
key data used in the study. We investigated a generalized cellular automaton (GCA) of
secrete-and-sense cells on a two-dimensional triangular lattice with four discrete
gene-expression states and exponentially decaying diffusion-weighted coupling. Starting from
maximally disordered initial conditions, the system self-organizes into one of three
macroscopic outcomes: static configurations, rectilinear waves, or spiral waves.

Although the dynamics are deterministic, so that the initial configuration completely fixes the
outcome, that outcome cannot be reliably inferred from the initial configuration by any
practical means we tested. Predictability is instead constructed as the system evolves,
through discrete topological objects that appear during self-organization: vortices,
non-contractible-loop strings, and the winding field.

Each subfolder has its own README describing the scripts it contains.

---

## Where to start

| If you want to | go to |
|---|---|
| watch the automaton self-organize | `GCA_simulation/` and run `main.py` |
| see whether the fate can be predicted from the initial state | `ML_analyses_on_initialConfigs/` |
| see when along a trajectory the fate becomes predictable | `ML_along_trajectory/` |
| see what carries that predictability | `winding_field/` |
| see the sparse, late-time version of the same topology | `NCL_strings/` |
| see the vortex analyses | `analyze_topologicalModes/` |

---

## Quick start

To run a single simulation and watch the self-organization unfold:

```bash
cd GCA_simulation
python main.py
```

This runs the automaton from a randomly drawn initial configuration using the parameter values
reported in the paper, displays a movie of the evolving lattice, and prints the final pattern
type and the timestep at which it formed. The final frame stays on screen until you close the
window.

The outcome varies from run to run, which is the point: it illustrates that the outcome cannot
be reliably inferred from the initial configuration, the central finding of the paper.

Requires Python 3.9 or later with NumPy, SciPy, Matplotlib, pandas and imageio, and a
Matplotlib backend supporting interactive display (`TkAgg` is set by default; change it at the
top of `main.py` if your system needs something else).

**A note on timing.** The simulation runs until the pattern forms, up to a limit of 60,000
timesteps set at the top of `main.py`. Static and rectilinear-wave outcomes usually form within
a few hundred to a few thousand timesteps; spiral-wave outcomes can take much longer. A single
run typically completes in 10 to 60 seconds. The movie is capped at the first 300 timesteps,
because that is where the topological self-organization happens and a full trajectory would be
far too long to watch; raise `max_frames_to_show` in `main.py` to see more.

---

## Repository structure

```
GCA-predictability-2026/
|-- README.md                        <- this file
|-- GCA_simulation/                  <- simulation engine and demo script
|-- ML_analyses_on_initialConfigs/   <- can the fate be predicted from t = 0 alone?
|-- ML_along_trajectory/             <- when does the fate become predictable?
|-- winding_field/                   <- the field that carries the predictability
|-- NCL_strings/                     <- non-contractible-loop strings
|-- analyze_topologicalModes/        <- vortex analyses and perturbation experiments
|-- clustering_trajectory/           <- graph-based trajectory clustering (MATLAB)
`-- movies/                          <- movies of the automaton self-organizing
```

---

## The folders

### GCA_simulation

The simulation engine (`HY_CA_secrete_and_sense_cells.py`) and the demo script (`main.py`).
The automaton runs on a triangular lattice with periodic boundary conditions. At each
synchronous timestep every cell senses the concentrations of two diffusible molecules secreted
by its neighbours and updates its gene-expression state according to a fixed interaction
matrix, threshold matrix and set of diffusion lengths. The engine handles lattice construction,
simulation, trajectory analysis, vortex detection and visualization.

### ML_analyses_on_initialConfigs

Tests whether the outcome can be predicted from the initial configuration alone. To make the
test as forgiving as possible the task was reduced to a binary one, static versus dynamic, and
tried across logistic regression, extremely randomized trees, gradient boosting, XGBoost, a
deep multilayer perceptron and a convolutional network, all trained on splits derived from one
million independent runs.

No model class beat chance. A model-agnostic mutual-information analysis separately found that
no individual lattice site carries detectable dependence on the outcome. This folder also holds
the one-million-run labelled dataset used for those analyses.

### ML_along_trajectory

Asks a different question: not whether the outcome can be predicted from the start, but **when**
along a trajectory it becomes predictable. A single snapshot is taken from a chosen point of
each trajectory and handed to a model, and that point is slid from the initial configuration to
the moment the pattern forms.

Six model classes were tested, including a transformer and a neural cellular automaton. All
behave the same way: chance at the initial configuration, progressively more accurate on later
configurations, near-perfect once the pattern has formed.

### winding_field

The winding field labels every cell according to how the connected same-state region it belongs
to wraps around the torus. It is computed deterministically from the configuration, so it adds
no information; what it changes is how directly the global topology can be read. A network
given only the winding field recovers nearly all the accuracy obtainable from the full
configuration, which identifies it as the carrier of the emerging predictability.

This folder contains the extraction code, the time-series code, the network comparisons, an
ablation testing which parts of the field matter, and validation scripts.

### NCL_strings

A non-contractible-loop string is a path of same-state cells that wraps around the torus and
therefore cannot be shrunk to a point. These strings are the sparse, late-time instance of the
same topology the winding field records densely. Their disappearance progressively announces
the static and rectilinear-wave outcomes, while their continued presence cannot establish that
they will persist, which is why spiral-wave fate resolves only near pattern formation.

### analyze_topologicalModes

Analyses of the discrete vortices. Recoding the four cell states as discrete phase vectors
reveals three classes of vortex core: +1 (counterclockwise winding), -1 (clockwise), and 0 (no
net winding). These are invisible in the raw cell-state representation. Scripts here
characterize peak vortex abundance, the staircase structure of vortex counts over time, the
interval between the last vortex-pair annihilation and pattern formation, and the single-cell
perturbation experiments.

### clustering_trajectory

MATLAB scripts for graph-based clustering of trajectories, using total vortex-core size as a
function of backward time to ask when trajectories headed for different outcomes begin to
separate. `graph_evolution.gif` animates the result.

### movies

Movies of the automaton self-organizing, in both the colour representation (four cell states as
four colours) and the phase-field representation (non-white cells marking vortex cores). Two
larger movies are hosted at [youklab.org](https://www.youklab.org/koopmans_data.html).

---

## Data

Small result files are included in this repository so that figures can be checked or replotted
without rerunning anything. The large datasets are not: the trajectory splits and the extracted
topological fields run to several gigabytes, beyond what GitHub is intended to hold.

Those are deposited at [10.5281/zenodo.21728102](https://doi.org/10.5281/zenodo.21728102), and
everything can also be regenerated from the code here.

---

## Running the analysis code

The analysis scripts were written against specific local paths. In this repository those have
been replaced by placeholders that you must set before running anything:

| placeholder | what it should point to |
|---|---|
| `<PATH_TO_SPLITS>` | the trajectory splits, from Zenodo or regenerated locally |
| `<PATH_TO_OUTPUT>` | wherever you want results written |
| `<PATH_TO_CODE>` | `winding_field/codes`, which holds the module the topology scripts import |

They appear near the top of each script. The README in each folder says which ones that folder
needs. The demo in `GCA_simulation` needs none of this and runs as-is.

---

## Dependencies

```bash
pip install numpy scipy pandas matplotlib imageio scikit-learn xgboost torch tqdm
```

PyTorch is needed only for the neural-network scripts, and MATLAB only for
`clustering_trajectory/`. Tested on macOS with Python 3.9 to 3.14 and on Ubuntu 22.04;
MATLAB R2025a with the Parallel Computing Toolbox for the clustering code.

No specialized hardware is needed for the demo or for individual analyses. The large-scale
machine-learning experiments benefit from a GPU and 32 GB or more of RAM, but neither is
required to reproduce any single result.

On a machine with a working scientific-Python stack, installing the dependencies takes a few
minutes; a fresh install may take 10 to 15, mostly PyTorch.

---

## Running with custom parameters

Lattice size, interaction matrix `M`, threshold matrix `K`, secretion constants `C`, diffusion
lengths `LAMB`, cell radius `RCELL`, lattice spacing `A0` and initial ON-fractions `P0` can all
be changed at the top of `main.py`. The values used throughout the paper are:

```python
RCELL = 0.2
A0    = 1.5
M     = [[1, 1], [-1, 0]]
K     = [[3, 8.465876331385257], [11, 4]]
C     = [18, 16]
LAMB  = [1.0, 1.2]
P0    = [0.5, 0.55]
LATTICE_SIZE = 14
```

Changing `LATTICE_SIZE` runs the automaton on a different lattice; the paper reports analyses
up to L = 100. Changing `M`, `K`, `C` or `LAMB` runs it under a different gene-regulatory
circuit or different signalling parameters.

---

## Citation

If you use this code, please cite the corresponding paper ([arXiv:2604.01088v2](https://arxiv.org/abs/2604.01088v2))
the data deposit, [10.5281/zenodo.21728102](https://doi.org/10.5281/zenodo.21728102), and the code archive, [10.5281/zenodo.21754318](https://doi.org/10.5281/zenodo.21754318).

---

## Acknowledgements

Supported by the National Institutes of Health (NIH-NIGMS R35 Grant GM147508).
