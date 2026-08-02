# Machine learning along the trajectory

This folder contains the machine-learning code and results for the analyses that ask
**when** the final pattern of a trajectory becomes predictable, rather than whether it can
be predicted from the initial configuration alone.

For the analyses restricted to initial configurations only, see
`../ML_analyses_on_initialConfigs`.

---

## The idea in one paragraph

Every trajectory of the automaton ends in one of three fates: a static configuration, a
rectilinear wave, or a spiral wave. That fate is fixed the moment the initial configuration
is set, because the automaton is deterministic. We therefore take a single snapshot of the
lattice from a chosen point along each trajectory, hand only that snapshot to a machine-learning
model, and ask the model to name the fate. Sliding the snapshot from the start of the
trajectory to the moment the pattern forms shows how the accuracy changes.

The position of the snapshot is set by a number **f** between 0 and 1:

| f | snapshot taken at |
|---|---|
| 0.00 | the initial configuration, before any update |
| 0.50 | halfway to pattern formation |
| 1.00 | the moment the final pattern forms |

Concretely, a trajectory that takes `T` timesteps to reach its attractor is sampled at
timestep `round(f * T)`. The median `T` in this system is about 1,000 timesteps.

A model that guesses at random scores 1/3. We report **balanced accuracy**, which averages
the accuracy over the three fates separately, so that a model cannot score well simply by
always naming the most common fate.

---

## The models

No prior machine-learning background is assumed below.

- **Logistic regression** and **gradient-boosted trees (XGBoost)**. Standard classifiers that
  treat the 196 cells as an unordered list of features, with no notion of which cells are
  neighbours. Run by `codes/run_tabular_sweep.py`.

- **Multilayer perceptron (MLP)**. A fully connected neural network. Also has no built-in
  notion of neighbours. Run by `codes/run_mlp_gpu.py`.

- **Convolutional neural network (CNN)**. Scans the lattice with small filters, so it knows
  which cells are adjacent. Uses circular padding, meaning it can see across the periodic
  boundary of the torus. Run by `codes/stress_depth.py --model cnn`.

- **Neural cellular automaton (NCA)**. A learnable cellular automaton: one small update rule,
  applied to the lattice over and over. The number of applications is called the *unroll* count.
  Because the same rule is reused every time, running it longer costs no extra parameters, so
  the unroll count measures how long the network computes rather than how large it is.
  Run by `codes/stress_depth.py --model nca`.

- **Transformer**. The architecture underlying modern large language models. Every cell can
  consult every other cell directly. Here attention depends only on the separation between two
  cells measured across the torus, which makes the network give the same answer if the whole
  configuration is shifted bodily across the lattice. Run by `codes/stress_transformer.py`.

- **Hidden Markov model (HMM)**. A classical probabilistic model for sequences. Because the
  lattice is two-dimensional, it is first read out as a sequence of 196 symbols. One model is
  fitted per fate, and a configuration is assigned to whichever model finds it most likely.
  Run by `codes/stress_hmm.py`.

---

## Before you run anything

**1. Python packages.** Python 3.11 was used. Install:

```bash
pip install numpy scikit-learn matplotlib torch xgboost
```

PyTorch will use an NVIDIA GPU (CUDA) or an Apple Silicon GPU (MPS) automatically if one is
present, and fall back to the CPU otherwise. The CPU will work but will be slow for the
transformer and for the deeper neural cellular automata.

**2. The trajectory data are not in this repository.** The snapshot datasets are several
gigabytes, which exceeds what GitHub is intended to hold. They can be regenerated from the
simulation code in `../GCA_simulation`, or obtained from the data repository cited in the
Data Availability statement of the paper.

The scripts expect one folder per value of f, laid out like this:

```
<PATH_TO_SPLITS>/
  ml_splits_v260529_Ttrue_f0.00/
      X_train_pool.npy     (N x 196 array of cell states, values 1-4)
      X_val.npy
      X_test.npy
      meta_train_pool.npz  (contains final_pattern_type, values 1-3, and T)
      meta_val.npz
      meta_test.npz
  ml_splits_v260529_Ttrue_f0.10/
  ...
  ml_splits_v260529_Ttrue_f1.00/
```

Fate labels are `1 = static`, `2 = spiral wave`, `3 = rectilinear wave`.

**3. Set the paths.** Every script has two placeholders near the top:

```python
SPLITS = r"<PATH_TO_SPLITS>"      # where the data folders above live
OUT    = r"<PATH_TO_RESULTS>"     # where results should be written
```

Replace both before running. Nothing will work until you do.

---

## Running a model

Each run trains one model on one value of f and writes one small CSV of results.

```bash
# convolutional network, 4 layers, on initial configurations
python codes/stress_depth.py --tag cnn_f000_d4 --model cnn --depth 4 --frac 0.00

# the same network, on snapshots from halfway along the trajectory
python codes/stress_depth.py --tag cnn_f050_d4 --model cnn --depth 4 --frac 0.50

# neural cellular automaton, 32 update steps, on initial configurations
python codes/stress_depth.py --tag nca_f000_u32 --model nca --depth 32 --frac 0.00

# multilayer perceptron, 3 hidden layers, at the moment of pattern formation
python codes/stress_depth.py --tag mlp_f100_d3 --model mlp --depth 3 --frac 1.00

# transformer, on initial configurations
python codes/stress_transformer.py --tag tf_f000 --frac 0.00 --pos rel --stem 2

# hidden Markov model, 16 hidden states, serpentine reading order
python codes/stress_hmm.py --tag hmm_f000_K16 --frac 0.00 --n_states 16 --scan serpentine
```

Options shared by the neural-network scripts:

| option | meaning |
|---|---|
| `--frac` | which f to use, matching the data folder name |
| `--depth` | convolutional layers, unroll steps, or attention blocks |
| `--n_train` | number of training configurations |
| `--labels permuted` | shuffle the fate labels before training (see below) |
| `--seed` | random seed |

`--labels permuted` is the control that makes a negative result meaningful. It destroys any
relationship between configuration and fate, so whatever score it produces is what the whole
procedure yields when there is certainly nothing to find. A real result must beat it.

Expect hours rather than minutes on a single GPU to reproduce every run reported in the paper.
The transformer runs and the deepest neural cellular automata dominate the total.

---

## What is in `results/`

**`results/per_run/`** holds one small CSV per training run. Each file records balanced
accuracy, the recall for each of the three fates separately, the number of training and test
configurations, the random seed, and how long training took. Filenames encode the run, so
`shard_nca_f000_u32.csv` is the neural cellular automaton at f = 0 with 32 update steps.
Files beginning `evaldepth_` record a diagnostic in which a trained neural cellular automaton
is re-run for a different number of update steps without being retrained.

Note that these files use two different column layouts: the hidden Markov model files place
balanced accuracy in a different column from the neural-network files. **Read them by column
name, never by column position.**

**`results/figure_data/`** holds exactly the numbers plotted in each supplementary figure, one
file per figure, so that any panel can be checked or replotted without rerunning anything.

| Supplementary figure | shows | data file |
|---|---|---|
| 33 | logistic regression, XGBoost and MLP against f | `supp_fig33_tabular_models_vs_f.csv` |
| 34 | neural cellular automaton against number of update steps | `supp_fig34_neuralCA_depth.csv` |
| 35 | transformer | `supp_fig35_transformer.csv` |
| 36 | hidden Markov model | `supp_fig36_hiddenMarkov.csv` |
| 37 | recall for each fate against model depth | `supp_fig37_perfate_recall.csv` |

The transformer figure also has `supp_fig35_transformer_learning_curves.csv`, recording
accuracy on held-out configurations after every pass through the training data.
