# analyze_topologicalModes

This folder contains scripts for analyzing the topological collective modes — discrete vortices and non-contractible loop (NCL) strings — that emerge during CA self-organization, as described in Koopmans, Kay, and Youk (2026).

---

## Overview

The analyses fall into three groups.

**Vortex statistics across an ensemble of runs.** Three scripts characterize how vortex populations evolve and eventually annihilate across many independent CA runs:

- `main_maxVortices.py` — for each lattice size, runs many CA simulations in parallel and records the peak vortex count per run, producing histograms of the maximum vortex abundance.
- `main_patternTiming_vortexStats.py` — records the time between the disappearance of the last vortex pair and the formation of the final pattern (for static and rectilinear-wave runs), and the number of vortices present when spiral waves form.
- `main_vortexDeclineSteps.py` — plots the descending-staircase structure of vortex count versus time (the "staircase"), with per-type breakdown into +1, −1, and 0 vortices and net topological charge. One-timestep detection artefacts are removed before plotting.
- `main_vortexFormationHistograms_tmax150000_parallel.py` — a longer-tmax version of the staircase and timing analyses, designed for larger lattice sizes.

**NCL and string analysis.** One script collects the raw data on non-contractible loops and vortex-connecting strings used for the NCL predictability analysis:

- `ncl_data_collection_script.py` — runs many CA simulations in parallel and extracts, per frame, the fraction of vortex-connecting strings that are NCL strings, the number of strings, and the shortest and longest string lengths. Results are saved as `.npy` files for downstream analysis and plotting. Runs are batched until a target number of non-spiral runs is collected. Requires `tqdm` (`pip install tqdm`).

**Sensitivity to initial conditions.** Two scripts characterize how single-cell perturbations to the initial configuration redirect macroscopic pattern formation:

- `perturb_initial_state_oneByone.py` — takes a single randomly initialized 16×16 CA, perturbs each cell one at a time (replacing its state with a randomly chosen different state), runs the CA from each modified initial configuration, and saves the final pattern type, run time, and final binary configuration for every perturbation.
- `Analyze_perturb_initial_state_oneByone.py` — loads and analyzes the output of the above script, producing histograms of final pattern type distribution, run times, and the number of cells in the final configuration that differ from the unperturbed run's final configuration.

---

## Dependencies

- Python ≥ 3.9
- NumPy, SciPy, Matplotlib
- `tqdm` (required by `ncl_data_collection_script.py` only — `pip install tqdm`)
- `HY_CA_secrete_and_sense_cells.py` (from `GCA_simulation/`) must be importable from the working directory

---

## Reference

Koopmans, Kay, and Youk (2026). For full model specification and analysis details, see **Supplementary Note 1** (CA model), **Supplementary Note 3** (accuracy of vortex-detection algorithm), and **Supplementary Note 5** (proof of topological charge neutrality).

