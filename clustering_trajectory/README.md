# clustering_trajectory

This directory contains MATLAB code and one animation file for trajectory-level analyses based on the scalar observable used in the manuscript: the **total vortex-core size** as a function of backward time. These analyses ask when trajectories that end in different final pattern types begin to separate, cluster, or collapse onto shared late-time profiles.

## Contents

### `graph_clustering_reproducibility.m`
Runs the graph-based clustering reproducibility analysis across multiple trajectory sets.

For each input data chunk, the script:
- builds a trajectory-similarity graph using the total vortex-core-size time series over a backward-time window \([1,\tau]\),
- identifies connected components in that graph,
- tracks how the clustering structure changes as \(\tau\) varies, and
- summarizes reproducibility across trajectory sets.

The script reports three quantities across backward-time windows:
- the total number of connected components,
- the fraction of trajectories in singleton components, and
- the size of the largest connected component.

These quantities quantify when trajectories remain fragmented and when they begin to merge into larger fate-aligned groups.

### `graph_evolution_movie.m`
Generates movies showing how the graph of similar trajectories evolves as the backward-time window changes.

Each node represents one trajectory. Edges connect pairs of trajectories whose total vortex-core-size time series are sufficiently similar within the window \([1,\tau]\). As \(\tau\) changes, trajectories can remain isolated or merge into connected components.

The node colors encode final pattern type:
- **red**: rectilinear wave
- **teal**: static configuration
- **gold**: spiral wave

The script processes all `chunk_*.mat` files in a selected folder and exports one animated graph sequence per chunk.

### `heatmap_reproducibility.m`
Generates heat maps of the total vortex-core-size trajectories across multiple trajectory sets.

For each chunk, the script:
- loads the trajectory matrix,
- reorders rows by nearest-neighbor similarity in trajectory space,
- plots the total vortex-core size as a function of backward time, and
- assembles the resulting heat maps into a multi-panel reproducibility figure.

This provides a visual comparison of late-time trajectory structure across independently generated trajectory sets.

### `graph_evolution.gif`
Animated example of the graph-based clustering analysis.

The label `time window = #` indicates the backward-time window size \(\tau\). Each frame shows the trajectory-similarity graph at that value of \(\tau\). Node colors denote final pattern type:
- **red**: rectilinear wave
- **teal**: static configuration
- **gold**: spiral wave

## Input data format

These scripts expect MATLAB files named `chunk_*.mat` containing trajectory-level outputs from the cellular automaton simulations. Depending on the script, the required variables include:

- `subset_core_size`
- `subset_num_vortices`
- `subset_final_pattern_type`
- `subset_computation_time`

All analyses use **backward time**, so column 1 corresponds to the timestep immediately preceding final-pattern formation, column 2 to two timesteps before final-pattern formation, and so on.

## Notes

- These scripts analyze trajectories using a **coarse scalar observable** rather than the full lattice state.
- In the graph-based analyses, trajectory similarity is defined by Euclidean distance between total-vortex-core-size time series within a chosen backward-time window.
- The reproducibility scripts use **chunks 1–19**. The movie-generation script can use all available `chunk_*.mat` files in the selected directory.
