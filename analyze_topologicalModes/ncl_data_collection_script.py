"""
Collects NCL (non-contractible loop) and vortex-string statistics from an ensemble
of CA simulations run in parallel, and saves the results as .npy files for downstream
analysis and plotting.

For each run, the script extracts:
  - Final pattern type (1 = static, 2 = spiral wave, 3 = rectilinear wave)
  - Pattern formation time (first_recurrent_state_time)
  - End frame (last frame before the final event, excluding zero-charge vortices)
  - NCL ratio: fraction of vortex-connecting strings that are NCL strings, per frame
  - Number of vortex-connecting strings per frame
  - Shortest and longest vortex-connecting string lengths per frame
  - Total number of NCLs present at the final frame

Runs are batched in parallel until a target number of non-spiral runs is collected.
Results are checkpointed to disk after every few non-spiral runs so that a long
job can be resumed if interrupted.

Note: any run that does not reach a recurring state before tmax is automatically
classified as a spiral wave (final_pattern_type = 2). Before plotting, filter out
runs where first_recurrent_state_time == tmax, as these have not formed a pattern.

Dependencies: tqdm (pip install tqdm)
"""

from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import os
import numpy as np
import HY_CA_secrete_and_sense_cells as CA


# -----------------------------------------------------------------------
# CA parameters (match those used in the paper)
# -----------------------------------------------------------------------
output_dir   = 'output_ncl_data'
os.makedirs(output_dir, exist_ok=True)

tmax        = 75000
rcell       = 0.2
a0          = 1.5
lattice_type = 'triangular'
periodic_bc  = [1, 1]

p0    = np.array([0.5, 0.55])
I_min = 0.01
dI    = 0.05

M_matrix = np.array([[1,  1], [-1, 0]])
K_matrix = np.array([[3, 10], [11, 4]])
C_matrix = np.array([18, 16])
lamb     = np.array([1, 1.2])


# -----------------------------------------------------------------------
# Per-run worker function
# -----------------------------------------------------------------------
def run_single_CA(iteration, gridsize):
    """
    Run one CA simulation and extract NCL and vortex-string statistics.

    Returns a dict with per-frame arrays of NCL ratios, string counts, and
    string lengths, plus scalar summary fields. Returns None on failure.
    """
    num_cells  = gridsize ** 2
    trajectory = CA.CellularLattice(gridsize, num_cells, rcell, a0, tmax)
    trajectory.init_lattice(gridsize, periodic_bc, a0, rcell, lattice_type)
    trajectory.init_general_parameters(K_matrix, C_matrix, M_matrix, lamb)
    trajectory.init_cell_state(p0, I_min, dI)
    trajectory.convert_currentBinaryState_to_4cell_state()
    trajectory.run_model(tmax)
    trajectory.analyse_trajectory()

    finalPattern_Label = trajectory.final_pattern_type
    computationTime    = int(trajectory.first_recurrent_state_time)
    _, end_frame       = trajectory.determine_end_frame(include_zero_charge=False)
    nclratiosdetails   = trajectory.get_percent_of_strings_that_are_NCL(
                             include_zero_charge=False, return_details=True)
    num_vortices = trajectory.n_vortex

    shortest_paths = []
    longest_paths  = []
    percentncl     = []
    numstring      = []

    for frame in range(0, end_frame):
        ncl       = trajectory.find_noncontractible_loops(frame)
        num_ncls  = len(ncl)
        _, clusterconnects = trajectory.get_all_cluster_connections(
            frame, include_zero_charge=False, return_all=True)

        nclratio       = nclratiosdetails[frame]['percentage_ncl']
        numstringsframe = nclratiosdetails[frame]['total_strings']
        percentncl.append(nclratio)
        numstring.append(numstringsframe)

        all_entries = [entry for entries in clusterconnects.values() for entry in entries]
        if all_entries:
            longest_path  = max(all_entries, key=lambda x: x['distance'])['distance']
            shortest_path = min(all_entries, key=lambda x: x['distance'])['distance']
            longest_paths.append(longest_path)
            shortest_paths.append(shortest_path)
        else:
            longest_paths.append(np.nan)
            shortest_paths.append(np.nan)

    return {
        'iteration':          iteration,
        'gridsize':           gridsize,
        'computationTime':    computationTime,
        'end_frame':          end_frame,
        'finalPattern_Label': finalPattern_Label,
        'ncl_ratios':         percentncl,
        'num_vortices':       num_vortices,
        'num_strings':        numstring,
        'longest_paths':      longest_paths,
        'shortest_paths':     shortest_paths,
        'num_ncl':            num_ncls,
    }


def run_single_CA_wrapper(args):
    return run_single_CA(*args)


# -----------------------------------------------------------------------
# Main: run ensemble in parallel until n_target non-spiral runs collected
# -----------------------------------------------------------------------
if __name__ == '__main__':
    num_workers = min(cpu_count(), 8)

    # Target number of non-spiral runs to collect per grid size.
    # Runs are batched until this count is reached; results are checkpointed
    # to disk every checkpoint_every non-spiral runs.
    n_target         = 100
    checkpoint_every = 5

    for gridsize in [20]:
        save_path        = os.path.join(output_dir, f'results_gridsize_{gridsize}.npy')
        combined_results = []

        # Resume from existing file if present
        if os.path.exists(save_path):
            try:
                existing_data    = np.load(save_path, allow_pickle=True)
                combined_results = list(existing_data)
                print(f'Loaded {len(combined_results)} existing results for gridsize={gridsize}')
            except Exception as e:
                print(f'Could not load existing file, starting fresh: {e}')
                combined_results = []

        non_spiral_count = sum(r['finalPattern_Label'] != 2 for r in combined_results)
        total_iterations = len(combined_results)

        print(f'Starting gridsize={gridsize} '
              f'(already have {non_spiral_count} non-spiral runs; '
              f'target = {n_target})')
        print(f'Results will be saved to: {save_path}')

        with Pool(processes=num_workers) as pool:
            with tqdm(total=n_target, initial=non_spiral_count,
                      desc=f'Non-spiral runs (grid={gridsize})') as pbar:
                while non_spiral_count < n_target:
                    batch_size        = num_workers * 2
                    tasks             = [(total_iterations + i, gridsize)
                                         for i in range(batch_size)]
                    total_iterations += batch_size

                    for result in pool.imap_unordered(run_single_CA_wrapper, tasks):
                        if result is None:
                            continue
                        combined_results.append(result)
                        if result['finalPattern_Label'] != 2:
                            non_spiral_count += 1
                            pbar.update(1)
                            if non_spiral_count % checkpoint_every == 0:
                                np.save(save_path,
                                        np.array(combined_results, dtype=object))
                            if non_spiral_count >= n_target:
                                break

        np.save(save_path, np.array(combined_results, dtype=object))
        print(f'Done. Saved {non_spiral_count} non-spiral runs '
              f'for gridsize={gridsize} to {save_path}')
