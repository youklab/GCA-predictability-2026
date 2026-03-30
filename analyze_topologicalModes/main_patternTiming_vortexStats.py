"""
Runs many CA simulations in parallel across multiple lattice sizes and computes
two complementary statistics about the relationship between vortex dynamics and
final pattern formation:

1. For static configurations (type 1) and rectilinear waves (type 3): the number
   of timesteps between the disappearance of the last vortex pair and the formation
   of the final pattern. This quantifies how quickly the system settles after all
   vortices have annihilated.

2. For spiral waves (type 2): the number of vortices present when the spiral wave
   first establishes itself as the recurring pattern.

Outputs per grid size: histogram PDFs and a summary statistics text file.

Usage:
  python main_patternTiming_vortexStats.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import HY_CA_secrete_and_sense_cells as CA
from multiprocessing import Pool, cpu_count
import os
from datetime import datetime

# Simulation settings
TMAX = 20000
N_RUNS = 20000
GRID_SIZES = [14, 16, 18, 20]
N_CORES = max(1, cpu_count() - 1)

# CA parameters (match those used in the paper)
RCELL = 0.2
A0 = 1.5
PERIODIC_BC = [1, 1]

M_MATRIX = np.array([[1,  1],
                     [-1, 0]])
K_MATRIX = np.array([[3,  10],
                     [11,  4]])
C_MATRIX = np.array([18, 16])
LAMB = np.array([1.0, 1.2])

# Initial condition: maximally disordered, approximately equal gene fractions
P0    = np.array([0.5, 0.55])
I_MIN = 0.01
DI    = 0.05


def run_single_simulation(args):
    """
    Run a single CA simulation and extract timing and vortex statistics.

    Args:
        args: tuple of (run_id, gridsize)

    Returns:
        dict with keys:
            - 'run_id': simulation identifier
            - 'gridsize': grid size used
            - 'final_type': pattern type (1=static, 2=spiral wave, 3=rectilinear wave)
            - 'last_vortex_time': timestep when the last vortex pair disappears
            - 'pattern_form_time': timestep when the final pattern first recurs
            - 'time_after_vortex': timesteps between vortex disappearance and pattern formation
            - 'final_n_vortices': number of vortices present when spiral pattern forms (type 2 only)
            - 'completed': whether the simulation reached a stable pattern before tmax
        Returns None if the simulation raises an exception.
    """
    run_id, gridsize = args

    try:
        n_cells = gridsize ** 2
        trajectory = CA.CellularLattice(gridsize, n_cells, RCELL, A0, TMAX)
        trajectory.init_lattice(gridsize, PERIODIC_BC, A0, RCELL, 'triangular')
        trajectory.init_cell_state(P0, I_MIN, DI)
        trajectory.init_general_parameters(K_MATRIX, C_MATRIX, M_MATRIX, LAMB)
        trajectory.run_model(TMAX)
        trajectory.analyse_trajectory()

        final_type = trajectory.final_pattern_type
        n_vortex = trajectory.n_vortex

        vortex_times = np.where(n_vortex > 0)[0]
        if len(vortex_times) > 0:
            last_vortex_time = vortex_times[-1]
        else:
            last_vortex_time = -1  # no vortices formed during this run

        pattern_form_time = trajectory.first_recurrent_state_time

        if last_vortex_time >= 0 and pattern_form_time > last_vortex_time:
            time_after_vortex = pattern_form_time - last_vortex_time
        else:
            time_after_vortex = -1

        if final_type == 2:  # spiral wave: record vortex count at pattern formation
            if pattern_form_time < len(n_vortex):
                final_n_vortices = n_vortex[pattern_form_time - 1]
            else:
                final_n_vortices = n_vortex[-1]
        else:
            final_n_vortices = 0

        completed = pattern_form_time < TMAX

        return {
            'run_id': run_id,
            'gridsize': gridsize,
            'final_type': final_type,
            'last_vortex_time': last_vortex_time,
            'pattern_form_time': pattern_form_time,
            'time_after_vortex': time_after_vortex,
            'final_n_vortices': final_n_vortices,
            'completed': completed
        }

    except Exception as e:
        print(f"Error in run {run_id} for grid size {gridsize}: {str(e)}")
        return None


def generate_histograms_for_gridsize(results, gridsize, output_dir):
    """
    Generate PDF histograms for a specific grid size.

    Args:
        results: list of result dictionaries from simulations
        gridsize: the grid size being analyzed
        output_dir: directory to save PDF files
    """
    grid_results = [r for r in results if r is not None and r['gridsize'] == gridsize]

    if len(grid_results) == 0:
        print(f"No valid results for grid size {gridsize}")
        return

    type1_results = [r for r in grid_results if r['final_type'] == 1]
    type2_results = [r for r in grid_results if r['final_type'] == 2]
    type3_results = [r for r in grid_results if r['final_type'] == 3]

    print(f"\nGrid size {gridsize}x{gridsize}:")
    print(f"  Type 1 (static):           {len(type1_results)} runs")
    print(f"  Type 2 (spiral wave):      {len(type2_results)} runs")
    print(f"  Type 3 (rectilinear wave): {len(type3_results)} runs")

    if len(type1_results) > 0:
        times_type1 = [r['time_after_vortex'] for r in type1_results if r['time_after_vortex'] >= 0]

        if len(times_type1) > 0:
            pdf_path = os.path.join(output_dir, f'histogram_type1_static_grid{gridsize}.pdf')
            with PdfPages(pdf_path) as pdf:
                fig, ax = plt.subplots(figsize=(10, 6))

                counts, bins, patches = ax.hist(times_type1, bins=50, density=True,
                                                alpha=0.7, color='blue', edgecolor='black')
                ax.clear()
                percentages = (counts / counts.sum()) * 100
                ax.bar(bins[:-1], percentages, width=np.diff(bins),
                       alpha=0.7, color='blue', edgecolor='black', align='edge')

                ax.set_xlabel('Timesteps after last vortex pair disappears', fontsize=12)
                ax.set_ylabel('Percentage of static pattern runs (%)', fontsize=12)
                ax.set_title(f'Static Patterns (Type 1) - Grid {gridsize}\u00d7{gridsize}\n'
                             f'Time until pattern formation after vortex disappearance\n'
                             f'(n = {len(times_type1)} runs)', fontsize=14)
                ax.grid(True, alpha=0.3)

                plt.tight_layout()
                pdf.savefig(fig)
                plt.close()

            print(f"  Saved: {pdf_path}")

    if len(type3_results) > 0:
        times_type3 = [r['time_after_vortex'] for r in type3_results if r['time_after_vortex'] >= 0]

        if len(times_type3) > 0:
            pdf_path = os.path.join(output_dir, f'histogram_type3_rectilinear_grid{gridsize}.pdf')
            with PdfPages(pdf_path) as pdf:
                fig, ax = plt.subplots(figsize=(10, 6))

                counts, bins, patches = ax.hist(times_type3, bins=50, density=True,
                                                alpha=0.7, color='green', edgecolor='black')
                ax.clear()
                percentages = (counts / counts.sum()) * 100
                ax.bar(bins[:-1], percentages, width=np.diff(bins),
                       alpha=0.7, color='green', edgecolor='black', align='edge')

                ax.set_xlabel('Timesteps after last vortex pair disappears', fontsize=12)
                ax.set_ylabel('Percentage of rectilinear wave runs (%)', fontsize=12)
                ax.set_title(f'Rectilinear Waves (Type 3) - Grid {gridsize}\u00d7{gridsize}\n'
                             f'Time until pattern formation after vortex disappearance\n'
                             f'(n = {len(times_type3)} runs)', fontsize=14)
                ax.grid(True, alpha=0.3)

                plt.tight_layout()
                pdf.savefig(fig)
                plt.close()

            print(f"  Saved: {pdf_path}")

    if len(type2_results) > 0:
        n_vortices_type2 = [r['final_n_vortices'] for r in type2_results if r['final_n_vortices'] > 0]

        if len(n_vortices_type2) > 0:
            pdf_path = os.path.join(output_dir, f'histogram_type2_spiral_grid{gridsize}.pdf')
            with PdfPages(pdf_path) as pdf:
                fig, ax = plt.subplots(figsize=(10, 6))

                max_vortices = int(np.max(n_vortices_type2))
                bins = np.arange(0, max_vortices + 2) - 0.5

                counts, _, _ = ax.hist(n_vortices_type2, bins=bins, density=True,
                                       alpha=0.7, color='red', edgecolor='black')
                ax.clear()
                percentages = (counts / counts.sum()) * 100
                bin_centers = np.arange(0, max_vortices + 1)
                ax.bar(bin_centers, percentages, width=0.8,
                       alpha=0.7, color='red', edgecolor='black')

                ax.set_xlabel('Final number of vortices (= 2 \u00d7 number of \u00b11 vortex pairs)', fontsize=12)
                ax.set_ylabel('Percentage of spiral wave runs (%)', fontsize=12)
                ax.set_title(f'Spiral Waves (Type 2) - Grid {gridsize}\u00d7{gridsize}\n'
                             f'Number of vortices when spiral waves form\n'
                             f'(n = {len(n_vortices_type2)} runs)', fontsize=14)
                ax.grid(True, alpha=0.3, axis='y')
                ax.set_xticks(bin_centers)

                plt.tight_layout()
                pdf.savefig(fig)
                plt.close()

            print(f"  Saved: {pdf_path}")


def generate_statistics_file(all_results, output_dir):
    """
    Write a summary statistics text file covering all grid sizes and pattern types.

    Args:
        all_results: list of all result dictionaries
        output_dir: directory to save the statistics file
    """
    stats_path = os.path.join(output_dir, 'statistics_summary.txt')

    with open(stats_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("CELLULAR AUTOMATON PATTERN FORMATION STATISTICS\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total simulations requested per grid size: {N_RUNS}\n")
        f.write(f"Maximum timesteps: {TMAX}\n\n")

        for gridsize in GRID_SIZES:
            grid_results = [r for r in all_results if r is not None and r['gridsize'] == gridsize]

            f.write("-" * 80 + "\n")
            f.write(f"GRID SIZE: {gridsize}\u00d7{gridsize}\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total completed runs: {len(grid_results)}\n\n")

            type1 = [r for r in grid_results if r['final_type'] == 1]
            type2 = [r for r in grid_results if r['final_type'] == 2]
            type3 = [r for r in grid_results if r['final_type'] == 3]

            f.write(f"Pattern type distribution:\n")
            f.write(f"  Type 1 (static):           {len(type1):6d} runs ({100 * len(type1) / len(grid_results):.2f}%)\n")
            f.write(f"  Type 2 (spiral wave):      {len(type2):6d} runs ({100 * len(type2) / len(grid_results):.2f}%)\n")
            f.write(f"  Type 3 (rectilinear wave): {len(type3):6d} runs ({100 * len(type3) / len(grid_results):.2f}%)\n\n")

            type1_final_vortices = [r['final_n_vortices'] for r in type1]
            type3_final_vortices = [r['final_n_vortices'] for r in type3]

            type1_zero = sum([1 for v in type1_final_vortices if v == 0])
            type3_zero = sum([1 for v in type3_final_vortices if v == 0])

            f.write(f"Vortex count at pattern formation for non-spiral runs:\n")
            f.write(f"  Type 1: {type1_zero}/{len(type1)} runs ended with 0 vortices\n")
            f.write(f"  Type 3: {type3_zero}/{len(type3)} runs ended with 0 vortices\n")

            if type1_zero + type3_zero == len(type1) + len(type3):
                f.write(f"  OK: all non-spiral runs ended with 0 vortices\n\n")
            else:
                f.write(f"  WARNING: some non-spiral runs ended with non-zero vortices\n\n")

            incomplete = [r for r in grid_results if not r['completed']]
            f.write(f"Runs that did not form a pattern by t={TMAX}: {len(incomplete)}\n")
            if len(incomplete) > 0:
                f.write(f"  Type breakdown:\n")
                f.write(f"    Type 1: {len([r for r in incomplete if r['final_type'] == 1])}\n")
                f.write(f"    Type 2: {len([r for r in incomplete if r['final_type'] == 2])}\n")
                f.write(f"    Type 3: {len([r for r in incomplete if r['final_type'] == 3])}\n")
            f.write("\n")

            if len(type1) > 0:
                times_type1 = [r['time_after_vortex'] for r in type1 if r['time_after_vortex'] >= 0]
                if len(times_type1) > 0:
                    f.write(f"Type 1 timing (timesteps after vortex disappearance):\n")
                    f.write(f"  Mean:   {np.mean(times_type1):.2f}\n")
                    f.write(f"  Median: {np.median(times_type1):.2f}\n")
                    f.write(f"  Std:    {np.std(times_type1):.2f}\n")
                    f.write(f"  Min:    {np.min(times_type1):.0f}\n")
                    f.write(f"  Max:    {np.max(times_type1):.0f}\n\n")

            if len(type3) > 0:
                times_type3 = [r['time_after_vortex'] for r in type3 if r['time_after_vortex'] >= 0]
                if len(times_type3) > 0:
                    f.write(f"Type 3 timing (timesteps after vortex disappearance):\n")
                    f.write(f"  Mean:   {np.mean(times_type3):.2f}\n")
                    f.write(f"  Median: {np.median(times_type3):.2f}\n")
                    f.write(f"  Std:    {np.std(times_type3):.2f}\n")
                    f.write(f"  Min:    {np.min(times_type3):.0f}\n")
                    f.write(f"  Max:    {np.max(times_type3):.0f}\n\n")

            if len(type2) > 0:
                n_vort = [r['final_n_vortices'] for r in type2 if r['final_n_vortices'] > 0]
                if len(n_vort) > 0:
                    f.write(f"Type 2 final vortex counts:\n")
                    f.write(f"  Mean:   {np.mean(n_vort):.2f}\n")
                    f.write(f"  Median: {np.median(n_vort):.0f}\n")
                    f.write(f"  Std:    {np.std(n_vort):.2f}\n")
                    f.write(f"  Min:    {np.min(n_vort):.0f}\n")
                    f.write(f"  Max:    {np.max(n_vort):.0f}\n\n")

            f.write("\n")

    print(f"\nSaved statistics: {stats_path}")


def main():
    """
    Run parallel simulations and generate histogram outputs for all grid sizes.
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f'output_patternTiming_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("CELLULAR AUTOMATON PATTERN FORMATION ANALYSIS")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print(f"Simulations per grid size: {N_RUNS}")
    print(f"Grid sizes: {GRID_SIZES}")
    print(f"Workers: {N_CORES}")
    print("=" * 80)

    all_args = []
    for gridsize in GRID_SIZES:
        for run_id in range(N_RUNS):
            all_args.append((run_id, gridsize))

    print(f"\nStarting {len(all_args)} total simulations...\n")

    with Pool(processes=N_CORES) as pool:
        all_results = pool.map(run_single_simulation, all_args)

    valid_results = [r for r in all_results if r is not None]
    print(f"\nCompleted {len(valid_results)} successful simulations out of {len(all_args)} total")

    print("\nGenerating histograms...")
    for gridsize in GRID_SIZES:
        generate_histograms_for_gridsize(valid_results, gridsize, output_dir)

    print("\nGenerating statistics summary...")
    generate_statistics_file(valid_results, output_dir)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print(f"All outputs saved to: {output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()