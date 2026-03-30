# main_maxVortices.py
"""
For each of a set of grid sizes, runs many independent CA simulations in parallel,
tracks the vortex count at each timestep, records the maximum number of vortices
observed during each run, and saves a histogram (% of runs vs. max vortex count)
as a PDF along with the raw per-run arrays as a compressed NumPy archive.

This analysis establishes that vortices appear in every run and quantifies the
typical peak vortex abundance as a function of lattice size.

Usage:
  python main_maxVortices.py
  python main_maxVortices.py --workers 12
  python main_maxVortices.py --workers 0   # auto = (cpu_count - 1)
  python main_maxVortices.py --grids 14 16 18 20 --n_runs 1000 --tmax 100
"""

from __future__ import annotations

import os
import time
import argparse
import numpy as np
import matplotlib
import sys

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import multiprocessing as mp

import HY_CA_secrete_and_sense_cells as CA


# -----------------------------
# CA parameters (match those used in the paper)
# -----------------------------
TYPE_LATTICE = "triangular"
PERIODIC_BC = [1, 1]
A0 = 1.5
RCELL = 0.2
LAMB = np.array([1.0, 1.2])

M_MATRIX = np.array([[1, 1],
                     [-1, 0]])
K_MATRIX = np.array([[3, 10],
                     [11, 4]])
C_MATRIX = np.array([18, 16])

# Initial condition: maximally disordered, approximately equal gene fractions
P0 = np.array([0.5, 0.55])
I_MIN = 0.01
DI = 0.05


# -----------------------------
# One simulation -> max vortices
# -----------------------------
def max_vortices_in_one_run(gridsize: int, tmax: int, seed: int | None) -> int:
    """
    Run CA once and return max_t (# vortices at time t).
    Seed is per-run to make results reproducible across parallelization.
    """
    if seed is not None:
        np.random.seed(int(seed))

    num_cells = gridsize**2
    traj = CA.CellularLattice(gridsize, num_cells, RCELL, A0, tmax)
    traj.init_lattice(gridsize, PERIODIC_BC, A0, RCELL, TYPE_LATTICE)
    traj.init_general_parameters(K_MATRIX, C_MATRIX, M_MATRIX, LAMB)

    traj.init_cell_state(P0, I_MIN, DI)
    traj.convert_currentBinaryState_to_4cell_state()
    traj.run_model(tmax)

    # Minimal vortex pipeline (subset of analyse_trajectory) for speed
    traj.cell_4state = CA.get_4_number_seq(traj.cell_hist)
    traj.vortex_cores = CA.compute_phase_differences(traj)
    _, n_vortex = CA.label_triangular_lattice(traj)

    return int(np.max(n_vortex))


def _worker(job):
    gs, tmax, seed = job
    return max_vortices_in_one_run(gs, tmax, seed)


# -----------------------------
# Parallel runner
# -----------------------------
def run_many_parallel(
    gridsize: int,
    tmax: int,
    n_runs: int,
    base_seed: int | None,
    workers: int,
    chunksize: int | None = None,
    progress_every_runs: int = 200,   # print every 200 completed runs
    heartbeat_seconds: float = 10.0,  # print heartbeat if no results yet
) -> np.ndarray:
    """
    Parallel execution with:
      - immediate diagnostics about Pool startup
      - progress printing by completed-run count
      - heartbeat prints if results are slow to arrive
    """
    # Per-run deterministic seeds
    if base_seed is None:
        seeds = [None] * n_runs
    else:
        seeds = list(range(int(base_seed), int(base_seed) + n_runs))

    jobs = [(gridsize, tmax, seeds[i]) for i in range(n_runs)]

    if workers == 1:
        out = np.empty(n_runs, dtype=np.int32)
        t0 = time.time()
        last_print = 0
        for i, jb in enumerate(jobs, start=1):
            out[i - 1] = _worker(jb)
            if i - last_print >= progress_every_runs or i == n_runs:
                dt = (time.time() - t0) / 60
                print(f"[{gridsize}x{gridsize}] completed {i}/{n_runs} "
                      f"({100*i/n_runs:.1f}%) | elapsed {dt:.1f} min",
                      flush=True)
                last_print = i
        return out

    if chunksize is None:
        # Smaller chunksize => earlier first result / more frequent progress, at slight overhead
        chunksize = max(1, n_runs // (workers * 50))

    ctx = mp.get_context("spawn")
    out = np.empty(n_runs, dtype=np.int32)

    print(f"[{gridsize}x{gridsize}] starting Pool(workers={workers}, chunksize={chunksize}) ...",
          flush=True)

    t0 = time.time()
    last_print_count = 0
    last_heartbeat = time.time()
    got_first = False

    with ctx.Pool(processes=workers, maxtasksperchild=200) as pool:
        print(f"[{gridsize}x{gridsize}] pool started. Waiting for first results ...", flush=True)

        iterator = pool.imap_unordered(_worker, jobs, chunksize=chunksize)

        i = 0
        while i < n_runs:
            # Heartbeat: if no results have arrived for a while, print reassurance
            now = time.time()
            if now - last_heartbeat >= heartbeat_seconds and not got_first:
                dt = (now - t0) / 60
                print(f"[{gridsize}x{gridsize}] (heartbeat) still running; "
                      f"no results returned yet | elapsed {dt:.1f} min",
                      flush=True)
                last_heartbeat = now

            try:
                result = next(iterator)
            except StopIteration:
                break

            i += 1
            out[i - 1] = result

            if not got_first:
                got_first = True
                dt = (time.time() - t0) / 60
                print(f"[{gridsize}x{gridsize}] first result returned at {dt:.2f} min", flush=True)

            if i - last_print_count >= progress_every_runs or i == n_runs:
                dt = (time.time() - t0) / 60
                print(f"[{gridsize}x{gridsize}] completed {i}/{n_runs} "
                      f"({100*i/n_runs:.1f}%) | elapsed {dt:.1f} min",
                      flush=True)
                last_print_count = i

    return out
# -----------------------------
# Plotting
# -----------------------------
def save_histogram_pdf(
    maxv: np.ndarray,
    gridsize: int,
    tmax: int,
    outdir: str,
    n_runs: int,
) -> str:
    vmin = int(maxv.min())
    vmax = int(maxv.max())

    bins = np.arange(vmin - 0.5, vmax + 1.5, 1.0)
    counts, edges = np.histogram(maxv, bins=bins)
    perc = 100.0 * counts / float(len(maxv))
    centers = 0.5 * (edges[:-1] + edges[1:])

    fig = plt.figure(figsize=(6.5, 4.2))
    ax = fig.add_subplot(111)

    ax.bar(centers, perc, width=0.9)

    ax.set_xlabel("Max # of vortices observed in a run")
    ax.set_ylabel("% of runs")
    ax.set_title(f"Grid {gridsize}×{gridsize}  |  tmax={tmax}  |  n={n_runs}")

    ax.set_xticks(np.arange(vmin, vmax + 1, 1))
    ax.set_xlim(vmin - 1, vmax + 1)
    ax.set_ylim(0, max(perc.max() * 1.10, 5))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    pdf_path = os.path.join(
        outdir,
        f"maxVortices_hist_gridsize{gridsize}_tmax{tmax}_n{n_runs}.pdf",
    )
    fig.tight_layout()
    fig.savefig(pdf_path, dpi=300)  # PDF is vector; dpi mainly affects embedded raster bits
    plt.close(fig)
    return pdf_path


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tmax", type=int, default=100)
    parser.add_argument("--n_runs", type=int, default=1_000)
    parser.add_argument("--grids", type=int, nargs="+", default=[14, 16, 18, 20])
    parser.add_argument("--outdir", type=str, default="output_maxVortices")
    parser.add_argument("--seed", type=int, default=0,
                        help="Base seed. Use --seed -1 to disable seeding.")
    parser.add_argument("--workers", type=int, default=0,
                        help="Number of worker processes. 0 => auto (cpu_count-1).")
    parser.add_argument("--chunksize", type=int, default=0,
                        help="Pool map chunksize. 0 => auto.")
    args = parser.parse_args()

    tmax = int(args.tmax)
    n_runs = int(args.n_runs)
    grids = [int(g) for g in args.grids]
    outdir = args.outdir

    base_seed = None if args.seed == -1 else int(args.seed)

    cpu = mp.cpu_count()
    if args.workers is None or int(args.workers) == 0:
        workers = max(1, cpu - 1)
    else:
        workers = max(1, int(args.workers))

    chunksize = None if int(args.chunksize) == 0 else int(args.chunksize)

    os.makedirs(outdir, exist_ok=True)

    print("=== Max-vortices analysis (parallel) ===")
    print(f"cpu_count={cpu} | workers={workers} | tmax={tmax} | n_runs={n_runs} | grids={grids}")
    print("seeding:", "OFF" if base_seed is None else f"base_seed={base_seed} (seed_i=base_seed+i)")
    if chunksize is None:
        print("chunksize: auto")
    else:
        print(f"chunksize: {chunksize}")

    for gs in grids:
        t0 = time.time()
        print(f"\n--- Grid {gs}×{gs} ---")

        maxv = run_many_parallel(gs, tmax, n_runs, base_seed, workers, chunksize)

        min_maxv = int(maxv.min())
        if min_maxv == 0:
            print("WARNING: At least one run had max vortices = 0 (statement not supported at this tmax).")
        else:
            print(f"OK: min over runs of (max vortices) = {min_maxv} (never zero).")

        # Save raw
        npz_path = os.path.join(outdir, f"raw_maxVortices_gridsize{gs}_tmax{tmax}_n{n_runs}.npz")
        np.savez_compressed(
            npz_path,
            gridsize=gs,
            tmax=tmax,
            n_runs=n_runs,
            max_vortices_per_run=maxv,
        )

        pdf_path = save_histogram_pdf(maxv, gs, tmax, outdir, n_runs)

        dt = time.time() - t0
        print(f"Saved PDF: {pdf_path}")
        print(f"Saved raw: {npz_path}")
        print(f"Done in {dt:.1f} s")

    print("\nAll done.")


if __name__ == "__main__":
    main()