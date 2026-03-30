"""
Runs an ensemble of CA simulations in parallel across multiple lattice sizes
and computes histograms characterizing how vortex dynamics relate to final
pattern formation. For each lattice size and pattern type, produces:

  1. Static configurations (type 1) and rectilinear waves (type 3):
     histogram of Δt = t_form - t_lastPairDisappear, the number of timesteps
     between the disappearance of the last vortex pair and the formation of the
     final pattern.

  2. Spiral waves (type 2):
     histogram of the number of charged vortices present at pattern formation.

  3. A summary statistics text file with type fractions, timing statistics,
     and vortex sanity checks.

Raw per-run arrays are saved as compressed NumPy archives for downstream analysis.

Usage:
  python main_vortexFormationHistograms_tmax150000_parallel.py
  python main_vortexFormationHistograms_tmax150000_parallel.py --workers 16
  python main_vortexFormationHistograms_tmax150000_parallel.py --progress_every 200
  python main_vortexFormationHistograms_tmax150000_parallel.py --maxtasksperchild 25
  python main_vortexFormationHistograms_tmax150000_parallel.py --chunksize 5
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import multiprocessing as mp

import HY_CA_secrete_and_sense_cells as CA


# -----------------------------
# CA parameters (match those used in the paper)
# -----------------------------
PERIODIC_BC = [1, 1]
A0 = 1.5
RCELL = 0.2
LAMB = np.array([1.0, 1.2])

M_MATRIX = np.array([[1, 1],
                     [-1, 0]])
K_MATRIX = np.array([[3, 10],
                     [11, 4]])
C_MATRIX = np.array([18, 16])

P0 = np.array([0.5, 0.55])
I_MIN = 0.01
DI = 0.05


@dataclass(frozen=True)
class RunResult:
    final_type: int
    t_form: int
    t_last_pair_disappear: int
    n_charged_at_form: int
    n_charged_at_end: int


def _maybe_get_rss_gb() -> str:
    """
    Best-effort memory readout (RSS) for *this* process.
    On macOS/Linux, resource.ru_maxrss works but units differ:
      - macOS: bytes
      - Linux: KB
    We'll try to detect scale.
    """
    try:
        import resource  # stdlib
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS reports ru_maxrss in bytes; Linux reports in KB
        if rss > 10_000_000_000:
            gb = rss / (1024**3)
        else:
            gb = (rss * 1024) / (1024**3)
        return f"{gb:.2f} GB"
    except Exception:
        return "NA"


def _one_run(args: Tuple[int, int, int]) -> RunResult:
    """Worker: run one trajectory and extract summary metrics."""
    gridsize, tmax, seed = args

    # deterministic seed per run
    np.random.seed(seed)

    num_cells = gridsize ** 2
    traj = CA.CellularLattice(gridsize, num_cells, RCELL, A0, tmax)
    traj.init_lattice(gridsize, PERIODIC_BC, A0, RCELL, "triangular")
    traj.init_general_parameters(K_MATRIX, C_MATRIX, M_MATRIX, LAMB)
    traj.init_cell_state(P0, I_MIN, DI)
    traj.run_model(tmax)
    traj.analyse_trajectory()

    final_type = int(traj.final_pattern_type)
    t_form = int(traj.first_recurrent_state_time)

    # traj.charges has shape (tmax, 2+max_label); nonzero entries indicate charged vortex cores
    if isinstance(traj.charges, list) or traj.charges is None:
        n_charged_end = 0
        n_charged_form = 0
    else:
        charges = np.asarray(traj.charges)
        n_charged_end = int(np.sum(charges[-1, :] != 0))
        if t_form <= 0:
            n_charged_form = int(np.sum(charges[0, :] != 0))
        else:
            idx = min(t_form - 1, charges.shape[0] - 1)
            n_charged_form = int(np.sum(charges[idx, :] != 0))

    # Last vortex-pair disappearance time:
    # compute_annihilation_moments detects drops between t and t+1, so disappearance time is (idx_last + 1).
    if isinstance(traj.idx_annihilation_moments, list) or traj.idx_annihilation_moments is None:
        t_disappear = 0
    else:
        idx_ann = np.asarray(traj.idx_annihilation_moments).ravel()
        t_disappear = int(np.max(idx_ann) + 1) if idx_ann.size > 0 else 0

    # Encourage garbage collection of big arrays
    try:
        traj.clean_data()
    except Exception:
        pass

    return RunResult(
        final_type=final_type,
        t_form=t_form,
        t_last_pair_disappear=t_disappear,
        n_charged_at_form=n_charged_form,
        n_charged_at_end=n_charged_end,
    )


def _bins_for_integer_data(x: np.ndarray, max_bins: int = 120) -> np.ndarray:
    x = np.asarray(x)
    if x.size == 0:
        return np.arange(0, 2)
    xmax = int(np.max(x))
    if xmax <= max_bins:
        return np.arange(-0.5, xmax + 1.5, 1.0)
    return np.linspace(0, xmax, max_bins)


def _plot_percent_hist(data: np.ndarray, bins: np.ndarray, title: str, xlabel: str, outpath: str) -> None:
    data = np.asarray(data)
    plt.figure(figsize=(6.4, 4.2))
    if data.size == 0:
        plt.text(0.5, 0.5, "No runs in this category", ha="center", va="center")
        plt.axis("off")
        plt.savefig(outpath, bbox_inches="tight")
        plt.close()
        return

    weights = np.ones_like(data, dtype=float) * (100.0 / float(data.size))
    plt.hist(data, bins=bins, weights=weights, edgecolor="black", linewidth=0.6)
    plt.ylabel("% of runs (within this type)")
    plt.xlabel(xlabel)
    plt.title(title)
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def _fmt_hms(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h:d}h{m:02d}m{s:02d}s"
    if m > 0:
        return f"{m:d}m{s:02d}s"
    return f"{s:d}s"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tmax", type=int, default=150000)
    ap.add_argument("--n_runs", type=int, default=20000)
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    ap.add_argument("--outdir", type=str, default="output_vortexFormationHistograms_tmax150000")
    ap.add_argument("--seed", type=int, default=12345, help="Base seed for reproducibility.")

    ap.add_argument("--progress_every", type=int, default=200,
                    help="Print progress every N completed runs (per gridsize).")
    ap.add_argument("--chunksize", type=int, default=10,
                    help="Pool imap_unordered chunksize. Smaller improves responsiveness; larger improves throughput.")
    ap.add_argument("--maxtasksperchild", type=int, default=25,
                    help="Recycle worker processes after N tasks to limit memory growth.")
    args = ap.parse_args()

    t0 = time.time()
    os.makedirs(args.outdir, exist_ok=True)

    gridsizes = [14, 16, 18, 20]

    print("\n============================================================")
    print("Running CA ensemble with progress reporting")
    print(f"tmax={args.tmax} | n_runs={args.n_runs} per grid | workers={args.workers}")
    print(f"chunksize={args.chunksize} | maxtasksperchild={args.maxtasksperchild}")
    print("============================================================\n")

    for gz in gridsizes:
        gz_start = time.time()
        print(f"\n=== START gridsize {gz}x{gz} | n_runs={args.n_runs} | tmax={args.tmax} ===", flush=True)

        gz_dir = os.path.join(args.outdir, f"gridsize{gz}")
        os.makedirs(gz_dir, exist_ok=True)

        # Deterministic seeds for each run
        ss = np.random.SeedSequence(args.seed + gz)
        child_seeds = ss.spawn(args.n_runs)
        seeds = [int(s.generate_state(1)[0]) for s in child_seeds]
        work = [(gz, args.tmax, seeds[i]) for i in range(args.n_runs)]

        results: List[RunResult] = []
        last_print = time.time()
        done = 0

        ctx = mp.get_context("spawn")
        with ctx.Pool(
            processes=args.workers,
            maxtasksperchild=args.maxtasksperchild if args.maxtasksperchild > 0 else None,
        ) as pool:
            it = pool.imap_unordered(_one_run, work, chunksize=max(1, args.chunksize))
            for done, rr in enumerate(it, start=1):
                results.append(rr)

                if (done % args.progress_every == 0) or (done == args.n_runs):
                    now = time.time()
                    elapsed = now - gz_start
                    rate = done / elapsed if elapsed > 0 else 0.0
                    remaining = (args.n_runs - done) / rate if rate > 0 else float("inf")
                    rss = _maybe_get_rss_gb()

                    # Rate-limit progress output
                    if now - last_print > 0.2 or done == args.n_runs:
                        print(
                            f"[gz={gz:02d}] {done:5d}/{args.n_runs} "
                            f"({100.0*done/args.n_runs:5.1f}%) | "
                            f"elapsed={_fmt_hms(elapsed)} | "
                            f"rate={rate:6.2f} runs/s | "
                            f"ETA={_fmt_hms(remaining)} | "
                            f"RSS(main)={rss}",
                            flush=True,
                        )
                        last_print = now

        # Postprocess results for this grid size
        final_types = np.array([r.final_type for r in results], dtype=np.int16)
        t_form = np.array([r.t_form for r in results], dtype=np.int32)
        t_dis = np.array([r.t_last_pair_disappear for r in results], dtype=np.int32)
        n_ch_form = np.array([r.n_charged_at_form for r in results], dtype=np.int32)
        n_ch_end = np.array([r.n_charged_at_end for r in results], dtype=np.int32)

        mask1 = final_types == 1
        mask2 = final_types == 2
        mask3 = final_types == 3

        # "No pattern by tmax": no recurrence detected within window
        no_pattern = t_form >= args.tmax

        # Type 1 & 3: Δt after last vortex-pair disappearance
        dt_after_lastpair = t_form - t_dis
        dt_type1 = dt_after_lastpair[mask1]
        dt_type3 = dt_after_lastpair[mask3]

        # Type 2: vortices at formation (charged cores at t_form-1)
        vort_at_form_type2 = n_ch_form[mask2]

        # Save raw per-run arrays
        np.savez_compressed(
            os.path.join(gz_dir, f"raw_gridsize{gz}_tmax{args.tmax}_n{args.n_runs}.npz"),
            final_types=final_types,
            t_form=t_form,
            t_last_pair_disappear=t_dis,
            dt_after_lastpair=dt_after_lastpair,
            n_charged_at_form=n_ch_form,
            n_charged_at_end=n_ch_end,
        )

        out1 = os.path.join(
            gz_dir,
            f"hist_type1_static_dtAfterLastPair_gridsize{gz}_tmax{args.tmax}_n{args.n_runs}.pdf",
        )
        _plot_percent_hist(
            dt_type1,
            bins=_bins_for_integer_data(dt_type1),
            title=f"Static (type 1): Δt after last vortex-pair disappearance\n(grid {gz}×{gz}, tmax={args.tmax})",
            xlabel="timesteps after last vortex pair disappears",
            outpath=out1,
        )

        out3 = os.path.join(
            gz_dir,
            f"hist_type3_rectilinear_dtAfterLastPair_gridsize{gz}_tmax{args.tmax}_n{args.n_runs}.pdf",
        )
        _plot_percent_hist(
            dt_type3,
            bins=_bins_for_integer_data(dt_type3),
            title=f"Rectilinear waves (type 3): Δt after last vortex-pair disappearance\n(grid {gz}×{gz}, tmax={args.tmax})",
            xlabel="timesteps after last vortex pair disappears",
            outpath=out3,
        )

        out2 = os.path.join(
            gz_dir,
            f"hist_type2_spiral_vorticesAtFormation_gridsize{gz}_tmax{args.tmax}_n{args.n_runs}.pdf",
        )
        _plot_percent_hist(
            vort_at_form_type2,
            bins=_bins_for_integer_data(vort_at_form_type2, max_bins=60),
            title=f"Spiral waves (type 2): charged vortices at pattern formation\n(grid {gz}×{gz}, tmax={args.tmax})",
            xlabel="# charged vortices at formation ( = 2 × #pairs )",
            outpath=out2,
        )

        stats_path = os.path.join(gz_dir, f"stats_gridsize{gz}_tmax{args.tmax}_n{args.n_runs}.txt")
        n1 = int(np.sum(mask1))
        n2 = int(np.sum(mask2))
        n3 = int(np.sum(mask3))
        n_np = int(np.sum(no_pattern))

        # Sanity checks
        non_spiral_end_nonzero = int(np.sum((~mask2) & (n_ch_end != 0)))
        spiral_end_zero = int(np.sum(mask2 & (n_ch_end == 0)))

        def _summ(x: np.ndarray) -> str:
            if x.size == 0:
                return "(none)"
            return f"mean={np.mean(x):.2f}, median={np.median(x):.0f}, min={np.min(x)}, max={np.max(x)}"

        with open(stats_path, "w", encoding="utf-8") as f:
            f.write(f"Grid size: {gz}x{gz}\n")
            f.write(f"tmax: {args.tmax}\n")
            f.write(f"n_runs: {args.n_runs}\n")
            f.write("\n")
            f.write("Final pattern type counts:\n")
            f.write(f"  type 1 (static): {n1}\n")
            f.write(f"  type 2 (spiral): {n2}\n")
            f.write(f"  type 3 (rectilinear waves): {n3}\n")
            f.write("\n")
            f.write(f"Runs with no recurrent state detected by tmax (t_form >= tmax): {n_np}\n")
            f.write("\n")
            f.write("Vortex sanity checks:\n")
            f.write(f"  Non-spiral runs ending with nonzero charged vortices: {non_spiral_end_nonzero}\n")
            f.write(f"  Spiral runs ending with zero charged vortices: {spiral_end_zero}\n")
            f.write("\n")
            f.write("Δt = t_form - t_lastPairDisappear\n")
            f.write(f"  type 1: {_summ(dt_type1)}\n")
            f.write(f"  type 3: {_summ(dt_type3)}\n")
            f.write("\n")
            f.write("Charged vortices at formation (type 2):\n")
            f.write(f"  type 2: {_summ(vort_at_form_type2)}\n")

        gz_wall = time.time() - gz_start
        print(f"=== DONE gridsize {gz}x{gz} | saved to {gz_dir} | wall={_fmt_hms(gz_wall)} ===", flush=True)

    total_wall = time.time() - t0
    print(f"\nDone all gridsizes. Total wall time: {_fmt_hms(total_wall)}")


if __name__ == "__main__":
    main()