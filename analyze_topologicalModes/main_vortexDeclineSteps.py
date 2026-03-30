# main_vortexDeclineSteps.py
"""
Runs an ensemble of CA simulations in parallel and plots the descending-staircase
dynamics of vortex counts over time. One-timestep detection artefacts (blips) are
removed before plotting. For each lattice size, produces:

  - Vortex count vs. time (blips removed): total count, median and IQR across runs,
    with a representative individual trajectory overlaid.
  - Per-type vortex counts vs. time: n_pos (+1), n_neg (-1), n_zero (0), and
    n_total, shown as median with IQR shading.
  - Net topological charge (n_pos - n_neg) vs. time.
  - A blip-statistics report confirming the correction is conservative.
  - Raw per-run arrays saved as a compressed NumPy archive.

Blip definition: a one-timestep +1 spike in n_total where n(t) = n(t-1)+1 and
n(t+1) = n(t-1). Correction is applied only for t >= t_blip (default 150),
leaving early transient dynamics untouched.
"""

from __future__ import annotations

import os
import time
import argparse
import numpy as np
import matplotlib
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
# Helpers: blip cleaning on a single trajectory
# -----------------------------
def remove_one_step_total_blips(n_total: np.ndarray, t_blip_start: int = 150) -> np.ndarray:
    """
    Return a copy where one-step +1 blips (for t>=t_blip_start) are removed by setting n(t)=n(t-1).
    Definition: n(t)=n(t-1)+1 and n(t+1)=n(t-1).

    We do NOT touch early times (<t_blip_start).
    """
    n = n_total.astype(np.int32).copy()
    tmax = len(n)
    for t in range(max(t_blip_start, 1), tmax - 1):
        if (n[t] == n[t-1] + 1) and (n[t+1] == n[t-1]):
            n[t] = n[t-1]
    return n.astype(n_total.dtype)


# -----------------------------
# One simulation -> trajectories
# -----------------------------
def run_one_and_measure(gridsize: int, tmax: int, seed: int | None) -> dict:
    """
    Run CA once and return:
      n_total(t): # labeled cores (connected regions)
      n_pos(t), n_neg(t), n_zero(t): per-core charge classification
      net_charge(t) = n_pos - n_neg
      blip_times (t>=t_blip_start) for one-step +1 in n_total

    Returns dict of 1D arrays length tmax + small metadata.
    """
    if seed is not None:
        np.random.seed(int(seed))

    num_cells = gridsize ** 2
    traj = CA.CellularLattice(gridsize, num_cells, RCELL, A0, tmax)
    traj.init_lattice(gridsize, PERIODIC_BC, A0, RCELL, TYPE_LATTICE)
    traj.init_general_parameters(K_MATRIX, C_MATRIX, M_MATRIX, LAMB)

    traj.init_cell_state(P0, I_MIN, DI)
    traj.convert_currentBinaryState_to_4cell_state()
    traj.run_model(tmax)

    # Minimal vortex pipeline + phase sequence
    traj.cell_4state = CA.get_4_number_seq(traj.cell_hist)
    traj.cell_4phase = CA.get_phase_seq_4(traj.cell_4state)

    traj.vortex_cores = CA.compute_phase_differences(traj)
    traj.vortex_cores_labeled, n_total = CA.label_triangular_lattice(traj)

    # Explicit per-core charge + counts
    charges_int, n_pos, n_neg, n_zero, n_tot2, net = CA.compute_core_charges_and_type_counts(traj)

    # Sanity: n_tot2 should match n_total from label
    # (rarely, a degenerate contour could cause oddities; if so, trust n_total)
    n_tot2 = n_tot2.astype(np.int16)
    n_total = n_total.astype(np.int16)

    return {
        "n_total": n_total,
        "n_pos": n_pos.astype(np.int16),
        "n_neg": n_neg.astype(np.int16),
        "n_zero": n_zero.astype(np.int16),
        "net": net.astype(np.int16),
    }


def _worker(job):
    gs, tmax, seed = job
    return run_one_and_measure(gs, tmax, seed)


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
    progress_every_runs: int = 50,
) -> dict:
    """
    Returns dict of arrays, each shape (n_runs, tmax):
      n_total, n_pos, n_neg, n_zero, net
    """
    if base_seed is None:
        seeds = [None] * n_runs
    else:
        seeds = list(range(int(base_seed), int(base_seed) + n_runs))

    jobs = [(gridsize, tmax, seeds[i]) for i in range(n_runs)]

    out = {
        "n_total": np.zeros((n_runs, tmax), dtype=np.int16),
        "n_pos":   np.zeros((n_runs, tmax), dtype=np.int16),
        "n_neg":   np.zeros((n_runs, tmax), dtype=np.int16),
        "n_zero":  np.zeros((n_runs, tmax), dtype=np.int16),
        "net":     np.zeros((n_runs, tmax), dtype=np.int16),
    }

    if workers == 1:
        t0 = time.time()
        last_print = 0
        for i, jb in enumerate(jobs, start=1):
            res = _worker(jb)
            for k in out:
                out[k][i-1, :] = res[k]
            if i - last_print >= progress_every_runs or i == n_runs:
                dt = (time.time() - t0) / 60
                print(f"[{gridsize}x{gridsize}] completed {i}/{n_runs} "
                      f"({100*i/n_runs:.1f}%) | elapsed {dt:.1f} min",
                      flush=True)
                last_print = i
        return out

    if chunksize is None:
        chunksize = max(1, n_runs // (workers * 50))

    ctx = mp.get_context("spawn")

    print(f"[{gridsize}x{gridsize}] starting Pool(workers={workers}, chunksize={chunksize}) ...", flush=True)
    t0 = time.time()
    last_print_count = 0

    with ctx.Pool(processes=workers, maxtasksperchild=200) as pool:
        iterator = pool.imap_unordered(_worker, jobs, chunksize=chunksize)

        i = 0
        for res in iterator:
            for k in out:
                out[k][i, :] = res[k]
            i += 1

            if i - last_print_count >= progress_every_runs or i == n_runs:
                dt = (time.time() - t0) / 60
                print(f"[{gridsize}x{gridsize}] completed {i}/{n_runs} "
                      f"({100*i/n_runs:.1f}%) | elapsed {dt:.1f} min",
                      flush=True)
                last_print_count = i

            if i >= n_runs:
                break

    return out


# -----------------------------
# Plotting utilities
# -----------------------------
def _median_iqr(x: np.ndarray):
    med = np.median(x, axis=0)
    p25 = np.percentile(x, 25, axis=0)
    p75 = np.percentile(x, 75, axis=0)
    return med, p25, p75


def save_steps_figure(n_total: np.ndarray, gridsize: int, tmax: int, outdir: str,
                      n_runs: int, t_blip_start: int, n_show: int = 20) -> str:
    """
    Plot total vortex count vs. time with one-step blips removed for t >= t_blip_start.
    Shows individual trajectories (n_show randomly selected), median, and IQR shading.
    """
    cleaned = np.zeros_like(n_total)
    for i in range(n_total.shape[0]):
        cleaned[i, :] = remove_one_step_total_blips(n_total[i, :], t_blip_start=t_blip_start)

    areas = cleaned.sum(axis=1)
    rep_idx = int(np.argsort(areas)[len(areas)//2])

    rng = np.random.default_rng(0)
    show_idx = rng.choice(n_runs, size=min(n_show, n_runs), replace=False)

    t = np.arange(tmax)
    med, p25, p75 = _median_iqr(cleaned)

    fig = plt.figure(figsize=(6.8, 4.6))
    ax = fig.add_subplot(111)

    for i in show_idx:
        ax.step(t, cleaned[i, :], where="post", linewidth=0.8, alpha=0.25)

    ax.fill_between(t, p25, p75, step="post", alpha=0.25)
    ax.step(t, med, where="post", linewidth=2.0, label="median (IQR shaded)")
    ax.step(t, cleaned[rep_idx, :], where="post", linewidth=2.5, label="representative run")

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Total # of vortices")
    ax.set_title(f"Grid {gridsize}×{gridsize} | n={n_runs} | tmax={tmax}")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=9)

    pdf_path = os.path.join(outdir, f"vortexCount_steps_gridsize{gridsize}_tmax{tmax}_n{n_runs}.pdf")
    fig.tight_layout()
    fig.savefig(pdf_path)
    plt.close(fig)
    return pdf_path


def save_vortex_type_counts_pdf(n_pos, n_neg, n_zero, n_total, gridsize, tmax, outdir, n_runs):
    t = np.arange(tmax)

    fig = plt.figure(figsize=(7.2, 4.8))
    ax = fig.add_subplot(111)

    for arr, lab in [(n_pos, "n_pos (+1)"), (n_neg, "n_neg (-1)"), (n_zero, "n_zero (0)"), (n_total, "n_total")]:
        med, p25, p75 = _median_iqr(arr)
        ax.fill_between(t, p25, p75, alpha=0.18)
        ax.plot(t, med, linewidth=2.0, label=lab)

    ax.set_xlabel("Timestep")
    ax.set_ylabel("# cores")
    ax.set_title(f"Grid {gridsize}×{gridsize} | n={n_runs} | tmax={tmax} | median (IQR shaded)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=9)

    pdf_path = os.path.join(outdir, f"vortexTypeCounts_gridsize{gridsize}_tmax{tmax}_n{n_runs}.pdf")
    fig.tight_layout()
    fig.savefig(pdf_path)
    plt.close(fig)
    return pdf_path


def save_net_charge_pdf(net, gridsize, tmax, outdir, n_runs):
    t = np.arange(tmax)
    med, p25, p75 = _median_iqr(net)

    fig = plt.figure(figsize=(7.2, 4.8))
    ax = fig.add_subplot(111)

    ax.fill_between(t, p25, p75, alpha=0.25)
    ax.plot(t, med, linewidth=2.5, label="net_charge = n_pos - n_neg")

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Net topological charge")
    ax.set_title(f"Grid {gridsize}×{gridsize} | n={n_runs} | tmax={tmax} | median (IQR shaded)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=10)

    pdf_path = os.path.join(outdir, f"netCharge_gridsize{gridsize}_tmax{tmax}_n{n_runs}.pdf")
    fig.tight_layout()
    fig.savefig(pdf_path)
    plt.close(fig)
    return pdf_path


# -----------------------------
# Blip analysis + report
# -----------------------------
def blip_report_for_grid(n_total, n_pos, n_neg, n_zero, t_blip_start: int):
    """
    Compute blip stats across all runs.
    Returns a dict with totals and per-run rates.
    """
    n_runs, tmax = n_total.shape
    analysis_len = tmax - t_blip_start

    blips_per_run = np.zeros(n_runs, dtype=np.int32)
    zero_blips_per_run = np.zeros(n_runs, dtype=np.int32)
    other_blips_per_run = np.zeros(n_runs, dtype=np.int32)

    for r in range(n_runs):
        blip_times = CA.detect_one_step_total_blips(n_total[r, :], t_blip_start=t_blip_start)
        cls = CA.classify_blips(blip_times, n_pos[r, :], n_neg[r, :], n_zero[r, :], n_total[r, :])

        blips_per_run[r] = cls["n_blips"]
        zero_blips_per_run[r] = cls["n_zero_vortex_blips"]
        other_blips_per_run[r] = cls["n_other_blips"]

    total_blips = int(np.sum(blips_per_run))
    total_zero = int(np.sum(zero_blips_per_run))
    total_other = int(np.sum(other_blips_per_run))

    # Per-run blip rate: fraction of analysis timesteps that contain a blip
    rate_per_run = blips_per_run / float(max(1, analysis_len))

    return {
        "t_blip_start": t_blip_start,
        "analysis_len": analysis_len,
        "total_blips": total_blips,
        "total_zero_blips": total_zero,
        "total_other_blips": total_other,
        "mean_rate_per_run": float(np.mean(rate_per_run)),
        "median_rate_per_run": float(np.median(rate_per_run)),
        "p95_rate_per_run": float(np.percentile(rate_per_run, 95)),
        "runs_with_any_blip": int(np.sum(blips_per_run > 0)),
        "blips_per_run": blips_per_run,
        "zero_blips_per_run": zero_blips_per_run,
        "other_blips_per_run": other_blips_per_run,
    }


def write_blip_report(path, gridsize, tmax, n_runs, report):
    with open(path, "w") as f:
        f.write(f"Grid {gridsize}x{gridsize} | n_runs={n_runs} | tmax={tmax}\n")
        f.write(f"Blip detection uses t >= {report['t_blip_start']} (analysis_len={report['analysis_len']})\n\n")

        f.write("Definition: one-step +1 blip in n_total(t): n(t)=n(t-1)+1 and n(t+1)=n(t-1)\n")
        f.write("Classification: ZERO_VORTEX_BLIP if dn_zero=+1 and dn_pos=dn_neg=0 at blip step.\n\n")

        f.write(f"Total blips: {report['total_blips']}\n")
        f.write(f"  ZERO_VORTEX_BLIP: {report['total_zero_blips']}\n")
        f.write(f"  OTHER:           {report['total_other_blips']}\n\n")

        f.write(f"Runs with >=1 blip: {report['runs_with_any_blip']} / {n_runs}\n\n")

        f.write("Per-run blip-rate (blip timesteps / analysis timesteps):\n")
        f.write(f"  mean   = {report['mean_rate_per_run']:.6g}\n")
        f.write(f"  median = {report['median_rate_per_run']:.6g}\n")
        f.write(f"  95th%%  = {report['p95_rate_per_run']:.6g}\n\n")

        # Also print raw counts summary
        f.write("Per-run blip-count summary:\n")
        f.write(f"  mean blips/run   = {np.mean(report['blips_per_run']):.4g}\n")
        f.write(f"  max  blips/run   = {np.max(report['blips_per_run'])}\n")
        f.write(f"  mean zero/run    = {np.mean(report['zero_blips_per_run']):.4g}\n")
        f.write(f"  mean other/run   = {np.mean(report['other_blips_per_run']):.4g}\n")


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tmax", type=int, default=15000)
    parser.add_argument("--n_runs", type=int, default=1000)
    parser.add_argument("--grids", type=int, nargs="+", default=[14, 16, 18, 20])
    parser.add_argument("--outdir", type=str, default="output_vortexDeclineSteps")
    parser.add_argument("--seed", type=int, default=0, help="Base seed. Use --seed -1 to disable seeding.")
    parser.add_argument("--workers", type=int, default=0, help="0 => auto (cpu_count-1).")
    parser.add_argument("--chunksize", type=int, default=0, help="0 => auto.")
    parser.add_argument("--n_show", type=int, default=20)
    parser.add_argument("--t_blip", type=int, default=150, help="Only detect/remove blips for t>=t_blip.")
    args = parser.parse_args()

    tmax = int(args.tmax)
    n_runs = int(args.n_runs)
    grids = [int(g) for g in args.grids]
    outdir = args.outdir
    n_show = int(args.n_show)
    t_blip = int(args.t_blip)

    base_seed = None if args.seed == -1 else int(args.seed)

    cpu = mp.cpu_count()
    if args.workers is None or int(args.workers) == 0:
        workers = max(1, cpu - 1)
    else:
        workers = max(1, int(args.workers))

    chunksize = None if int(args.chunksize) == 0 else int(args.chunksize)

    os.makedirs(outdir, exist_ok=True)

    print("=== Vortex decline analysis (parallel) ===")
    print(f"cpu_count={cpu} | workers={workers} | tmax={tmax} | n_runs={n_runs} | grids={grids}")
    print("seeding:", "OFF" if base_seed is None else f"base_seed={base_seed} (seed_i=base_seed+i)")
    print("chunksize:", "auto" if chunksize is None else chunksize)
    print(f"blip detection/removal for t >= {t_blip}")

    for gs in grids:
        print(f"\n--- Grid {gs}×{gs} ---")
        t0 = time.time()

        out = run_many_parallel(gs, tmax, n_runs, base_seed, workers, chunksize)

        n_total = out["n_total"]
        n_pos   = out["n_pos"]
        n_neg   = out["n_neg"]
        n_zero  = out["n_zero"]
        net     = out["net"]

        # Apply blip cleaning to n_total once; use cleaned version for all plots
        n_total_clean = np.zeros_like(n_total)
        for i in range(n_total.shape[0]):
            n_total_clean[i, :] = remove_one_step_total_blips(n_total[i, :], t_blip_start=t_blip)

        pdf_steps = save_steps_figure(n_total, gs, tmax, outdir, n_runs, t_blip_start=t_blip, n_show=n_show)
        pdf_types = save_vortex_type_counts_pdf(n_pos, n_neg, n_zero, n_total_clean, gs, tmax, outdir, n_runs)
        pdf_net   = save_net_charge_pdf(net, gs, tmax, outdir, n_runs)

        rep = blip_report_for_grid(n_total, n_pos, n_neg, n_zero, t_blip_start=t_blip)
        rep_path = os.path.join(outdir, f"blipReport_after{t_blip}_gridsize{gs}_tmax{tmax}_n{n_runs}.txt")
        write_blip_report(rep_path, gs, tmax, n_runs, rep)

        npz_path = os.path.join(outdir, f"raw_vortexAndChargeTraj_gridsize{gs}_tmax{tmax}_n{n_runs}.npz")
        np.savez_compressed(
            npz_path,
            gridsize=gs,
            tmax=tmax,
            n_runs=n_runs,
            n_total=n_total,
            n_pos=n_pos,
            n_neg=n_neg,
            n_zero=n_zero,
            net_charge=net,
            t_blip_start=t_blip,
            total_blips=rep["total_blips"],
            total_zero_blips=rep["total_zero_blips"],
            total_other_blips=rep["total_other_blips"],
            mean_blip_rate_per_run=rep["mean_rate_per_run"],
        )

        dt = time.time() - t0
        print(f"Saved PDF (steps):  {pdf_steps}")
        print(f"Saved PDF (types):  {pdf_types}")
        print(f"Saved PDF (net):    {pdf_net}")
        print(f"Saved blip report:  {rep_path}")
        print(f"Saved raw arrays:   {npz_path}")
        print(f"Done in {dt/60:.2f} min")

    print("\nAll done.")


if __name__ == "__main__":
    main()