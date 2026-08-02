"""
ncl_string_ensemble_v260625.py

Run Elinor's validated NCL-string pipeline on our stored gridsize-14 trajectories
(by injection -- proven equivalent to a fresh sim by ncl_string_adapter_v260625.py)
and produce the real Fig-4 NCL-string quantity as a function of fraction-of-
formation f, per fate, with error bands.

Per trajectory, per frame (0..end_frame): ncl_count, total_strings, percentage_ncl
from get_percent_of_strings_that_are_NCL(return_details=True). Each trajectory's
frame index is mapped to fraction f under BOTH clocks (Ttrue=true_formation_time,
Tfirst=first_recurrent_state_time) and resampled onto a common f-grid. Means and
std are aggregated per fate.

Injection mirrors analyse_trajectory() from the cell_4state entry point (skips
get_4_number_seq(cell_hist)); validated to reproduce fresh-sim NCL series exactly.

Runs from the folder containing Elinor_secrete_sense_module.py. Loads chunks
fully into RAM (no mmap; the chunks are Box-synced and mmap caused bus faults).

Usage:
    python ncl_string_ensemble_v260625.py --pilot          # ~8 per fate, timed, to gauge speed
    python ncl_string_ensemble_v260625.py --per-fate 500   # full run
    python ncl_string_ensemble_v260625.py --per-fate 500 --nproc 24
"""

import os
import csv
import time
import argparse
import importlib.util
import numpy as np
import multiprocessing as mp

MODULE_PATH = "Elinor_secrete_sense_module.py"

# stored trajectory chunks (Box). Add/adjust if needed.
CHUNK_DIR_CANDIDATES = [
    "<PATH_TO_CODE>/+v260529_full_trajectories",
    "<PATH_TO_OUTPUT>/+v260529_full_trajectories",
]
OUT_DIR = "<PATH_TO_OUTPUT>/ncl_string_ensemble_v260625"

GRIDSIZE = 14
NUM_CELLS = GRIDSIZE ** 2

# CA parameters (match the paper / her driver)
TMAX = 75000
RCELL = 0.2
A0 = 1.5
LATTICE_TYPE = "triangular"
PERIODIC_BC = [1, 1]
P0 = np.array([0.5, 0.55])
I_MIN = 0.01
DI = 0.05
M_MATRIX = np.array([[1, 1], [-1, 0]])
K_MATRIX = np.array([[3, 10], [11, 4]])
C_MATRIX = np.array([18, 16])
LAMB = np.array([1, 1.2])

FATE_NAMES = {1: "static", 2: "spiral", 3: "rectilinear"}
# common fraction grid for resampling each trajectory's series
F_GRID = np.linspace(0.0, 1.0, 21)   # 0.00, 0.05, ..., 1.00

_G = {}


def load_module(path=MODULE_PATH):
    spec = importlib.util.spec_from_file_location("esm", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def chunk_dir():
    for d in CHUNK_DIR_CANDIDATES:
        if os.path.isdir(d):
            return d
    raise FileNotFoundError("No chunk dir found; edit CHUNK_DIR_CANDIDATES")


def analyse_from_4state(esm, cell_4state):
    """Inject cell_4state and run analyse_trajectory() body from the injection point."""
    t = esm.CellularLattice(GRIDSIZE, NUM_CELLS, RCELL, A0, TMAX)
    t.init_lattice(GRIDSIZE, PERIODIC_BC, A0, RCELL, LATTICE_TYPE)
    t.init_general_parameters(K_MATRIX, C_MATRIX, M_MATRIX, LAMB)
    t.quiet_warnings = True
    t.cell_4state = np.asarray(cell_4state)
    t.tmax = t.cell_4state.shape[1]
    t.cell_4phase = esm.get_phase_seq_4(t.cell_4state)
    t.state_fractions = esm.compute_4_state_fractions(t)
    t.first_recurrent_state_time = esm.get_first_recurrent_state_time(t)
    t.repeated_time = esm.get_time_that_is_repeated(t)
    t.final_pattern_type = esm.get_final_configuration_type(t)
    if t.number_of_nn == 6:
        t.vortex_cores, t.Hxy, t.Hxy_adjusted = esm.compute_phase_differences(t)
        t.vortex_cores_labeled, t.n_vortex = esm.label_triangular_lattice(t)
        t.xcom, t.ycom = esm.compute_centroid(t)
        t.closest_pos, t.charges, t.xmin_core, t.ymin_core = esm.compute_smallest_movement(t)
        t.idx_annihilation_moments, t.idx_annihilation_moments_core_count = esm.compute_annihilation_moments(t)
    return t


def resample_to_grid(values, T):
    """Map per-frame `values` (frames 0..len-1) onto F_GRID using fraction = frame/T."""
    n = len(values)
    if n == 0 or T <= 0:
        return np.full(F_GRID.shape, np.nan)
    frac = np.arange(n) / float(T)
    # interpolate values at F_GRID; outside observed range -> nan
    out = np.interp(F_GRID, frac, values, left=np.nan, right=np.nan)
    return out


def _process_traj(idx):
    """Worker: idx into the global list of (cell_4state, fate, Ttrue, Tfirst)."""
    esm = _G["esm"]
    cs, fate, Ttrue, Tfirst = _G["traj"][idx]
    try:
        t = analyse_from_4state(esm, cs)
        details = t.get_percent_of_strings_that_are_NCL(include_zero_charge=False, return_details=True)
        nframes = len(details)
        ncl_count = np.array([d["ncl_count"] for d in details], dtype=float)
        total_strings = np.array([d["total_strings"] for d in details], dtype=float)
        pct = np.array([d["percentage_ncl"] for d in details], dtype=float)
    except Exception as e:
        return dict(idx=idx, fate=int(fate), ok=False, err=str(e))
    # exact Fig-4e/f quantity: per-frame 0/1 -- is the FINAL pair on an NCL string.
    # wrapped separately so a failure here doesn't lose the ncl_count series.
    try:
        fpn = np.asarray(t.final_pair_on_ncl(include_zero_charge=False), dtype=float)
    except Exception:
        fpn = np.array([], dtype=float)
    res = dict(idx=idx, fate=int(fate), ok=True, end_frame=nframes,
               Ttrue=int(Ttrue), Tfirst=int(Tfirst))
    # raw per-frame series (absolute timestep axis) for later alignment
    res["raw_ncl_count"] = ncl_count.astype(np.float32)
    res["raw_total_strings"] = total_strings.astype(np.float32)
    res["raw_finalpair_ncl"] = fpn.astype(np.float32)
    for clock_name, T in (("Ttrue", Ttrue), ("Tfirst", Tfirst)):
        res[f"ncl_count_{clock_name}"] = resample_to_grid(ncl_count, T)
        res[f"total_strings_{clock_name}"] = resample_to_grid(total_strings, T)
        res[f"pct_{clock_name}"] = resample_to_grid(pct, T)
        res[f"finalpair_ncl_{clock_name}"] = resample_to_grid(fpn, T) if fpn.size else np.full(F_GRID.shape, np.nan)
    return res


def sample_trajectories(per_fate, seed=0):
    """Load chunks one at a time (full RAM, no mmap), collect per_fate trajectories
    of each fate as (cell_4state (196,nframes), fate, Ttrue, Tfirst)."""
    d = chunk_dir()
    files = sorted(f for f in os.listdir(d) if f.startswith("v260529_chunk_") and f.endswith(".npz"))
    rng = np.random.default_rng(seed)
    want = {1: per_fate, 2: per_fate, 3: per_fate}
    have = {1: 0, 2: 0, 3: 0}
    out = []
    for fname in files:
        if all(have[k] >= want[k] for k in (1, 2, 3)):
            break
        z = np.load(os.path.join(d, fname), allow_pickle=True)  # full load, no mmap
        data = np.asarray(z["data"])                  # (1000, 30000, 196) int8
        fate = np.asarray(z["final_pattern_type"]).ravel().astype(int)
        Ttrue = np.asarray(z["true_formation_time"]).ravel().astype(int)
        Tfirst = np.asarray(z["first_recurrent_state_time"]).ravel().astype(int)
        for k in (1, 2, 3):
            if have[k] >= want[k]:
                continue
            idxs = np.where(fate == k)[0]
            rng.shuffle(idxs)
            for i in idxs:
                if have[k] >= want[k]:
                    break
                cs = data[i].T.copy()             # (196, 30000) int -> cell_4state layout
                out.append((cs.astype(int), int(k), int(Ttrue[i]), int(Tfirst[i])))
                have[k] += 1
        print(f"  after {fname}: have static={have[1]} spiral={have[2]} rect={have[3]}", flush=True)
        del data, z
    return out


def aggregate(results):
    """Mean +/- std over trajectories per fate, per clock, per measure, on F_GRID."""
    rows = []
    measures = ["ncl_count", "total_strings", "pct", "finalpair_ncl"]
    for clock in ("Ttrue", "Tfirst"):
        for fate in (1, 2, 3):
            subset = [r for r in results if r.get("ok") and r["fate"] == fate]
            n = len(subset)
            for meas in measures:
                key = f"{meas}_{clock}"
                avail = [r[key] for r in subset if key in r]
                M = np.vstack(avail) if avail else np.zeros((0, len(F_GRID)))
                mean = np.nanmean(M, axis=0) if M.shape[0] else np.full(len(F_GRID), np.nan)
                std = np.nanstd(M, axis=0) if M.shape[0] else np.full(len(F_GRID), np.nan)
                cnt = np.sum(~np.isnan(M), axis=0) if M.shape[0] else np.zeros(len(F_GRID))
                for j, f in enumerate(F_GRID):
                    rows.append([clock, FATE_NAMES[fate], fate, meas, round(float(f), 3),
                                 float(mean[j]), float(std[j]), int(cnt[j]), n])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true", help="~8 per fate, timed")
    ap.add_argument("--per-fate", type=int, default=500)
    ap.add_argument("--nproc", type=int, default=0)
    args = ap.parse_args()
    per_fate = 8 if args.pilot else args.per_fate
    nproc = args.nproc if args.nproc > 0 else mp.cpu_count()

    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Loading {per_fate} trajectories per fate ...", flush=True)
    t0 = time.time()
    traj = sample_trajectories(per_fate)
    print(f"Loaded {len(traj)} trajectories in {time.time()-t0:.0f}s", flush=True)

    esm = load_module()
    _G["esm"] = esm
    _G["traj"] = traj

    ctx = mp.get_context("fork")
    t1 = time.time()
    results = []
    done = 0
    with ctx.Pool(processes=nproc) as pool:
        for r in pool.imap_unordered(_process_traj, range(len(traj))):
            results.append(r)
            done += 1
            if done % 25 == 0 or done == len(traj):
                el = time.time() - t1
                rate = el / done
                print(f"  {done}/{len(traj)}  ({el:.0f}s, {rate:.2f}s/traj, "
                      f"~{rate*len(traj)/60:.1f}min total)", flush=True)
    n_fail = sum(1 for r in results if not r.get("ok"))
    print(f"\nProcessed {len(results)} ({n_fail} failed) in {(time.time()-t1)/60:.1f} min", flush=True)
    if n_fail:
        for r in results:
            if not r.get("ok"):
                print("  FAIL idx", r["idx"], "fate", r["fate"], ":", r.get("err"))
                break

    rows = aggregate(results)
    csv_path = os.path.join(OUT_DIR, f"ncl_string_vs_f_perfate{'_pilot' if args.pilot else ''}.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["clock", "fate_name", "fate", "measure", "fraction",
                    "mean", "std", "n_contrib", "n_traj"])
        w.writerows(rows)
    print("Wrote:", csv_path)

    # save raw per-frame series per trajectory (object arrays; ragged lengths) for
    # absolute-timestep alignment (forward-t, backward-Tfirst, backward-Ttrue).
    ok = [r for r in results if r.get("ok")]
    raw_path = os.path.join(OUT_DIR, f"ncl_raw_series{'_pilot' if args.pilot else ''}.npz")
    np.savez_compressed(
        raw_path,
        fate=np.array([r["fate"] for r in ok], dtype=np.int8),
        Ttrue=np.array([r["Ttrue"] for r in ok], dtype=np.int32),
        Tfirst=np.array([r["Tfirst"] for r in ok], dtype=np.int32),
        end_frame=np.array([r["end_frame"] for r in ok], dtype=np.int32),
        finalpair_ncl=np.array([r["raw_finalpair_ncl"] for r in ok], dtype=object),
        ncl_count=np.array([r["raw_ncl_count"] for r in ok], dtype=object),
        total_strings=np.array([r["raw_total_strings"] for r in ok], dtype=object),
    )
    print("Wrote:", raw_path)

    # quick print: ncl_count vs f, Ttrue, per fate
    print("\nNCL-string count vs f (Ttrue, mean):")
    for fate in (1, 2, 3):
        vals = [(r[4], r[5]) for r in rows if r[0] == "Ttrue" and r[2] == fate and r[3] == "ncl_count"]
        line = "  ".join(f"{f:.2f}:{m:.2f}" for f, m in vals if not np.isnan(m))
        print(f"  {FATE_NAMES[fate]:>11s}: {line}")


if __name__ == "__main__":
    main()
