"""
string_mask_extract_v260625.py   [STAGE 1 of the single-color string-configuration experiment]

For each fraction split (Ttrue), for each config, inject the single snapshot into
a fresh CellularLattice, run the per-frame vortex+string pipeline, and extract
PER-COLOR PER-CELL boolean masks:
  - on_any_string[c]  : cell is on a vortex-connecting string of color c
  - on_ncl_string[c]  : cell is on a NON-CONTRACTIBLE vortex-connecting string of color c

Single-frame definition (correct for isolated snapshots, matches what the CNN sees):
  strings = get_all_cluster_connections(frame=0, include_zero_charge=False).best_results
            -> each entry: {'path': [cells], 'state': color, ...}  (opposite-charge pairs only)
  a string is NCL if its path shares cells with a non-contractible loop of that color
            (find_noncontractible_loops(0, phase_value=color), nonzero winding).

Outputs (per fraction) -> string_masks_v260625/masks_Ttrue_f{f:.2f}.npz
  any_string : (N, 4, 196) uint8   [color order 1,2,3,4 -> index 0..3]
  ncl_string : (N, 4, 196) uint8
  (aligned row-for-row with X_train_pool.npy of that split)
Also processes X_val.npy and X_test.npy -> masks_Ttrue_f{f:.2f}_{val,test}.npz

Run from the folder with Elinor_secrete_sense_module.py.

Usage:
    python string_mask_extract_v260625.py --validate          # small batch + sanity checks
    python string_mask_extract_v260625.py --split train        # full train_pool, all f
    python string_mask_extract_v260625.py --split val
    python string_mask_extract_v260625.py --split test
    python string_mask_extract_v260625.py --split train --nproc 24
"""

import os
import time
import argparse
import importlib.util
import numpy as np
import multiprocessing as mp

MODULE_PATH = "Elinor_secrete_sense_module.py"
SPLITS_ROOT = "<PATH_TO_OUTPUT>/ml_splits_v260529"
OUT_DIR = "<PATH_TO_OUTPUT>/string_masks_v260625"
CLOCK = "Ttrue"
FRACTIONS = [0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
GRIDSIZE = 14
NUM_CELLS = GRIDSIZE ** 2
COLORS = [1, 2, 3, 4]

# CA parameters (match the validated injection)
TMAX = 75000
RCELL = 0.2; A0 = 1.5; LATTICE_TYPE = "triangular"; PERIODIC_BC = [1, 1]
M_MATRIX = np.array([[1, 1], [-1, 0]]); K_MATRIX = np.array([[3, 10], [11, 4]])
C_MATRIX = np.array([18, 16]); LAMB = np.array([1, 1.2])

_G = {}


def load_module(path=MODULE_PATH):
    spec = importlib.util.spec_from_file_location("esm", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def build_object(esm, snapshot_196):
    """Inject a single snapshot as a 2-frame (duplicated) cell_4state and run the
    full per-snapshot vortex+charge+string pipeline. Vortices/strings/NCL-strings
    are snapshot properties (core island + contour winding = charge). We duplicate
    the frame to (196,2) ONLY to satisfy the module's 2-D shape assumptions
    (compute_phase_differences' squeeze, label_triangular_lattice, charges); the
    physics is per-frame and we read frame 0. NO 3<->4 swap: the ml_splits are in
    the same encoding as the chunks (verified via global_index/t_extracted: split
    config == chunk[row, t_extracted] exactly, no swap)."""
    t = esm.CellularLattice(GRIDSIZE, NUM_CELLS, RCELL, A0, TMAX)
    t.init_lattice(GRIDSIZE, PERIODIC_BC, A0, RCELL, LATTICE_TYPE)
    t.init_general_parameters(K_MATRIX, C_MATRIX, M_MATRIX, LAMB)
    t.quiet_warnings = True
    snap = np.asarray(snapshot_196, dtype=int)
    cs = np.repeat(snap.reshape(NUM_CELLS, 1), 2, axis=1)   # (196, 2) shape fix
    t.cell_4state = cs
    t.tmax = 2
    # phase map: state 1->0, 2->pi/2, 4->pi, 3->3pi/2 (canonical)
    phase = np.zeros_like(cs, dtype=float)
    phase[cs == 1] = 0.0
    phase[cs == 2] = 0.5 * np.pi
    phase[cs == 4] = 1.0 * np.pi
    phase[cs == 3] = 1.5 * np.pi
    t.cell_4phase = phase
    # full per-snapshot vortex pipeline incl. charges (winding around contours)
    t.vortex_cores, t.Hxy, t.Hxy_adjusted = esm.compute_phase_differences(t)
    t.vortex_cores_labeled, t.n_vortex = esm.label_triangular_lattice(t)
    t.xcom, t.ycom = esm.compute_centroid(t)
    t.closest_pos, t.charges, t.xmin_core, t.ymin_core = esm.compute_smallest_movement(t)
    return t


def extract_masks(esm, snapshot_196):
    """Return (any_string (4,196) uint8, ncl_string (4,196) uint8) for one snapshot."""
    any_mask = np.zeros((4, NUM_CELLS), dtype=np.uint8)
    ncl_mask = np.zeros((4, NUM_CELLS), dtype=np.uint8)
    try:
        t = build_object(esm, snapshot_196)
    except Exception:
        return any_mask, ncl_mask  # degenerate (e.g. t=0): no vortices -> empty masks
    # strings (opposite-charge pairs only)
    try:
        best = t.get_all_cluster_connections(0, include_zero_charge=False)
    except Exception:
        return any_mask, ncl_mask
    # non-contractible loop cells per color (for NCL test)
    ncl_cells_by_color = {}
    for c in COLORS:
        cells = set()
        try:
            loops = t.find_noncontractible_loops(0, phase_value=c)
            for lp in loops:
                cells.update(int(x) for x in lp['nodes'])
        except Exception:
            pass
        ncl_cells_by_color[c] = cells
    # walk strings
    for key, info in best.items():
        path = info.get('path'); state = info.get('state')
        if path is None or state not in (1, 2, 3, 4):
            continue
        ci = state - 1
        pcells = [int(x) for x in path]
        for cell in pcells:
            if 0 <= cell < NUM_CELLS:
                any_mask[ci, cell] = 1
        # NCL if this string's path shares any cell with a non-contractible loop of its color
        if ncl_cells_by_color[state].intersection(pcells):
            for cell in pcells:
                if 0 <= cell < NUM_CELLS:
                    ncl_mask[ci, cell] = 1
    # No encoding swap: splits are chunk-encoded, so mask color index c already
    # aligns with split-encoding color c (what the CNN sees in Stage 2).
    return any_mask, ncl_mask


def _init_worker():
    _G["esm"] = load_module()


def _process_range(args):
    start, end = args
    X = _G["X"]
    a_out = np.zeros((end - start, 4, NUM_CELLS), dtype=np.uint8)
    n_out = np.zeros((end - start, 4, NUM_CELLS), dtype=np.uint8)
    esm = _G["esm"]
    for r, i in enumerate(range(start, end)):
        a, n = extract_masks(esm, X[i])
        a_out[r] = a; n_out[r] = n
    return start, end, a_out, n_out


def process_split(split, nproc):
    fname = {"train": "X_train_pool.npy", "val": "X_val.npy", "test": "X_test.npy"}[split]
    suffix = {"train": "", "val": "_val", "test": "_test"}[split]
    os.makedirs(OUT_DIR, exist_ok=True)
    ctx = mp.get_context("fork")
    for frac in FRACTIONS:
        sd = os.path.join(SPLITS_ROOT, f"ml_splits_v260529_{CLOCK}_f{frac:.2f}")
        xpath = os.path.join(sd, fname)
        if not os.path.isfile(xpath):
            print(f"[f={frac:.2f}] MISSING {xpath}", flush=True); continue
        X = np.load(xpath)
        n = X.shape[0]
        print(f"[f={frac:.2f}] {split}: {n} configs on {nproc} cores", flush=True)
        _G["X"] = X
        any_all = np.zeros((n, 4, NUM_CELLS), dtype=np.uint8)
        ncl_all = np.zeros((n, 4, NUM_CELLS), dtype=np.uint8)
        nchunks = nproc * 4
        bounds = np.linspace(0, n, nchunks + 1).astype(int)
        ranges = [(int(bounds[i]), int(bounds[i + 1])) for i in range(nchunks) if bounds[i + 1] > bounds[i]]
        t1 = time.time(); done = 0
        with ctx.Pool(processes=nproc, initializer=_init_worker) as pool:
            for start, end, a_out, n_out in pool.imap_unordered(_process_range, ranges):
                any_all[start:end] = a_out; ncl_all[start:end] = n_out
                done += (end - start)
                if done % 50000 < (end - start):
                    el = time.time() - t1
                    print(f"    {done}/{n} ({el:.0f}s, ~{el/done*n/3600:.1f}h for this f)", flush=True)
        outp = os.path.join(OUT_DIR, f"masks_{CLOCK}_f{frac:.2f}{suffix}.npz")
        np.savez_compressed(outp, any_string=any_all, ncl_string=ncl_all)
        print(f"  [f={frac:.2f}] wrote {outp} ({time.time()-t1:.0f}s)", flush=True)


def validate():
    esm = load_module()
    print("VALIDATION: extracting masks on a small batch and checking sanity\n")
    for frac in [0.00, 0.50, 0.90]:
        sd = os.path.join(SPLITS_ROOT, f"ml_splits_v260529_{CLOCK}_f{frac:.2f}")
        xpath = os.path.join(sd, "X_train_pool.npy")
        if not os.path.isfile(xpath):
            print(f"[f={frac:.2f}] MISSING {xpath}"); continue
        X = np.load(xpath, mmap_mode="r")[:300]
        X = np.array(X)
        any_dens = np.zeros(4); ncl_dens = np.zeros(4)
        color_ok = True; subset_ok = True
        nb = X.shape[0]
        for i in range(nb):
            a, n = extract_masks(esm, X[i])
            snap = X[i]
            for ci, c in enumerate(COLORS):
                # check 1: string cells of color c must actually be color c
                cells_a = np.where(a[ci] == 1)[0]
                if cells_a.size and not np.all(snap[cells_a] == c):
                    color_ok = False
                # check 2: NCL-string subset of any-string
                if np.any((n[ci] == 1) & (a[ci] == 0)):
                    subset_ok = False
            any_dens += a.sum(axis=1) / nb
            ncl_dens += n.sum(axis=1) / nb
        print(f"[f={frac:.2f}] n={nb}")
        print(f"   mean any-string cells / config, by color: {np.round(any_dens,2)}")
        print(f"   mean NCL-string cells / config, by color: {np.round(ncl_dens,2)}")
        print(f"   check string-cells-are-that-color: {'PASS' if color_ok else 'FAIL'}")
        print(f"   check NCL-subset-of-any-string:    {'PASS' if subset_ok else 'FAIL'}")
        if frac == 0.00:
            empt = (any_dens.sum() < 0.5)
            print(f"   check ~empty at f=0 (no vortices): {'PASS' if empt else 'FAIL (unexpected strings at t=0)'}")
        print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--split", choices=["train", "val", "test"])
    ap.add_argument("--nproc", type=int, default=0)
    args = ap.parse_args()
    if args.validate:
        validate(); return
    if not args.split:
        print("specify --validate or --split {train,val,test}"); return
    nproc = args.nproc if args.nproc > 0 else mp.cpu_count()
    t0 = time.time()
    process_split(args.split, nproc)
    print(f"\nTotal elapsed: {(time.time()-t0)/3600:.2f} h")


if __name__ == "__main__":
    main()
