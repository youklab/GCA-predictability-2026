"""
winding_field_extract_v260625.py  [STAGE 1 of the per-cell winding-field experiment]

For each split config, extract two per-cell topological fields, both aligned to
the split X (row-for-row), saving in one pass:

  (A) WINDING field (2 channels): per-cell (wx, wy) = the componentwise max-abs
      signed winding of the same-color CLUSTER the cell belongs to. Every cell in
      a cluster inherits that cluster's winding class; cells in non-wrapping
      clusters get (0,0). Computed via label_phase_clusters (per-cell cluster id)
      + find_noncontractible_loops (per-loop windings), aggregated per cluster by
      componentwise max-abs over its loops' windings. Kind-2 (global), dense
      (~half the lattice nonzero at mid/late f). No vortices invoked.

  (B) VORTEX-core field (2 channels): signed core field as in the v260529
      reexpression. Channel 0 marks +1 core cells, channel 1 marks -1 core cells,
      charge painted onto all cells of each labeled core blob. Computed via
      compute_phase_differences -> label_triangular_lattice -> compute_smallest_movement
      (per-core charges; same winding-of-contour quantity).

Single-snapshot pipeline: duplicate the snapshot to (196,2) for shape, tmax=2,
read frame 0. NO 3<->4 encoding swap (splits are chunk-encoded). Must run from
the Box HY_CA folder (imports Elinor_secrete_sense_module by path).

Outputs (Loomis string-style) -> winding_fields_v260625/fields_Ttrue_f{f}{,_val,_test}.npz
  winding : (N, 2, 196) int8   [channel 0 = wx, channel 1 = wy]
  vortex  : (N, 2, 196) int8   [channel 0 = +1 cores, channel 1 = -1 cores]

Usage:
    python winding_field_extract_v260625.py --validate
    python winding_field_extract_v260625.py --split train
    python winding_field_extract_v260625.py --split val
    python winding_field_extract_v260625.py --split test
    python winding_field_extract_v260625.py --split train --nproc 24
"""

import os
import time
import argparse
import importlib.util
import numpy as np
import multiprocessing as mp

MODULE_PATH = "Elinor_secrete_sense_module.py"
SPLITS_ROOT = "<PATH_TO_OUTPUT>/ml_splits_v260529"
OUT_DIR = "<PATH_TO_OUTPUT>/winding_fields_v260625"
CLOCK = "Ttrue"
FRACTIONS = [0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
GRIDSIZE = 14
NUM_CELLS = GRIDSIZE ** 2

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
    t = esm.CellularLattice(GRIDSIZE, NUM_CELLS, RCELL, A0, TMAX)
    t.init_lattice(GRIDSIZE, PERIODIC_BC, A0, RCELL, LATTICE_TYPE)
    t.init_general_parameters(K_MATRIX, C_MATRIX, M_MATRIX, LAMB)
    t.quiet_warnings = True
    snap = np.asarray(snapshot_196, dtype=int)
    cs = np.repeat(snap.reshape(NUM_CELLS, 1), 2, axis=1)   # (196,2) shape fix
    t.cell_4state = cs
    t.tmax = 2
    phase = np.zeros_like(cs, dtype=float)
    phase[cs == 1] = 0.0
    phase[cs == 2] = 0.5 * np.pi
    phase[cs == 4] = 1.0 * np.pi
    phase[cs == 3] = 1.5 * np.pi
    t.cell_4phase = phase
    return t


def extract_fields(esm, snapshot_196):
    """Return (winding (2,196) int8, vortex (2,196) int8) for one snapshot."""
    winding = np.zeros((2, NUM_CELLS), dtype=np.int8)
    vortex = np.zeros((2, NUM_CELLS), dtype=np.int8)
    try:
        t = build_object(esm, snapshot_196)
    except Exception:
        return winding, vortex

    # (A) winding field: per-cell cluster id, per-loop windings -> per-cluster max-abs
    try:
        labels = t.label_phase_clusters(0, phase_value=None)        # (196,) cluster id (0=unlabeled)
        loops = t.find_noncontractible_loops(0, phase_value=None)   # list of {nodes, windings}
        cluster_w = {}   # cid -> [wx, wy] componentwise max-abs
        for lp in loops:
            wx, wy = lp.get("windings", (0, 0))
            cids = set(int(labels[n]) for n in lp.get("nodes", []) if 0 <= int(n) < NUM_CELLS and labels[int(n)] > 0)
            for cid in cids:
                cur = cluster_w.get(cid, [0, 0])
                if abs(wx) > abs(cur[0]):
                    cur[0] = int(wx)
                if abs(wy) > abs(cur[1]):
                    cur[1] = int(wy)
                cluster_w[cid] = cur
        for cid, (wx, wy) in cluster_w.items():
            sel = (labels == cid)
            winding[0, sel] = wx
            winding[1, sel] = wy
    except Exception:
        pass

    # (B) vortex-core field: cores labeled + per-core charge, painted onto core blobs
    try:
        t.vortex_cores, t.Hxy, t.Hxy_adjusted = esm.compute_phase_differences(t)
        t.vortex_cores_labeled, t.n_vortex = esm.label_triangular_lattice(t)
        t.xcom, t.ycom = esm.compute_centroid(t)
        t.closest_pos, t.charges, t.xmin_core, t.ymin_core = esm.compute_smallest_movement(t)
        vcl = np.asarray(t.vortex_cores_labeled)          # (196,2) labeled cores, frame 0 in col 0
        ch = np.asarray(t.charges)                        # (2, 2+nlabels): row0 frame0; cols 2.. per label
        lab0 = vcl[:, 0] if vcl.ndim == 2 else vcl
        nlab = int(lab0.max()) if lab0.size else 0
        for lab in range(1, nlab + 1):
            # charge for this label at frame 0
            q = 0
            try:
                q = int(round(float(ch[0, 1 + lab])))     # cols: 0,1 are coords; charges at 2.. -> label l at col 1+l
            except Exception:
                q = 0
            if q == 0:
                continue
            cells = np.where(lab0 == lab)[0]
            if q > 0:
                vortex[0, cells] = 1
            elif q < 0:
                vortex[1, cells] = 1
    except Exception:
        pass

    return winding, vortex


def _init_worker():
    _G["esm"] = load_module()


def _process_range(args):
    start, end = args
    X = _G["X"]; esm = _G["esm"]
    w_out = np.zeros((end - start, 2, NUM_CELLS), dtype=np.int8)
    v_out = np.zeros((end - start, 2, NUM_CELLS), dtype=np.int8)
    for r, i in enumerate(range(start, end)):
        w, v = extract_fields(esm, X[i])
        w_out[r] = w; v_out[r] = v
    return start, end, w_out, v_out


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
        w_all = np.zeros((n, 2, NUM_CELLS), dtype=np.int8)
        v_all = np.zeros((n, 2, NUM_CELLS), dtype=np.int8)
        nchunks = nproc * 4
        bounds = np.linspace(0, n, nchunks + 1).astype(int)
        ranges = [(int(bounds[i]), int(bounds[i + 1])) for i in range(nchunks) if bounds[i + 1] > bounds[i]]
        t1 = time.time(); done = 0
        with ctx.Pool(processes=nproc, initializer=_init_worker) as pool:
            for start, end, w_out, v_out in pool.imap_unordered(_process_range, ranges):
                w_all[start:end] = w_out; v_all[start:end] = v_out
                done += (end - start)
                if done % 50000 < (end - start):
                    el = time.time() - t1
                    print(f"    {done}/{n} ({el:.0f}s, ~{el/done*n/3600:.1f}h this f)", flush=True)
        outp = os.path.join(OUT_DIR, f"fields_{CLOCK}_f{frac:.2f}{suffix}.npz")
        np.savez_compressed(outp, winding=w_all, vortex=v_all)
        print(f"  [f={frac:.2f}] wrote {outp} ({time.time()-t1:.0f}s)", flush=True)


def validate():
    esm = load_module()
    print("VALIDATION: winding + vortex fields on a small batch\n")
    for frac in [0.00, 0.50, 0.90]:
        sd = os.path.join(SPLITS_ROOT, f"ml_splits_v260529_{CLOCK}_f{frac:.2f}")
        xpath = os.path.join(sd, "X_train_pool.npy")
        if not os.path.isfile(xpath):
            print(f"[f={frac:.2f}] MISSING {xpath}"); continue
        X = np.array(np.load(xpath, mmap_mode="r")[:300])
        nb = X.shape[0]
        wz = []; vz = []; netq = []
        for i in range(nb):
            w, v = extract_fields(esm, X[i])
            wz.append(np.mean(np.any(w != 0, axis=0)))         # frac cells with nonzero (wx,wy)
            vz.append(int(np.sum(v[0]) + np.sum(v[1])))        # total core cells flagged
            netq.append(int(np.sum(v[0]) - np.sum(v[1])))      # +cores minus -cores (cells)
        print(f"[f={frac:.2f}] n={nb}")
        print(f"   winding: mean fraction of cells with nonzero (wx,wy) = {np.mean(wz):.3f}")
        print(f"   vortex:  mean core cells/config = {np.mean(vz):.1f}  "
              f"(mean +cells minus -cells = {np.mean(netq):.2f})")
        if frac == 0.00:
            print(f"   ~empty winding at f=0: {'PASS' if np.mean(wz) < 0.02 else 'note: nonzero at f=0'}")
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
