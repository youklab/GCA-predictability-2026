"""
winding_order_full_v260625.py
Full forward-time order-parameter computation over the trajectory chunks.

Per trajectory (to Ttrue), sampled densely early + sparsely late (B-sampling:
every frame for t in [0, T_DENSE], then every STRIDE-th frame to Ttrue, always
including t=0 and t=Ttrue), compute six scalar series:
  wnz  : fraction of cells with nonzero (wx, wy)   [winding field "switched on"]
  I1w  : Moran's I of wx field                      [winding spatial order, x]
  I2w  : Moran's I of wy field                      [winding spatial order, y]
  I1c  : Moran's I of gene1 binary field            [color spatial order, gene1 = states 3,4]
  I2c  : Moran's I of gene2 binary field            [color spatial order, gene2 = states 2,4]
  nv   : total vortex core-blob count (lumps 0/+1/-1)

Stored ragged per chunk, keyed by global_index, with each trajectory's sampled
t-values and Ttrue, so f = t/Ttrue resampling happens at plot time (raw primitive
preserved). NO encoding swap (chunks are native encoding). Moran's I uses the
module's normalized relative_distance with exp(-dist)/dist weights (matches
calculate_moranI); winding fields mean-centered raw (no binary remap), color
fields use binary {0,1}->{-1,+1}.

Parallel over trajectories within each chunk (fork pool). Incremental: writes one
output per chunk; skips chunks already done.

Usage:
    python winding_order_full_v260625.py --validate            # one chunk, timing + sanity
    python winding_order_full_v260625.py --chunks 0 1000        # all chunks [start,end)
    python winding_order_full_v260625.py --chunks 0 1000 --nproc 24
"""

import os
import time
import argparse
import importlib.util
import numpy as np
import multiprocessing as mp

MODULE_PATH = "Elinor_secrete_sense_module.py"
CHUNK_DIR = "+v260529_full_trajectories"
OUT_DIR = "<PATH_TO_OUTPUT>/winding_order_time_v260625"
GRIDSIZE = 14; NUM_CELLS = 196; TMAX = 75000
RCELL = 0.2; A0 = 1.5; LATTICE_TYPE = "triangular"; PERIODIC_BC = [1, 1]
M_MATRIX = np.array([[1, 1], [-1, 0]]); K_MATRIX = np.array([[3, 10], [11, 4]])
C_MATRIX = np.array([18, 16]); LAMB = np.array([1, 1.2])

T_DENSE = 100      # every frame up to here
STRIDE = 10        # every STRIDE-th frame after

_G = {}


def load_module():
    spec = importlib.util.spec_from_file_location("esm", MODULE_PATH)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def make_lattice(esm):
    t = esm.CellularLattice(GRIDSIZE, NUM_CELLS, RCELL, A0, TMAX)
    t.init_lattice(GRIDSIZE, PERIODIC_BC, A0, RCELL, LATTICE_TYPE)
    t.init_general_parameters(K_MATRIX, C_MATRIX, M_MATRIX, LAMB)
    t.quiet_warnings = True
    return t


def set_frame(t, snap):
    cs = np.repeat(np.asarray(snap, int).reshape(NUM_CELLS, 1), 2, axis=1)
    t.cell_4state = cs; t.tmax = 2
    ph = np.zeros_like(cs, dtype=float)
    ph[cs == 1] = 0.0; ph[cs == 2] = 0.5 * np.pi; ph[cs == 4] = np.pi; ph[cs == 3] = 1.5 * np.pi
    t.cell_4phase = ph


def sample_times(Ttrue):
    dense = np.arange(0, min(T_DENSE, Ttrue) + 1)
    if Ttrue > T_DENSE:
        sparse = np.arange(T_DENSE + STRIDE, Ttrue + 1, STRIDE)
        ts = np.concatenate([dense, sparse])
        if ts[-1] != Ttrue:
            ts = np.concatenate([ts, [Ttrue]])
    else:
        ts = dense
    return ts.astype(int)


def moran(values, Mw, w_sum, binary_pm=False):
    v = np.asarray(values, float)
    if binary_pm:
        v = 2 * v - 1
    var = np.var(v)
    if var == 0:
        return 0.0
    vm = v - v.mean()
    return (Mw * np.outer(vm, vm)).sum() / w_sum / var


def winding_field(t, esm):
    wx = np.zeros(NUM_CELLS); wy = np.zeros(NUM_CELLS)
    try:
        labels = t.label_phase_clusters(0, phase_value=None)
        loops = t.find_noncontractible_loops(0, phase_value=None)
        cw = {}
        for lp in loops:
            a, b = lp.get("windings", (0, 0))
            for n in lp.get("nodes", []):
                n = int(n)
                if 0 <= n < NUM_CELLS and labels[n] > 0:
                    cid = int(labels[n]); cur = cw.get(cid, [0, 0])
                    if abs(a) > abs(cur[0]): cur[0] = int(a)
                    if abs(b) > abs(cur[1]): cur[1] = int(b)
                    cw[cid] = cur
        for cid, (a, b) in cw.items():
            sel = (labels == cid); wx[sel] = a; wy[sel] = b
    except Exception:
        pass
    return wx, wy


def nvort(t, esm):
    try:
        t.vortex_cores, t.Hxy, t.Hxy_adjusted = esm.compute_phase_differences(t)
        t.vortex_cores_labeled, t.n_vortex = esm.label_triangular_lattice(t)
        vcl = np.asarray(t.vortex_cores_labeled); lab0 = vcl[:, 0] if vcl.ndim == 2 else vcl
        return int(lab0.max()) if lab0.size else 0
    except Exception:
        return 0


def _init_worker():
    esm = load_module()
    tL = make_lattice(esm)
    dist = np.asarray(tL.relative_distance); idx = dist > 0
    Mw = np.zeros_like(dist); Mw[idx] = np.exp(-dist[idx]) / dist[idx]
    _G["esm"] = esm; _G["tL"] = tL; _G["Mw"] = Mw; _G["w_sum"] = Mw.sum()


def _process_traj(arg):
    row, Ttrue, traj_slice = arg
    esm = _G["esm"]; tL = _G["tL"]; Mw = _G["Mw"]; w_sum = _G["w_sum"]
    ts = sample_times(int(Ttrue))
    out = {k: np.zeros(len(ts), dtype=np.float32) for k in ["wnz", "I1w", "I2w", "I1c", "I2c", "nv"]}
    for j, fr in enumerate(ts):
        snap = np.asarray(traj_slice[j], int)
        set_frame(tL, snap)
        wx, wy = winding_field(tL, esm)
        out["wnz"][j] = np.mean((wx != 0) | (wy != 0))
        out["I1w"][j] = moran(wx, Mw, w_sum)
        out["I2w"][j] = moran(wy, Mw, w_sum)
        out["I1c"][j] = moran((snap >= 3).astype(float), Mw, w_sum, binary_pm=True)
        out["I2c"][j] = moran(((snap == 2) | (snap == 4)).astype(float), Mw, w_sum, binary_pm=True)
        out["nv"][j] = nvort(tL, esm)
    return row, int(Ttrue), ts.astype(np.int32), out


def process_chunk(cidx, nproc, validate=False):
    path = os.path.join(CHUNK_DIR, f"v260529_chunk_{cidx:04d}.npz")
    if not os.path.isfile(path):
        return None
    z = np.load(path, allow_pickle=True)
    data = z["data"]; fate = np.asarray(z["final_pattern_type"]).astype(int)
    Ttrue = np.asarray(z["true_formation_time"]).astype(int)
    nframes = data.shape[1]              # stored frames per trajectory (e.g. 30000); valid indices 0..nframes-1
    nrows = data.shape[0] if not validate else 60
    # pre-slice the sampled frames per traj (so workers don't hold full 30000-frame arrays)
    args = []
    for row in range(nrows):
        Tt = int(Ttrue[row]); ts = sample_times(Tt)
        ts = np.minimum(ts, nframes - 1)   # clamp: Ttrue can equal nframes (last stored frame is the formed state)
        args.append((row, Tt, np.asarray(data[row][ts], dtype=np.int8)))
    ctx = mp.get_context("fork")
    results = {}
    t0 = time.time()
    with ctx.Pool(processes=nproc, initializer=_init_worker) as pool:
        for row, Tt, ts, out in pool.imap_unordered(_process_traj, args):
            results[row] = (Tt, ts, out, int(fate[row]))
    if validate:
        # quick sanity per fate
        for ft in (1, 2, 3):
            rows = [r for r in results if results[r][3] == ft][:20]
            wnz0 = np.mean([results[r][2]["wnz"][0] for r in rows])
            # value at absolute t=10 (index 10 since dense early)
            wnz10 = np.mean([results[r][2]["wnz"][10] for r in rows if len(results[r][2]["wnz"]) > 10])
            nv5 = np.mean([results[r][2]["nv"][5] for r in rows if len(results[r][2]["nv"]) > 5])
            nm = {1: "static", 2: "spiral", 3: "rect"}[ft]
            print(f"  {nm}: wnz(t=0)={wnz0:.3f} wnz(t=10)={wnz10:.3f} nv(t=5)={nv5:.1f}  (n={len(rows)})")
        print(f"  chunk {cidx} validate: {nrows} traj in {time.time()-t0:.0f}s "
              f"(~{(time.time()-t0)/nrows*1000*1000/3600:.1f}h projected for 1e6)")
        return None
    # store ragged
    os.makedirs(OUT_DIR, exist_ok=True)
    save = {}
    for row, (Tt, ts, out, ft) in results.items():
        g = cidx * 1000 + row
        save[f"t_{g}"] = ts
        save[f"Ttrue_{g}"] = np.int32(Tt)
        save[f"fate_{g}"] = np.int8(ft)
        for k in ["wnz", "I1w", "I2w", "I1c", "I2c", "nv"]:
            save[f"{k}_{g}"] = out[k]
    outp = os.path.join(OUT_DIR, f"order_chunk_{cidx:04d}.npz")
    np.savez_compressed(outp, **save)
    return time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--chunks", type=int, nargs=2, default=[0, 1000])
    ap.add_argument("--nproc", type=int, default=0)
    args = ap.parse_args()
    nproc = args.nproc if args.nproc > 0 else mp.cpu_count()
    if args.validate:
        print("VALIDATION (chunk 0, 60 trajectories):")
        process_chunk(0, nproc, validate=True)
        return
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time(); done = 0
    for cidx in range(args.chunks[0], args.chunks[1]):
        outp = os.path.join(OUT_DIR, f"order_chunk_{cidx:04d}.npz")
        if os.path.isfile(outp):
            continue
        el = process_chunk(cidx, nproc)
        if el is None:
            continue
        done += 1
        tot = time.time() - t0
        print(f"chunk {cidx}: {el:.0f}s  | {done} done, {tot/3600:.2f}h elapsed, "
              f"~{tot/done*(args.chunks[1]-args.chunks[0]-done)/3600:.1f}h remaining", flush=True)
    print(f"\nDone. {done} chunks in {(time.time()-t0)/3600:.2f}h")


if __name__ == "__main__":
    main()
