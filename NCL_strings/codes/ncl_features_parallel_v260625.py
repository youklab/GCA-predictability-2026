"""
ncl_features_parallel_v260625.py

Same parallel NCL detector as ncl_emergence_parallel_v260625.py, but instead of
only aggregating means it saves the PER-CONFIG NCL count vector for each config,
so we can train an "NCL-count-only" fate predictor (the analog of the density-
only predictor).

For each fraction, writes ncl_features_v260625/ncl_features_f{frac:.2f}.npz with:
  counts : (N, 5) int   columns = [pooled, color1, color2, color3, color4] NCL counts
  fate   : (N,) int      final pattern type (1 static, 2 spiral, 3 rectilinear)

Detector ported verbatim from Elinor_secrete_sense_module.py. Reads the local
train_pool splits. Full 850k per fraction.

Usage:
    python ncl_features_parallel_v260625.py
    python ncl_features_parallel_v260625.py --nproc 24
"""

import os
import sys
import time
import argparse
import numpy as np
import multiprocessing as mp

LOCAL_ROOT = "<PATH_TO_OUTPUT>"
SPLITS_ROOT = os.path.join(LOCAL_ROOT, "ml_splits_v260529")
OUT_DIR = os.path.join(LOCAL_ROOT, "ncl_features_v260625")

CLOCK = "Ttrue"
FRACTIONS = [0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
GRIDSIZE = 14
N_CELLS = GRIDSIZE * GRIDSIZE
STATES = [1, 2, 3, 4]
COLOR_KEYS = [0] + STATES               # column 0 = pooled, then colors 1..4
COLOR_PV = {0: None, 1: 1, 2: 2, 3: 3, 4: 4}

sys.setrecursionlimit(10000)


def calculate_triangular_lattice(gridsize):
    lx = 1
    delx = lx / gridsize
    dely = np.sqrt(3) / 2 * delx
    ly = dely * gridsize
    x = np.linspace(0, gridsize - 1, gridsize)
    xm, ym = np.meshgrid(x, x)
    x = (xm + np.mod(ym, 2) / 2) * delx
    y = ym * dely
    pos = np.column_stack((x.flatten('F'), y.flatten('F')))
    return pos, lx, ly


def build_cell_number_nn(gridsize):
    pos, lx, ly = calculate_triangular_lattice(gridsize)
    n = gridsize ** 2
    x = pos[:, 0]; y = pos[:, 1]
    x1, x2 = np.meshgrid(x, x); y1, y2 = np.meshgrid(y, y)
    dx = np.mod(np.abs(x1 - x2), lx); dx = np.minimum(dx, lx - dx)
    dy = np.mod(np.abs(y1 - y2), ly); dy = np.minimum(dy, ly - dy)
    dist = (dx ** 2 + dy ** 2) ** 0.5 / (lx / gridsize)
    nn = np.round(dist, 1) == 1
    rows_cols = np.argwhere(nn)
    counts = nn.sum(axis=1)
    assert np.all(counts == 6), f"expected 6 nn, got {np.unique(counts)}"
    cell_number_nn = np.zeros((n, 7), dtype=int)
    cell_number_nn[:, 0] = np.arange(n)
    cell_number_nn[:, 1:] = rows_cols[:, 1].reshape(n, 6)
    return cell_number_nn, pos


class NCLShim:
    def __init__(self, gridsize, cell_number_nn, positions):
        self.gridsize = gridsize
        self.N = gridsize ** 2
        self.number_of_nn = 6
        self.cell_number_nn = cell_number_nn
        self.positions = positions
        self.tmax = 1
        self.cell_4state = np.zeros((self.N, 1), dtype=int)

    def set_config(self, state_vec):
        self.cell_4state[:, 0] = state_vec.astype(int)

    def label_phase_clusters(self, frame, phase_value=None):
        N = int(self.N)
        values = self.cell_4state[:, frame].astype(int)
        nn_cols = self.cell_number_nn[:, 1:1 + int(self.number_of_nn)].astype(int)
        labels = np.zeros(N, dtype=int)
        visited = np.zeros(N, dtype=bool)
        next_label = 1
        for i in range(N):
            if visited[i]:
                continue
            if (phase_value is not None) and (not np.all(values[i] == phase_value)):
                visited[i] = True
                continue
            stack = [i]
            visited[i] = True
            labels[i] = next_label
            while stack:
                u = stack.pop()
                for v in nn_cols[u]:
                    v_int = int(v)
                    if v_int < 0 or v_int >= N or visited[v_int]:
                        continue
                    if np.all(values[v_int] == values[u]):
                        visited[v_int] = True
                        labels[v_int] = next_label
                        stack.append(v_int)
                    else:
                        if phase_value is not None:
                            visited[v_int] = True
            next_label += 1
        return labels

    def find_noncontractible_loops(self, frame, phase_value=None, return_zero_winding=False):
        N = int(self.N)
        cluster_labels = self.label_phase_clusters(frame, phase_value=phase_value)
        pos = np.asarray(self.positions)
        if self.number_of_nn == 6:
            lx = np.max(pos[:, 0]) + pos[0, 0] / 2.0
            ly = np.max(pos[:, 1]) + pos[0, 1] / 2.0
        else:
            lx = np.max(pos[:, 0]) + pos[0, 0]
            ly = np.max(pos[:, 1]) + pos[0, 1]
        nn_table = self.cell_number_nn[:, 1:1 + int(self.number_of_nn)].astype(int)

        def neighbors(i):
            return [int(x) for x in nn_table[int(i)] if int(x) >= 0]

        results = []
        all_cycles = []

        def canonical_cycle(cycle):
            L = len(cycle)
            if L == 0:
                return ()
            seqs = []
            for k in range(L):
                seqs.append(tuple(cycle[k:] + cycle[:k]))
            rev = list(reversed(cycle))
            for k in range(L):
                seqs.append(tuple(rev[k:] + rev[:k]))
            return min(seqs)

        seen_cycles = set()
        for cluster_id in np.unique(cluster_labels):
            if cluster_id == 0:
                continue
            comp = np.where(cluster_labels == cluster_id)[0].tolist()
            if len(comp) < 3:
                continue
            comp_set = set(comp)
            adj = {c: [nb for nb in neighbors(c) if nb in comp_set] for c in comp}
            parent = {}; depth = {}; visited = set(); time_counter = 0

            def dfs(u):
                nonlocal time_counter
                visited.add(u); time_counter += 1; depth[u] = time_counter
                for v in adj.get(u, []):
                    if v not in visited:
                        parent[v] = u; dfs(v)
                    else:
                        if depth.get(v, 0) < depth.get(u, 0):
                            p = [u]; w = u
                            while w != v and w in parent:
                                w = parent[w]; p.append(w)
                            p.reverse()
                            can = canonical_cycle(p)
                            if can in seen_cycles:
                                continue
                            seen_cycles.add(can)
                            cycle_nodes = list(p); L = len(cycle_nodes)
                            total_dx = 0.0; total_dy = 0.0
                            for i in range(L):
                                a = cycle_nodes[i]; b = cycle_nodes[(i + 1) % L]
                                x1, y1 = pos[int(a)]; x2, y2 = pos[int(b)]
                                dx = float(x2) - float(x1); dy = float(y2) - float(y1)
                                if dx > lx / 2.0: dx -= lx
                                elif dx < -lx / 2.0: dx += lx
                                if dy > ly / 2.0: dy -= ly
                                elif dy < -ly / 2.0: dy += ly
                                total_dx += dx; total_dy += dy
                            wx = int(round(total_dx / lx)) if lx != 0 else 0
                            wy = int(round(total_dy / ly)) if ly != 0 else 0
                            rec = {'nodes': cycle_nodes, 'windings': (wx, wy), 'length': len(cycle_nodes)}
                            all_cycles.append(rec)
                            if wx != 0 or wy != 0:
                                rec['wrap_count'] = max(abs(wx), abs(wy)); results.append(rec)

            for node in comp:
                if node not in visited:
                    parent[node] = None; dfs(node)

        if return_zero_winding:
            return all_cycles, results
        return results


def load_fate(meta_path):
    with np.load(meta_path) as z:
        return np.asarray(z["final_pattern_type"]).astype(np.int64)


_G = {}


def _init_worker(gridsize, cell_number_nn, positions):
    _G["shim"] = NCLShim(gridsize, cell_number_nn, positions)


def _process_range(rng):
    start, end = rng
    shim = _G["shim"]; X = _G["X"]
    out = np.zeros((end - start, 5), dtype=np.int32)
    for r, n in enumerate(range(start, end)):
        shim.set_config(X[n])
        for ci, ckey in enumerate(COLOR_KEYS):
            out[r, ci] = len(shim.find_noncontractible_loops(0, phase_value=COLOR_PV[ckey]))
    return start, end, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nproc", type=int, default=0)
    args = ap.parse_args()
    nproc = args.nproc if args.nproc > 0 else mp.cpu_count()
    ctx = mp.get_context("fork")
    cell_number_nn, positions = build_cell_number_nn(GRIDSIZE)
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Using {nproc} workers", flush=True)
    t0 = time.time()

    for frac in FRACTIONS:
        sd = os.path.join(SPLITS_ROOT, f"ml_splits_v260529_{CLOCK}_f{frac:.2f}")
        if not os.path.isdir(sd):
            print(f"[f={frac:.2f}] MISSING {sd}", flush=True); continue
        X = np.load(os.path.join(sd, "X_train_pool.npy"))
        fate = load_fate(os.path.join(sd, "meta_train_pool.npz"))
        nconf = X.shape[0]
        print(f"[f={frac:.2f}] {nconf} configs", flush=True)
        t1 = time.time()
        _G["X"] = X
        counts = np.zeros((nconf, 5), dtype=np.int32)
        nchunks = nproc * 4
        bounds = np.linspace(0, nconf, nchunks + 1).astype(int)
        ranges = [(int(bounds[i]), int(bounds[i + 1])) for i in range(nchunks) if bounds[i + 1] > bounds[i]]
        with ctx.Pool(processes=nproc, initializer=_init_worker,
                      initargs=(GRIDSIZE, cell_number_nn, positions)) as pool:
            for start, end, out in pool.imap_unordered(_process_range, ranges):
                counts[start:end] = out
        outpath = os.path.join(OUT_DIR, f"ncl_features_f{frac:.2f}.npz")
        np.savez_compressed(outpath, counts=counts, fate=fate)
        print(f"  [f={frac:.2f}] done ({time.time()-t1:.0f}s); wrote {outpath}", flush=True)

    print(f"\nTotal elapsed: {(time.time()-t0)/3600:.2f} h")


if __name__ == "__main__":
    main()
