"""
ncl_emergence_parallel_v260625.py

Parallel version of ncl_emergence_v260625.py. Identical NCL detector (ported
verbatim from Elinor_secrete_sense_module.py), identical output, but the per-
config loop is distributed across CPU cores with multiprocessing.

Each worker builds its own NCLShim once and processes a contiguous index range,
returning a small partial accumulator (per fate, per color). Uses a 'fork'
context so the per-fraction X array is shared copy-on-write, not pickled to
every worker.

Usage:
    python ncl_emergence_parallel_v260625.py --validate
    python ncl_emergence_parallel_v260625.py
    python ncl_emergence_parallel_v260625.py --nproc 24   # cap worker count
"""

import os
import csv
import sys
import time
import argparse
import numpy as np
import multiprocessing as mp

LOCAL_ROOT = "<PATH_TO_OUTPUT>"
SPLITS_ROOT = os.path.join(LOCAL_ROOT, "ml_splits_v260529")
OUT_DIR = os.path.join(LOCAL_ROOT, "ncl_emergence_results_v260625")

CLOCK = "Ttrue"
FRACTIONS = [0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
GRIDSIZE = 14
N_CELLS = GRIDSIZE * GRIDSIZE
STATES = [1, 2, 3, 4]
COLOR_NAMES = {1: "phase0", 2: "red", 3: "phase3pi2", 4: "phasepi"}
FATE_NAMES = {1: "static", 2: "spiral", 3: "rectilinear"}

SUBSAMPLE = None   # None = full 850k per fraction

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
    assert np.all(counts == 6), f"expected 6 nn for all cells, got {np.unique(counts)}"
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
            parent = {}
            depth = {}
            visited = set()
            time_counter = 0

            def dfs(u):
                nonlocal time_counter
                visited.add(u)
                time_counter += 1
                depth[u] = time_counter
                for v in adj.get(u, []):
                    if v not in visited:
                        parent[v] = u
                        dfs(v)
                    else:
                        if depth.get(v, 0) < depth.get(u, 0):
                            p = [u]
                            w = u
                            while w != v and w in parent:
                                w = parent[w]
                                p.append(w)
                            p.reverse()
                            can = canonical_cycle(p)
                            if can in seen_cycles:
                                continue
                            seen_cycles.add(can)
                            cycle_nodes = list(p)
                            L = len(cycle_nodes)
                            total_dx = 0.0
                            total_dy = 0.0
                            for i in range(L):
                                a = cycle_nodes[i]
                                b = cycle_nodes[(i + 1) % L]
                                x1, y1 = pos[int(a)]
                                x2, y2 = pos[int(b)]
                                dx = float(x2) - float(x1)
                                dy = float(y2) - float(y1)
                                if dx > lx / 2.0:
                                    dx -= lx
                                elif dx < -lx / 2.0:
                                    dx += lx
                                if dy > ly / 2.0:
                                    dy -= ly
                                elif dy < -ly / 2.0:
                                    dy += ly
                                total_dx += dx
                                total_dy += dy
                            wx = int(round(total_dx / lx)) if lx != 0 else 0
                            wy = int(round(total_dy / ly)) if ly != 0 else 0
                            cycle_record = {'nodes': cycle_nodes, 'windings': (wx, wy), 'length': len(cycle_nodes)}
                            all_cycles.append(cycle_record)
                            if wx != 0 or wy != 0:
                                cycle_record['wrap_count'] = max(abs(wx), abs(wy))
                                results.append(cycle_record)

            for node in comp:
                if node not in visited:
                    parent[node] = None
                    dfs(node)

        if return_zero_winding:
            return all_cycles, results
        return results


def load_fate(meta_path):
    with np.load(meta_path) as z:
        return np.asarray(z["final_pattern_type"]).astype(np.int64)


# ---- worker globals (set per fraction via fork inheritance) ----
_G = {}
COLOR_KEYS = STATES + [0]
COLOR_PV = {c: c for c in STATES}; COLOR_PV[0] = None


def _init_worker(gridsize, cell_number_nn, positions):
    _G["shim"] = NCLShim(gridsize, cell_number_nn, positions)


def _process_range(rng):
    start, end = rng
    shim = _G["shim"]
    X = _G["X"]; fate = _G["fate"]
    acc = {}  # (fate, color) -> [sum_nncl, sum_has, count]

    def add(key, n_ncl):
        a = acc.get(key)
        if a is None:
            a = [0.0, 0, 0]; acc[key] = a
        a[0] += n_ncl; a[1] += int(n_ncl > 0); a[2] += 1

    for n in range(start, end):
        shim.set_config(X[n])
        ft = int(fate[n])
        for ckey in COLOR_KEYS:
            n_ncl = len(shim.find_noncontractible_loops(0, phase_value=COLOR_PV[ckey]))
            add((ft, ckey), n_ncl)
            add((0, ckey), n_ncl)
    return acc


def validate():
    cell_number_nn, positions = build_cell_number_nn(GRIDSIZE)
    shim = NCLShim(GRIDSIZE, cell_number_nn, positions)
    rng = np.random.default_rng(0)
    zero = 0
    for _ in range(50):
        shim.set_config(rng.integers(1, 5, size=N_CELLS))
        if len(shim.find_noncontractible_loops(0, phase_value=None)) == 0:
            zero += 1
    print(f"validation: {zero}/50 random configs have 0 NCLs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--nproc", type=int, default=0, help="0 = use all cores")
    args = ap.parse_args()
    if args.validate:
        validate()
        return

    nproc = args.nproc if args.nproc > 0 else mp.cpu_count()
    ctx = mp.get_context("fork")
    cell_number_nn, positions = build_cell_number_nn(GRIDSIZE)

    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, "ncl_emergence_summary.csv")
    cols = ["clock", "fraction", "fate", "fate_name", "color", "color_name",
            "n_configs", "n_ncl_mean", "has_ncl_frac"]
    color_label = dict(COLOR_NAMES); color_label[0] = "pooled"
    rows = []
    t0 = time.time()
    print(f"Using {nproc} worker processes", flush=True)

    for frac in FRACTIONS:
        sd = os.path.join(SPLITS_ROOT, f"ml_splits_v260529_{CLOCK}_f{frac:.2f}")
        if not os.path.isdir(sd):
            print(f"[f={frac:.2f}] MISSING {sd}", flush=True)
            continue
        X = np.load(os.path.join(sd, "X_train_pool.npy"))
        fate = load_fate(os.path.join(sd, "meta_train_pool.npz"))
        if SUBSAMPLE is not None and X.shape[0] > SUBSAMPLE:
            idx = np.random.default_rng(0).choice(X.shape[0], SUBSAMPLE, replace=False)
            X = X[idx]; fate = fate[idx]
        nconf = X.shape[0]
        print(f"[f={frac:.2f}] {nconf} configs on {nproc} cores", flush=True)
        t1 = time.time()

        # set fraction data as globals so forked workers inherit them (no pickling)
        _G["X"] = X; _G["fate"] = fate

        # chunk the index range for load balancing
        nchunks = nproc * 4
        bounds = np.linspace(0, nconf, nchunks + 1).astype(int)
        ranges = [(int(bounds[i]), int(bounds[i + 1])) for i in range(nchunks)
                  if bounds[i + 1] > bounds[i]]

        combined = {}
        with ctx.Pool(processes=nproc, initializer=_init_worker,
                      initargs=(GRIDSIZE, cell_number_nn, positions)) as pool:
            for acc in pool.imap_unordered(_process_range, ranges):
                for key, a in acc.items():
                    c = combined.get(key)
                    if c is None:
                        combined[key] = list(a)
                    else:
                        c[0] += a[0]; c[1] += a[1]; c[2] += a[2]

        for (ft, ckey), a in sorted(combined.items(), key=lambda kv: (kv[0][0], kv[0][1])):
            cnt = a[2]
            rows.append([CLOCK, frac, ft, FATE_NAMES.get(ft, "all"),
                         ckey, color_label[ckey], cnt, a[0] / cnt, a[1] / cnt])
        with open(csv_path, "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(cols); w.writerows(rows)
        print(f"  [f={frac:.2f}] done ({time.time() - t1:.0f}s); wrote {csv_path}", flush=True)

    print(f"\nTotal elapsed: {(time.time() - t0) / 3600:.2f} h")
    print("Wrote:", csv_path)


if __name__ == "__main__":
    main()
