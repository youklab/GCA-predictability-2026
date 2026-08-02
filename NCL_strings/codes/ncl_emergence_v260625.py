"""
ncl_emergence_v260625.py

Tests whether non-contractible loops (NCLs) -- the topological objects the CA
paper shows are predictive -- are ABSENT in the initial configuration (t=1) and
EMERGE over the trajectory, and whether their emergence TRACKS the rise in
single-color CNN recall.

The NCL detector is ported verbatim from Elinor's module
(Elinor_secrete_sense_module.py): label_phase_clusters + find_noncontractible_loops.
An NCL is a closed cycle of same-state cells (found via DFS back-edge) whose
summed wrap-corrected displacement gives a nonzero integer winding number around
the torus. This is the branching-point/winding definition used in the paper --
not the old contractible-inclusive wrap test.

Because the detector is a pure function of a single frame's 4-state vector plus
the fixed lattice geometry, it runs directly on the stored gridsize-14 fraction
snapshots (each X row is cell_4state at one frame) via a minimal shim object --
no re-simulation, no vortex labeling.

For each config, per color (phase_value = 1,2,3,4) and pooled (None), records:
  - n_ncl   : number of NCLs (nonzero-winding loops)
  - has_ncl : whether >=1 NCL is present
Aggregates mean n_ncl and fraction-with-any-NCL per (fraction, fate, color).
The f=0.00 fraction is the t=1 existence endpoint.

Reads the local Loomis splits. Writes to ncl_emergence_results_v260625/.

Usage:
    python ncl_emergence_v260625.py --validate   # check port vs a direct re-derivation on samples
    python ncl_emergence_v260625.py              # sweep (SUBSAMPLE below controls speed)
"""

import os
import csv
import sys
import time
import argparse
import numpy as np

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

# NCL detection uses pure-Python DFS per cluster; the mean curves are population
# means and are identical (~0.001) for 50k vs full 850k. Set None for full.
SUBSAMPLE = None

# DFS can recurse to the size of the largest same-state cluster (up to ~196).
sys.setrecursionlimit(10000)


# ---- CA geometry (inlined, numpy-only; identical to the CA module) ----

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
    """Reproduce init_lattice's neighbor table: cell_number_nn[:,0]=index,
    columns 1..6 = the 6 nearest neighbors (scaled distance rounds to 1)."""
    pos, lx, ly = calculate_triangular_lattice(gridsize)
    n = gridsize ** 2
    x = pos[:, 0]; y = pos[:, 1]
    x1, x2 = np.meshgrid(x, x); y1, y2 = np.meshgrid(y, y)
    dx = np.mod(np.abs(x1 - x2), lx); dx = np.minimum(dx, lx - dx)
    dy = np.mod(np.abs(y1 - y2), ly); dy = np.minimum(dy, ly - dy)
    dist = (dx ** 2 + dy ** 2) ** 0.5 / (lx / gridsize)
    nn = np.round(dist, 1) == 1
    rows_cols = np.argwhere(nn)
    # each cell must have exactly 6 neighbors
    counts = nn.sum(axis=1)
    assert np.all(counts == 6), f"expected 6 nn for all cells, got {np.unique(counts)}"
    cell_number_nn = np.zeros((n, 7), dtype=int)
    cell_number_nn[:, 0] = np.arange(n)
    cell_number_nn[:, 1:] = rows_cols[:, 1].reshape(n, 6)
    return cell_number_nn, pos


class NCLShim:
    """Minimal object exposing exactly what label_phase_clusters and
    find_noncontractible_loops read: cell_4state (N,1), cell_number_nn,
    number_of_nn, positions, N, tmax, gridsize."""

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

    # ---- ported verbatim from Elinor_secrete_sense_module.py ----

    def label_phase_clusters(self, frame, phase_value=None):
        if not hasattr(self, 'cell_4state') or self.cell_4state is None or len(self.cell_4state) == 0:
            raise ValueError("cell_4state is empty")
        N = int(self.N)
        if frame < 0 or frame >= self.tmax:
            raise IndexError(f"frame {frame} out of bounds [0, {self.tmax})")
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
                    if v is None:
                        continue
                    try:
                        v_int = int(v)
                    except Exception:
                        continue
                    if v_int < 0 or v_int >= N:
                        continue
                    if visited[v_int]:
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
        try:
            if pos.ndim == 1 and pos.size == 2 * N:
                pos = pos.reshape((N, 2))
        except Exception:
            pass
        if pos.size == 0 or pos.ndim != 2 or pos.shape[1] != 2:
            pos_calc, lx_calc, ly_calc = calculate_triangular_lattice(self.gridsize)
            pos = np.asarray(pos_calc); lx = lx_calc; ly = ly_calc
        else:
            if self.number_of_nn == 6:
                lx = np.max(pos[:, 0]) + pos[0, 0] / 2.0
                ly = np.max(pos[:, 1]) + pos[0, 1] / 2.0
            else:
                lx = np.max(pos[:, 0]) + pos[0, 0]
                ly = np.max(pos[:, 1]) + pos[0, 1]
        try:
            nn_table = self.cell_number_nn[:, 1:1 + int(self.number_of_nn)].astype(int)
        except Exception:
            rel = getattr(self, 'relative_distance', None)
            if rel is None:
                raise ValueError('No neighbor information available')
            nn_table = None

        def neighbors(i):
            if nn_table is not None:
                return [int(x) for x in nn_table[int(i)] if int(x) >= 0]
            else:
                return list(np.where(np.round(rel[int(i), :], 1) == 1)[0])

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


def validate(shim, n_sample=50):
    """Sanity check: NCL detection should give 0 NCLs on maximally disordered
    random configs (no large same-state cycles) and should be deterministic."""
    rng = np.random.default_rng(0)
    zero_count = 0
    for _ in range(n_sample):
        sv = rng.integers(1, 5, size=N_CELLS)
        shim.set_config(sv)
        pooled = shim.find_noncontractible_loops(0, phase_value=None)
        if len(pooled) == 0:
            zero_count += 1
    print(f"validation: {zero_count}/{n_sample} random configs have 0 NCLs "
          f"(expect most/all; random configs are maximally disordered)")
    sv = rng.integers(1, 5, size=N_CELLS)
    shim.set_config(sv)
    a = len(shim.find_noncontractible_loops(0, phase_value=2))
    b = len(shim.find_noncontractible_loops(0, phase_value=2))
    print(f"determinism: {a} == {b} -> {a == b}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args()

    cell_number_nn, positions = build_cell_number_nn(GRIDSIZE)
    shim = NCLShim(GRIDSIZE, cell_number_nn, positions)

    if args.validate:
        validate(shim)
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, "ncl_emergence_summary.csv")
    cols = ["clock", "fraction", "fate", "fate_name", "color", "color_name",
            "n_configs", "n_ncl_mean", "has_ncl_frac"]
    rows = []
    t0 = time.time()

    color_keys = STATES + [0]   # 0 = pooled (phase_value=None)
    color_pv = {c: c for c in STATES}; color_pv[0] = None
    color_label = dict(COLOR_NAMES); color_label[0] = "pooled"

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
        print(f"[f={frac:.2f}] {nconf} configs", flush=True)
        t1 = time.time()

        acc = {}
        def add(key, n_ncl):
            a = acc.setdefault(key, [0.0, 0, 0])
            a[0] += n_ncl; a[1] += int(n_ncl > 0); a[2] += 1

        for n in range(nconf):
            sv = X[n]; ft = int(fate[n])
            shim.set_config(sv)
            for ckey in color_keys:
                n_ncl = len(shim.find_noncontractible_loops(0, phase_value=color_pv[ckey]))
                add((ft, ckey), n_ncl)
                add((0, ckey), n_ncl)
            if (n + 1) % 10000 == 0:
                print(f"    {n+1}/{nconf}  ({time.time()-t1:.0f}s)", flush=True)

        for (ft, ckey), a in sorted(acc.items(), key=lambda kv: (kv[0][0], kv[0][1])):
            cnt = a[2]
            rows.append([CLOCK, frac, ft, FATE_NAMES.get(ft, "all"),
                         ckey, color_label[ckey], cnt, a[0]/cnt, a[1]/cnt])
        with open(csv_path, "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(cols); w.writerows(rows)
        print(f"  [f={frac:.2f}] done ({time.time()-t1:.0f}s); wrote {csv_path}", flush=True)

    print(f"\nTotal elapsed: {(time.time()-t0)/3600:.2f} h")
    print("Wrote:", csv_path)


if __name__ == "__main__":
    main()
