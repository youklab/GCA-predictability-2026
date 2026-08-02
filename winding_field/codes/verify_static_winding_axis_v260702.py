"""
verify_static_winding_axis_v260702.py

Permanent verification behind the SI footnote on the static (|wx|,|wy|)=(1,14) value.

Claim being verified: for the fully-filled (all-one-color) static configuration, the
reported large one-axis winding is a SUMMARY-RULE + LATTICE-ASYMMETRY artifact, not a
geometric fact. Specifically:

  (A) Both clean single-wraps EXIST on the triangular lattice:
        - a single ROW of G cells is a closed all-neighbor loop with net x-wrap = 1  -> (1,0)
        - a single COLUMN of G cells is a closed all-neighbor loop with net y-wrap = 1 -> (0,1)
      => the achievable winding group of the filled lattice spans BOTH axes; neither is
         geometrically singled out. (1,14),(14,1),(14,14),(1,1) are all equally valid.

  (B) The module's detector, on the all-red config, returns a LOPSIDED set of fundamental
      cycles: max|wx| small, max|wy| large, with many |wy|>=2 loops and ~no |wx|>=2 loops.
      The per-component max-abs field summary then reports (small, large).

  (C) The lopsidedness is NOT merely column-major node numbering: transposing the node
      ordering + adjacency (column-major -> row-major) does NOT move the large winding to x.
      The cause is the triangular lattice's inequivalent periodic directions: the horizontal
      neighbor step (1,0) is straight (dy=0) while every vertical move is a staggered
      diagonal (dx=+-0.5*cell). A row is a straight chain; a column zig-zags.

Run from the Box HY_CA folder (imports Elinor's module):
    python verify_static_winding_axis_v260702.py

Deterministic; prints PASS/FAIL for (A) and (B) and reports the (C) transpose comparison.
"""
import importlib.util
import numpy as np

MODULE_PATH = "Elinor_secrete_sense_module.py"
G = 14
N = G * G


def load_module():
    spec = importlib.util.spec_from_file_location("esm", MODULE_PATH)
    esm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(esm)
    return esm


def build_lattice(esm, snap):
    t = esm.CellularLattice(G, N, 0.2, 1.5, 75000)
    t.init_lattice(G, [1, 1], 1.5, 0.2, "triangular")
    t.init_general_parameters(np.array([[3, 10], [11, 4]]), np.array([18, 16]),
                              np.array([[1, 1], [-1, 0]]), np.array([1, 1.2]))
    t.quiet_warnings = True
    cs = np.repeat(np.asarray(snap, int).reshape(N, 1), 2, axis=1)
    t.cell_4state = cs
    t.tmax = 2
    ph = np.zeros_like(cs, dtype=float)
    ph[cs == 1] = 0.0
    ph[cs == 2] = 0.5 * np.pi
    ph[cs == 4] = np.pi
    ph[cs == 3] = 1.5 * np.pi
    t.cell_4phase = ph
    return t


def idx(c, r):
    return (c % G) * G + (r % G)   # module convention: index = col*G + row


def cr(i):
    return (i // G, i % G)


def module_loops(esm, snap):
    t = build_lattice(esm, snap)
    loops = t.find_noncontractible_loops(0, phase_value=None)
    w = [tuple(lp.get("windings", (0, 0))) for lp in loops]
    return w


def cluster_field_winding(esm, snap):
    """Reproduce the per-cluster max-abs (signed) field summary used in production."""
    t = build_lattice(esm, snap)
    labels = t.label_phase_clusters(0, phase_value=None)
    loops = t.find_noncontractible_loops(0, phase_value=None)
    cw = {}
    for lp in loops:
        a, b = lp.get("windings", (0, 0))
        for n in lp.get("nodes", []):
            n = int(n)
            if 0 <= n < N and labels[n] > 0:
                cid = int(labels[n])
                cur = cw.get(cid, [0, 0])
                if abs(a) > abs(cur[0]):
                    cur[0] = int(a)
                if abs(b) > abs(cur[1]):
                    cur[1] = int(b)
                cw[cid] = cur
    return cw


def wrap_disp(POS, LX, LY, a, b):
    dx = POS[b, 0] - POS[a, 0]
    dy = POS[b, 1] - POS[a, 1]
    if dx > LX / 2:
        dx -= LX
    elif dx < -LX / 2:
        dx += LX
    if dy > LY / 2:
        dy -= LY
    elif dy < -LY / 2:
        dy += LY
    return dx, dy


def dfs_fundamental_windings(nn, POS, LX, LY):
    """Iterative-DFS fundamental-cycle windings on a given adjacency (nn) and positions."""
    visited = [False] * N
    parent = [-1] * N
    disp = [None] * N
    wind = set()
    for s in range(N):
        if visited[s]:
            continue
        visited[s] = True
        disp[s] = (0.0, 0.0)
        stack = [s]
        while stack:
            u = stack.pop()
            for v in nn[u]:
                v = int(v)
                if not visited[v]:
                    visited[v] = True
                    parent[v] = u
                    dx, dy = wrap_disp(POS, LX, LY, u, v)
                    disp[v] = (disp[u][0] + dx, disp[u][1] + dy)
                    stack.append(v)
                elif parent[u] != v:
                    dx, dy = wrap_disp(POS, LX, LY, u, v)
                    cx = (disp[u][0] - disp[v][0] + dx) / LX
                    cy = (disp[u][1] - disp[v][1] + dy) / LY
                    wind.add((round(cx), round(cy)))
    return wind


def main():
    esm = load_module()
    POS, LX, LY = esm.calculate_triangular_lattice(G)
    t0 = build_lattice(esm, np.full(N, 2, int))
    nn = t0.cell_number_nn[:, 1:7].astype(int)

    print("=" * 70)
    print("(A) Both clean single-wraps exist (row -> (1,0); column -> (0,1))")
    # a single row: cells (c, r0) for all c, at fixed row r0
    r0 = 0
    row_ok = all(idx((c + 1) % G, r0) in nn[idx(c, r0)] for c in range(G))
    netx = sum(wrap_disp(POS, LX, LY, idx(c, r0), idx((c + 1) % G, r0))[0] for c in range(G))
    c0 = 0
    col_ok = all(idx(c0, (r + 1) % G) in nn[idx(c0, r)] for r in range(G))
    nety = sum(wrap_disp(POS, LX, LY, idx(c0, r), idx(c0, (r + 1) % G))[1] for r in range(G))
    print(f"    row closed & all-neighbor: {row_ok}, net x-wrap = {netx / LX:.3f} (expect 1.000)")
    print(f"    col closed & all-neighbor: {col_ok}, net y-wrap = {nety / LY:.3f} (expect 1.000)")
    A_pass = row_ok and col_ok and abs(netx / LX - 1) < 1e-6 and abs(nety / LY - 1) < 1e-6
    print(f"    (A) {'PASS' if A_pass else 'FAIL'}: achievable winding spans BOTH axes; neither singled out.")

    print("=" * 70)
    print("(B) Module detector on all-red is lopsided; field summary = (small, large)")
    snap = np.full(N, 2, int)
    w = module_loops(esm, snap)
    wx = np.array([a for a, b in w]); wy = np.array([b for a, b in w])
    cw = cluster_field_winding(esm, snap)
    field_vals = {k: tuple(v) for k, v in cw.items()}
    print(f"    detected loops: {len(w)}")
    print(f"    max|wx| over loops = {np.abs(wx).max()},  max|wy| over loops = {np.abs(wy).max()}")
    print(f"    #loops |wx|>=2 = {(np.abs(wx) >= 2).sum()},  #loops |wy|>=2 = {(np.abs(wy) >= 2).sum()}")
    print(f"    per-cluster field summary (max-abs signed): {field_vals}")
    B_pass = np.abs(wy).max() > np.abs(wx).max() and (np.abs(wx) >= 2).sum() == 0
    print(f"    (B) {'PASS' if B_pass else 'FAIL'}: large winding sits on one axis via the max-abs summary.")

    print("=" * 70)
    print("(C) Lopsidedness is NOT node-numbering: transpose ordering+adjacency, axis stays")
    # our-own-DFS on module graph (column-major labels)
    w_norm = dfs_fundamental_windings(nn, POS, LX, LY)
    axc = max(abs(a) for a, b in w_norm); ayc = max(abs(b) for a, b in w_norm)
    # transpose node labels: old col-major index i=(c*G+r) -> new row-major j=(r*G+c)
    perm = np.empty(N, int)
    for i in range(N):
        c, r = cr(i)
        perm[i] = r * G + c
    nn_T = np.empty_like(nn)
    POS_T = np.empty_like(POS)
    for i in range(N):
        j = perm[i]
        nn_T[j] = [perm[int(x)] for x in nn[i]]
        POS_T[j] = POS[i]
    w_T = dfs_fundamental_windings(nn_T, POS_T, LX, LY)
    axt = max(abs(a) for a, b in w_T); ayt = max(abs(b) for a, b in w_T)
    print(f"    column-major node order: max|wx|={axc}, max|wy|={ayc} -> large on {'y' if ayc > axc else 'x'}")
    print(f"    row-major (transposed):  max|wx|={axt}, max|wy|={ayt} -> large on {'y' if ayt > axt else 'x'}")
    C_pass = (ayc > axc) and (ayt > axt)   # large winding stays on y under both orderings
    print(f"    (C) {'PASS' if C_pass else 'FAIL'}: axis did not flip -> not node-numbering; lattice-geometry driven.")

    print("=" * 70)
    print("Triangular neighbor offsets (dcol,drow) -> (dx,dy): horizontal is straight, vertical staggered")
    i0 = idx(7, 6); c0i, r0i = cr(i0)
    for j in nn[i0]:
        j = int(j); cj, rj = cr(j)
        dcr = (((cj - c0i + G // 2) % G) - G // 2, ((rj - r0i + G // 2) % G) - G // 2)
        dx, dy = wrap_disp(POS, LX, LY, i0, j)
        print(f"    {dcr} -> ({dx:+.3f}, {dy:+.3f})")

    print("=" * 70)
    ok = A_pass and B_pass and C_pass
    print(f"OVERALL: {'PASS - footnote claims verified' if ok else 'FAIL - re-examine'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
