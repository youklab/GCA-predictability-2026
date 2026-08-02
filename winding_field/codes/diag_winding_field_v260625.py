import importlib.util, numpy as np, os

MP = "<PATH_TO_CODE>/Elinor_secrete_sense_module.py"
SPL = "<PATH_TO_SPLITS>"
G = 14; N = 196
spec = importlib.util.spec_from_file_location("esm", MP)
esm = importlib.util.module_from_spec(spec); spec.loader.exec_module(esm)

def build(snap):
    t = esm.CellularLattice(G, N, 0.2, 1.5, 75000)
    t.init_lattice(G, [1, 1], 1.5, 0.2, "triangular")
    t.init_general_parameters(np.array([[3,10],[11,4]]), np.array([18,16]),
                              np.array([[1,1],[-1,0]]), np.array([1,1.2]))
    t.quiet_warnings = True
    cs = np.repeat(np.asarray(snap, int).reshape(N,1), 2, axis=1)
    t.cell_4state = cs; t.tmax = 2
    return t

for frac in [0.30, 0.50, 0.90]:
    X = np.array(np.load(os.path.join(SPL, f"ml_splits_v260529_Ttrue_f{frac:.2f}", "X_train_pool.npy"), mmap_mode="r")[:50])
    nonzero_frac = []
    multi_winding_clusters = 0
    total_wrapping_clusters = 0
    for i in range(50):
        t = build(X[i])
        labels = t.label_phase_clusters(0, phase_value=None)   # per-cell cluster id
        loops = t.find_noncontractible_loops(0, phase_value=None)  # wrapping loops with windings
        # map each wrapping loop to the cluster(s) its nodes belong to
        cluster_windings = {}   # cluster_id -> set of (wx,wy)
        for lp in loops:
            nodes = lp['nodes']; wx, wy = lp['windings']
            cids = set(int(labels[n]) for n in nodes if labels[n] > 0)
            for cid in cids:
                cluster_windings.setdefault(cid, set()).add((wx, wy))
        total_wrapping_clusters += len(cluster_windings)
        multi_winding_clusters += sum(1 for s in cluster_windings.values() if len(s) > 1)
        # build per-cell (wx,wy) by componentwise max-abs over the cluster's loop windings
        field = np.zeros((N, 2), int)
        for cid, wset in cluster_windings.items():
            wx = max((w[0] for w in wset), key=abs, default=0)
            wy = max((w[1] for w in wset), key=abs, default=0)
            field[labels == cid] = (wx, wy)
        nonzero_frac.append(np.mean(np.any(field != 0, axis=1)))
    print(f"f={frac:.2f}: mean fraction of cells with nonzero (wx,wy) = {np.mean(nonzero_frac):.3f}  "
          f"| wrapping clusters/config = {total_wrapping_clusters/50:.2f}  "
          f"| multi-winding clusters (need aggregation) = {multi_winding_clusters}")
