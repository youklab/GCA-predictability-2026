import numpy as np, os
FD = "<PATH_TO_OUTPUT>/winding_fields_v260625"
SPL = "<PATH_TO_OUTPUT>/ml_splits_v260529"

for frac in [0.00, 0.50, 0.90]:
    z = np.load(os.path.join(FD, f"fields_Ttrue_f{frac:.2f}.npz"))
    w = z["winding"]; v = z["vortex"]
    X = np.load(os.path.join(SPL, f"ml_splits_v260529_Ttrue_f{frac:.2f}", "X_train_pool.npy"), mmap_mode="r")
    print(f"f={frac:.2f}: winding {w.shape} {w.dtype}  vortex {v.shape} {v.dtype}  X {X.shape}")
    print(f"   rows match X: {w.shape[0]==X.shape[0]}")
    # winding density + value range
    nz = np.mean(np.any(w != 0, axis=1))   # frac of cells with nonzero (wx,wy)
    print(f"   winding: frac cells nonzero = {nz:.3f}  wx range [{w[:, 0].min()},{w[:, 0].max()}]  wy range [{w[:, 1].min()},{w[:, 1].max()}]")
    # vortex: mean core cells, net charge (should be ~0 at f>=0.3, noisy at f=0)
    pos = v[:, 0].sum(axis=1); neg = v[:, 1].sum(axis=1)
    print(f"   vortex: mean +cells={pos.mean():.1f} mean -cells={neg.mean():.1f} mean net={ (pos-neg).mean():.2f}")
    # winding-channel cells should be cells whose CLUSTER wraps -> nonzero only where a real region is; spot value histogram
    vals_x = np.unique(w[:, 0]); print(f"   wx values present: {vals_x}")
