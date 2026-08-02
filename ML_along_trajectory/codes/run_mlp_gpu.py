# GPU-resident MLP sweep. Entire one-hot dataset lives on the MPS device;
# no CPU work per batch. Each fraction writes its own CSV shard so that
# several fractions can be run concurrently without clobbering one file.
import os, sys, csv, time, argparse
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, accuracy_score, recall_score

SPLITS = r"<PATH_TO_SPLITS>"
OUT    = r"<PATH_TO_RESULTS>"
SEED = 0

def log(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)

def load_gpu(frac, dev):
    d = os.path.join(SPLITS, "ml_splits_v260529_Ttrue_f" + frac)
    out = []
    for xn, mn in (("X_train_pool","meta_train_pool"), ("X_val","meta_val"), ("X_test","meta_test")):
        X = np.load(d + "/" + xn + ".npy")
        y = np.load(d + "/" + mn + ".npz")["final_pattern_type"].astype(np.int64) - 1
        Xi = torch.from_numpy(X.astype(np.int64))
        Xoh = F.one_hot(Xi - 1, 4).reshape(X.shape[0], -1).to(torch.float32).to(dev)
        out.append((Xoh, torch.from_numpy(y).to(dev), y))
        del Xi
    return out

def run_fraction(frac, dev, max_epochs, batch, patience):
    torch.manual_seed(SEED); np.random.seed(SEED)
    (Xtr,ytr,_), (Xva,yva,yva_np), (Xte,yte,yte_np) = load_gpu(frac, dev)
    nin = Xtr.shape[1]; n = Xtr.shape[0]
    log("  f=%s data on device: train %s  (%.2f GB)" % (frac, tuple(Xtr.shape), Xtr.numel()*4/1e9))
    cnt = torch.bincount(ytr, minlength=3).float()
    cw  = (n / (3.0 * torch.clamp(cnt, min=1.0))).to(dev)
    layers = []; dcur = nin
    for h in [2048,2048,1024,512]:
        layers += [nn.Linear(dcur,h), nn.ReLU(), nn.Dropout(0.25)]; dcur = h
    layers += [nn.Linear(dcur,3)]
    net = nn.Sequential(*layers).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-5)
    lossf = nn.CrossEntropyLoss(weight=cw)
    def predict(X):
        net.eval(); out=[]
        with torch.no_grad():
            for i in range(0, X.shape[0], 16384):
                out.append(net(X[i:i+16384]).argmax(1).cpu().numpy())
        return np.concatenate(out)
    t0=time.time(); best=-1.0; best_state=None; bad=0; nep=0
    for ep in range(max_epochs):
        te=time.time(); net.train()
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, batch):
            j = perm[i:i+batch]
            opt.zero_grad(set_to_none=True)
            loss = lossf(net(Xtr[j]), ytr[j])
            loss.backward(); opt.step()
        vba = balanced_accuracy_score(yva_np, predict(Xva)); nep = ep+1
        log("  f=%s epoch %2d  val_balacc %.4f  (%.1f s/epoch)" % (frac, nep, vba, time.time()-te))
        if vba > best + 1e-4:
            best = vba; bad = 0
            best_state = {k: v.detach().cpu().clone() for k,v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience: break
    if best_state is not None: net.load_state_dict(best_state)
    pred = predict(Xte)
    ba  = balanced_accuracy_score(yte_np, pred)
    acc = accuracy_score(yte_np, pred)
    rec = recall_score(yte_np, pred, average=None, labels=[0,1,2], zero_division=0)
    secs = time.time()-t0
    note = "torch MLP 2048-2048-1024-512 drop0.25 bs%d epochs=%d gpu-resident" % (batch, nep)
    shard = os.path.join(OUT, "mlp_shard_f%s.csv" % frac)
    with open(shard, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model","fraction_f","balanced_accuracy","accuracy","recall_static",
                    "recall_spiral","recall_rectilinear","n_train","n_test","seed","train_seconds","notes"])
        w.writerow(["mlp", frac, "%.6f"%ba, "%.6f"%acc, "%.6f"%rec[0], "%.6f"%rec[1], "%.6f"%rec[2],
                    Xtr.shape[0], Xte.shape[0], SEED, "%.1f"%secs, note])
    log("  f=%s DONE balacc=%.4f  (%.1f s total)" % (frac, ba, secs))
    del Xtr,ytr,Xva,yva,Xte,yte,net
    if dev.type == "mps": torch.mps.empty_cache()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fracs", default="0.00,0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,1.00")
    ap.add_argument("--max_epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--patience", type=int, default=6)
    a = ap.parse_args()
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    log("device=%s  fracs=%s  batch=%d" % (dev, a.fracs, a.batch))
    for frac in a.fracs.split(","):
        shard = os.path.join(OUT, "mlp_shard_f%s.csv" % frac)
        if os.path.exists(shard):
            log("skip f=%s (shard exists)" % frac); continue
        log("=== MLP f=%s ===" % frac)
        try:
            run_fraction(frac, dev, a.max_epochs, a.batch, a.patience)
        except Exception as e:
            log("  FAILED f=%s : %r" % (frac, e))
    log("WORKER COMPLETE")

if __name__ == "__main__":
    main()