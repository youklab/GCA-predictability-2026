# Tabular/MLP model sweep across trajectory fraction f (3-way fate task)
# One seed (0), full 850k training pool, evaluated on the frozen 100k test set.
import os, sys, csv, json, time, gc
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, accuracy_score, recall_score

SPLITS = r"<PATH_TO_SPLITS>"
OUT    = r"<PATH_TO_RESULTS>"
os.makedirs(OUT, exist_ok=True)
RAW = os.path.join(OUT, "sweep_results_raw.csv")
FRACS = ["0.00","0.10","0.20","0.30","0.40","0.50","0.60","0.70","0.80","0.90","1.00"]
SEED = 0
HDR = ["model","fraction_f","balanced_accuracy","accuracy","recall_static","recall_spiral",
       "recall_rectilinear","n_train","n_test","seed","train_seconds","notes"]

def log(m):
    print("[%s] %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), m), flush=True)

def load_split(frac):
    d = os.path.join(SPLITS, "ml_splits_v260529_Ttrue_f" + frac)
    Xtr = np.load(d + "/X_train_pool.npy")
    Xva = np.load(d + "/X_val.npy")
    Xte = np.load(d + "/X_test.npy")
    ytr = np.load(d + "/meta_train_pool.npz")["final_pattern_type"].astype(np.int64) - 1
    yva = np.load(d + "/meta_val.npz")["final_pattern_type"].astype(np.int64) - 1
    yte = np.load(d + "/meta_test.npz")["final_pattern_type"].astype(np.int64) - 1
    return Xtr, ytr, Xva, yva, Xte, yte

def class_weights(y):
    cnt = np.bincount(y, minlength=3).astype(np.float64)
    return len(y) / (3.0 * np.maximum(cnt, 1.0))

def score(yte, pred):
    ba  = balanced_accuracy_score(yte, pred)
    acc = accuracy_score(yte, pred)
    rec = recall_score(yte, pred, average=None, labels=[0,1,2], zero_division=0)
    return ba, acc, rec

def append_row(row):
    isnew = not os.path.exists(RAW)
    with open(RAW, "a", newline="") as fh:
        w = csv.writer(fh)
        if isnew: w.writerow(HDR)
        w.writerow(row)

def already_done():
    s = set()
    if os.path.exists(RAW):
        with open(RAW) as fh:
            for r in csv.DictReader(fh): s.add((r["model"], r["fraction_f"]))
    return s

def onehot(Xtr, Xva, Xte, ncell):
    enc = OneHotEncoder(categories=[np.array([1,2,3,4])]*ncell, sparse_output=True,
                        handle_unknown="ignore", dtype=np.float32)
    return enc.fit_transform(Xtr), enc.transform(Xva), enc.transform(Xte)

def run_logreg(Xtr,ytr,Xva,yva,Xte,yte):
    Atr, Ava, Ate = onehot(Xtr, Xva, Xte, Xtr.shape[1])
    t0 = time.time()
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs", n_jobs=-1)
    clf.fit(Atr, ytr)
    pred = clf.predict(Ate)
    return score(yte, pred) + (time.time()-t0, "onehot+lbfgs,class_weight=balanced,max_iter=2000")

def run_xgb(Xtr,ytr,Xva,yva,Xte,yte):
    import xgboost as xgb
    Atr, Ava, Ate = onehot(Xtr, Xva, Xte, Xtr.shape[1])
    cw = class_weights(ytr)
    t0 = time.time()
    p = dict(objective="multi:softprob", num_class=3, eval_metric="mlogloss",
             learning_rate=0.05, max_depth=6, min_child_weight=1, subsample=0.8,
             colsample_bytree=0.8, reg_lambda=1.0, tree_method="hist", nthread=28, seed=SEED)
    dtr = xgb.DMatrix(Atr, label=ytr, weight=cw[ytr])
    dva = xgb.DMatrix(Ava, label=yva, weight=cw[yva])
    bst = xgb.train(p, dtr, num_boost_round=1200, evals=[(dva,"val")],
                    early_stopping_rounds=60, verbose_eval=200)
    best = bst.best_iteration + 1
    pred = bst.predict(xgb.DMatrix(Ate), iteration_range=(0,best)).argmax(1)
    return score(yte, pred) + (time.time()-t0, "onehot,hist,lr0.05,depth6,rounds=%d" % best)

def run_mlp(Xtr,ytr,Xva,yva,Xte,yte):
    import torch, torch.nn as nn, torch.nn.functional as F
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(SEED); np.random.seed(SEED)
    ncell = Xtr.shape[1]; nin = ncell*4
    Xtr_t = torch.from_numpy(Xtr.astype(np.uint8)); ytr_t = torch.from_numpy(ytr)
    Xva_t = torch.from_numpy(Xva.astype(np.uint8)); Xte_t = torch.from_numpy(Xte.astype(np.uint8))
    def enc(xb):
        return F.one_hot(xb.long()-1, 4).reshape(xb.shape[0], -1).float().to(dev)
    layers = []; dcur = nin
    for h in [2048,2048,1024,512]:
        layers += [nn.Linear(dcur,h), nn.ReLU(), nn.Dropout(0.25)]; dcur = h
    layers += [nn.Linear(dcur,3)]
    net = nn.Sequential(*layers).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-5)
    cw  = torch.tensor(class_weights(ytr), dtype=torch.float32, device=dev)
    lossf = nn.CrossEntropyLoss(weight=cw)
    def predict(Xt):
        net.eval(); out=[]
        with torch.no_grad():
            for i in range(0, Xt.shape[0], 8192):
                out.append(net(enc(Xt[i:i+8192])).argmax(1).cpu().numpy())
        return np.concatenate(out)
    t0=time.time(); best_ba=-1.0; best_state=None; bad=0; n=Xtr_t.shape[0]; nep=0
    for ep in range(50):
        net.train(); perm = torch.randperm(n)
        for i in range(0, n, 2048):
            j = perm[i:i+2048]
            xb = enc(Xtr_t[j]); yb = ytr_t[j].to(dev)
            opt.zero_grad(); loss = lossf(net(xb), yb); loss.backward(); opt.step()
        vba = balanced_accuracy_score(yva, predict(Xva_t)); nep = ep+1
        log("      MLP epoch %d val_balacc %.4f" % (nep, vba))
        if vba > best_ba + 1e-4:
            best_ba = vba; bad = 0
            best_state = {k: v.detach().cpu().clone() for k,v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 6: break
    if best_state is not None: net.load_state_dict(best_state)
    pred = predict(Xte_t)
    return score(yte, pred) + (time.time()-t0, "torch MLP 2048-2048-1024-512 drop0.25 bs2048 epochs=%d" % nep)

RUNNERS = [("logreg", run_logreg), ("xgboost", run_xgb), ("mlp", run_mlp)]

def main():
    done = already_done()
    log("starting sweep; already done: %d cells" % len(done))
    for mname, fn in RUNNERS:
        for frac in FRACS:
            if (mname, frac) in done:
                log("skip %s f=%s (done)" % (mname, frac)); continue
            log("=== %s  f=%s ===" % (mname, frac))
            try:
                Xtr,ytr,Xva,yva,Xte,yte = load_split(frac)
                ba, acc, rec, secs, note = fn(Xtr,ytr,Xva,yva,Xte,yte)
                append_row([mname, frac, "%.6f"%ba, "%.6f"%acc, "%.6f"%rec[0], "%.6f"%rec[1],
                            "%.6f"%rec[2], Xtr.shape[0], Xte.shape[0], SEED, "%.1f"%secs, note])
                log("    DONE %s f=%s balacc=%.4f (%.1f s)" % (mname, frac, ba, secs))
                del Xtr,ytr,Xva,yva,Xte,yte; gc.collect()
            except Exception as e:
                log("    FAILED %s f=%s : %r" % (mname, frac, e))
                append_row([mname, frac, "", "", "", "", "", "", "", SEED, "", "FAILED: %r" % e])
    log("SWEEP COMPLETE")

if __name__ == "__main__":
    main()