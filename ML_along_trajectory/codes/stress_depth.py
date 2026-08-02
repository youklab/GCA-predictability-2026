#!/usr/bin/env python
# ---------------------------------------------------------------------------
# stress_depth.py
#
# Depth sweep at f = 0. Three model families, each with computational depth
# as the swept variable:
#
#   mlp : fully connected, `depth` hidden layers of fixed width. No spatial
#         structure. Tests whether raw nonlinear capacity helps.
#
#   cnn : `depth` 3x3 convolutions with CIRCULAR padding, then global mean
#         pooling. Circular padding matters: with zero padding a CNN
#         structurally cannot see across the torus boundary, so it is blind
#         to exactly the wrapping structure this system is about. Global mean
#         pooling makes the classifier translation-invariant, matching the
#         symmetry of the fate label. Receptive field grows as 2*depth+1, so
#         depth >= 7 already covers the whole 14x14 torus.
#
#   nca : recurrent "neural cellular automaton". A SINGLE shared-weight
#         convolutional update block applied `depth` times, then a global
#         readout. This is the model family that could in principle learn to
#         SIMULATE the dynamics rather than pattern-match it. Weight sharing
#         keeps the parameter count fixed while computational depth grows, so
#         this isolates depth-as-computation from depth-as-capacity.
#
# NOTE ON SCOPE: the true transient length T_true is far larger than the
# unroll depths reachable here, so this sweep tests whether depth helps at
# all in the tractable regime. It does NOT reach the depth at which a model
# could simulate the automaton to its attractor. That crossover would need
# unrolling of order T_true steps.
#
# Every family also runs a positive control at f = 0.50 so that a null at
# f = 0 cannot be blamed on a broken training pipeline.
#
# Labels: final_pattern_type {1,2,3} -> {0,1,2} = static, spiral, rectilinear
# Chance balanced accuracy = 1/3.
# ---------------------------------------------------------------------------

import os, sys, csv, time, math, argparse
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, accuracy_score, recall_score

SPLITS = r"<PATH_TO_SPLITS>"
OUT    = r"<PATH_TO_RESULTS>"
L      = 14
NCELL  = L * L
HDR = ["tag", "model", "pos", "labels", "fraction_f", "depth", "d_model", "n_heads",
       "balanced_accuracy", "accuracy", "recall_static", "recall_spiral",
       "recall_rectilinear", "n_train", "n_test", "seed", "epochs",
       "val_balacc_best", "params", "train_seconds", "notes"]


def log(msg, fh=None):
    s = "[%s] %s" % (time.strftime("%H:%M:%S"), msg)
    print(s, flush=True)
    if fh is not None:
        fh.write(s + "\n"); fh.flush()


def load_split(frac, n_train, dev):
    d = os.path.join(SPLITS, "ml_splits_v260529_Ttrue_f%s" % frac)
    Xtr = np.load(d + "/X_train_pool.npy")
    Xva = np.load(d + "/X_val.npy")
    Xte = np.load(d + "/X_test.npy")
    ytr = np.load(d + "/meta_train_pool.npz")["final_pattern_type"].astype(np.int64) - 1
    yva = np.load(d + "/meta_val.npz")["final_pattern_type"].astype(np.int64) - 1
    yte = np.load(d + "/meta_test.npz")["final_pattern_type"].astype(np.int64) - 1
    if 0 < n_train < Xtr.shape[0]:
        rng = np.random.default_rng(12345)
        idx = rng.permutation(Xtr.shape[0])[:n_train]
        Xtr, ytr = Xtr[idx], ytr[idx]

    def prep(X):
        # one-hot to (N,4,14,14) float on device
        t = torch.from_numpy(X.astype(np.int64) - 1)
        oh = F.one_hot(t, 4).to(torch.float32)            # (N,196,4)
        return oh.permute(0, 2, 1).reshape(-1, 4, L, L).contiguous().to(dev)

    return (prep(Xtr), torch.from_numpy(ytr).to(dev), ytr,
            prep(Xva), torch.from_numpy(yva).to(dev), yva,
            prep(Xte), yte)


# ------------------------------- models ------------------------------------
class MLPd(nn.Module):
    def __init__(self, depth, width=1024, dropout=0.25):
        super().__init__()
        layers, dcur = [], 4 * NCELL
        for _ in range(depth):
            layers += [nn.Linear(dcur, width), nn.ReLU(), nn.Dropout(dropout)]
            dcur = width
        layers += [nn.Linear(dcur, 3)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.reshape(x.shape[0], -1))


class CNNd(nn.Module):
    """`depth` circular-padded 3x3 convs, then translation-invariant readout."""

    def __init__(self, depth, ch=64, dropout=0.1):
        super().__init__()
        blocks, cin = [], 4
        for _ in range(depth):
            blocks += [nn.Conv2d(cin, ch, 3, padding=1, padding_mode="circular"),
                       nn.BatchNorm2d(ch), nn.ReLU()]
            cin = ch
        self.body = nn.Sequential(*blocks)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(ch, 3)

    def forward(self, x):
        z = self.body(x).mean(dim=(2, 3))                 # global mean pool
        return self.head(self.drop(z))


class NCAd(nn.Module):
    """Shared-weight recurrent update applied `depth` times (fixed params)."""

    def __init__(self, depth, ch=32, dropout=0.1):
        super().__init__()
        self.depth = depth
        self.stem = nn.Conv2d(4, ch, 1)
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1, padding_mode="circular")
        self.c2 = nn.Conv2d(ch, ch, 1)
        self.norm = nn.GroupNorm(4, ch)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(ch, 3)

    def forward(self, x):
        z = self.stem(x)
        for _ in range(self.depth):
            z = z + self.c2(F.relu(self.c1(self.norm(z))))   # residual CA step
        z = z.mean(dim=(2, 3))
        return self.head(self.drop(z))


def build(model, depth):
    if model == "mlp":
        return MLPd(depth)
    if model == "cnn":
        return CNNd(depth)
    if model == "nca":
        return NCAd(depth)
    raise ValueError(model)


# ------------------------------- train -------------------------------------
def evaluate(net, X, bs=8192):
    net.eval(); out = []
    with torch.no_grad():
        for i in range(0, X.shape[0], bs):
            out.append(net(X[i:i + bs]).argmax(1).cpu().numpy())
    return np.concatenate(out)


def run_one(a, fh):
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    Xtr, ytr, ytr_np, Xva, yva, yva_np, Xte, yte_np = load_split(a.frac, a.n_train, dev)

    if a.labels == "permuted":
        rng = np.random.default_rng(9000 + a.seed)
        ytr_np = ytr_np[rng.permutation(len(ytr_np))]
        yva_np = yva_np[rng.permutation(len(yva_np))]
        ytr = torch.from_numpy(ytr_np).to(dev)
        log("  LABELS PERMUTED (null run)", fh)

    n = Xtr.shape[0]
    cnt = np.bincount(ytr_np, minlength=3).astype(np.float64)
    cw = torch.tensor(n / (3.0 * np.maximum(cnt, 1.0)), dtype=torch.float32, device=dev)
    lossf = nn.CrossEntropyLoss(weight=cw)

    net = build(a.model, a.depth).to(dev)
    nparam = sum(p.numel() for p in net.parameters())
    log("  %s depth=%d  n_train=%d  params=%.2fM  device=%s"
        % (a.model, a.depth, n, nparam / 1e6, dev), fh)

    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-5)
    t0 = time.time(); best, best_state, bad, nep = -1.0, None, 0, 0
    for ep in range(a.epochs):
        te = time.time(); net.train()
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, a.batch):
            j = perm[i:i + a.batch]
            opt.zero_grad(set_to_none=True)
            loss = lossf(net(Xtr[j]), ytr[j])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        vba = balanced_accuracy_score(yva_np, evaluate(net, Xva)); nep = ep + 1
        log("    epoch %2d  val_balacc %.4f  (%.1f s)" % (nep, vba, time.time() - te), fh)
        if vba > best + 1e-4:
            best, bad = vba, 0
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= a.patience:
                log("    early stop", fh); break
    if best_state is not None:
        net.load_state_dict(best_state)

    pred = evaluate(net, Xte)
    ba = balanced_accuracy_score(yte_np, pred)
    acc = accuracy_score(yte_np, pred)
    rec = recall_score(yte_np, pred, average=None, labels=[0, 1, 2], zero_division=0)

    # ----------------------- simulation-vs-shortcut diagnostics ------------
    # (1) Save per-example predictions. The test-set row order is fixed, so
    #     these can be joined against meta_test.npz['T'] afterwards to ask
    #     whether accuracy depends on how long a trajectory actually took to
    #     reach its attractor. A fixed-depth simulator cannot adapt its
    #     computation to trajectory length, so it should do relatively better
    #     on short-transient trajectories. A shortcut predictor need not.
    np.savez_compressed(os.path.join(OUT, "pred_%s.npz" % a.tag),
                        pred=pred.astype(np.int8), y=yte_np.astype(np.int8))

    # (2) Re-evaluate the SAME trained weights at other unroll counts. Because
    #     the update block is weight-shared, this is legal and free. A network
    #     that learned a genuine time-invariant local rule should tolerate
    #     being run for a different number of steps. One that fitted a
    #     fixed-depth feedforward circuit should degrade sharply off its
    #     training depth.
    if a.model == "nca":
        eds = sorted({max(1, a.depth // 4), max(1, a.depth // 2), a.depth,
                      a.depth * 2, a.depth * 4})
        rows_ed = []
        for ed in eds:
            net.depth = ed
            b = balanced_accuracy_score(yte_np, evaluate(net, Xte))
            rows_ed.append((ed, b))
            log("    re-evaluated at unroll %-4d -> balacc %.4f" % (ed, b), fh)
        net.depth = a.depth
        with open(os.path.join(OUT, "evaldepth_%s.csv" % a.tag), "w", newline="") as f:
            w2 = csv.writer(f)
            w2.writerow(["tag", "fraction_f", "labels", "trained_unrolls",
                         "eval_unrolls", "balanced_accuracy"])
            for ed, b in rows_ed:
                w2.writerow([a.tag, a.frac, a.labels, a.depth, ed, "%.6f" % b])

    secs = time.time() - t0

    note = {"mlp": "width1024 drop0.25",
            "cnn": "ch64 3x3 circular-pad, global-mean-pool",
            "nca": "ch32 shared-weight residual CA step, unroll=depth"}[a.model]
    row = [a.tag, a.model, "-", a.labels, a.frac, a.depth, "-", "-",
           "%.6f" % ba, "%.6f" % acc, "%.6f" % rec[0], "%.6f" % rec[1], "%.6f" % rec[2],
           n, Xte.shape[0], a.seed, nep, "%.6f" % best, nparam, "%.1f" % secs,
           note + " bs%d lr%.1e" % (a.batch, a.lr)]

    with open(os.path.join(OUT, "shard_%s.csv" % a.tag), "w", newline="") as f:
        w = csv.writer(f); w.writerow(HDR); w.writerow(row)
    master = os.path.join(OUT, "f0_stress_results.csv")
    isnew = not os.path.exists(master)
    with open(master, "a", newline="") as f:
        w = csv.writer(f)
        if isnew: w.writerow(HDR)
        w.writerow(row)
    log("  ==> %s  TEST balacc = %.4f   (chance 0.3333)  [%.1f s]" % (a.tag, ba, secs), fh)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--model", required=True, choices=["mlp", "cnn", "nca"])
    p.add_argument("--depth", type=int, required=True)
    p.add_argument("--frac", default="0.00")
    p.add_argument("--labels", default="real", choices=["real", "permuted"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_train", type=int, default=200000)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--patience", type=int, default=4)
    a = p.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if os.path.exists(os.path.join(OUT, "shard_%s.csv" % a.tag)):
        print("skip %s (shard exists)" % a.tag, flush=True); return
    with open(os.path.join(OUT, "log_%s.txt" % a.tag), "w") as fh:
        log("=== %s : %s depth=%d f=%s labels=%s ===" % (a.tag, a.model, a.depth, a.frac, a.labels), fh)
        run_one(a, fh)


if __name__ == "__main__":
    main()
