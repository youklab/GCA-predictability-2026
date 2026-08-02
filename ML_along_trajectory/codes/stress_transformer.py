#!/usr/bin/env python
# ---------------------------------------------------------------------------
# stress_transformer.py   (v2)
#
# A toroidal transformer as a stress test for extractable fate signal at
# f = 0 (the initial configuration) in the generalized cellular automaton.
#
# WHY A TRANSFORMER, AND WHY THIS ONE
# -----------------------------------
# The MLP already sees all 196 cells jointly, so raw "long-range access" is
# not the missing ingredient. What this architecture adds is the correct
# SYMMETRY: the fate label is invariant under lattice translations on the
# torus. Attention logits are biased by a learned function of the TOROIDAL
# DISPLACEMENT (dr mod 14, dc mod 14) only, giving translation-equivariant
# attention; the readout pools over tokens, giving a translation-invariant
# classifier. The hypothesis class is therefore restricted to the invariant
# functions, which is the statistically efficient way to hunt a weak signal.
#
# v2 FIX (important, and found by the positive control)
# ----------------------------------------------------
# In v1 the relative-bias table was initialized to zeros. That makes the model
# fully PERMUTATION-invariant at initialization, not merely translation-
# invariant, so it can only use the 4-number state histogram -- exactly the
# information available to logistic regression. It duly scored 0.34 at
# f = 0.50, which is logistic regression's number, and could not escape within
# the training budget. v2 initializes the bias with a LOCALITY PRIOR
# (bias = -scale_h * triangular-lattice toroidal distance, with a different
# range per head), so attention starts local like a CNN and can unlearn that
# if the data warrant. Translation equivariance is preserved exactly.
#
# Readout is now mean-pool concatenated with a learned attention pool (both
# permutation-invariant, so invariance is preserved) followed by an MLP head.
#
# POSITIONAL / TOKENIZATION SCHEMES
#   rel  : relative bias only. Fully translation-equivariant/invariant.
#   abs  : rel bias + learned absolute position embedding + toroidal Fourier
#          features. Strictly more expressive, NOT equivariant.
#   stem : --stem k prepends k circular-padded 3x3 convolutions before
#          tokenizing, hard-wiring local features (still translation-
#          equivariant, since circular conv commutes with torus translation).
#
# BUILT-IN CONTROLS (these are what make a null result mean anything)
#   --labels permuted : train/val labels shuffled. Gives the null distribution
#                       of TEST balanced accuracy through the entire pipeline,
#                       including early stopping and model selection. Any
#                       "above chance" result at f=0 must beat this.
#   --frac 0.50       : positive control. The same code and budget must find
#                       substantial signal here, or an f=0 null is worthless.
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


# ------------------------------- data --------------------------------------
def load_split(frac, n_train, dev):
    d = os.path.join(SPLITS, "ml_splits_v260529_Ttrue_f%s" % frac)
    Xtr = np.load(d + "/X_train_pool.npy")
    Xva = np.load(d + "/X_val.npy")
    Xte = np.load(d + "/X_test.npy")
    ytr = np.load(d + "/meta_train_pool.npz")["final_pattern_type"].astype(np.int64) - 1
    yva = np.load(d + "/meta_val.npz")["final_pattern_type"].astype(np.int64) - 1
    yte = np.load(d + "/meta_test.npz")["final_pattern_type"].astype(np.int64) - 1
    if 0 < n_train < Xtr.shape[0]:
        rng = np.random.default_rng(12345)          # fixed subsample across seeds
        idx = rng.permutation(Xtr.shape[0])[:n_train]
        Xtr, ytr = Xtr[idx], ytr[idx]

    def prep(X):
        return torch.from_numpy(X.astype(np.int64) - 1).to(dev)   # values 0..3

    return (prep(Xtr), torch.from_numpy(ytr).to(dev), ytr,
            prep(Xva), torch.from_numpy(yva).to(dev), yva,
            prep(Xte), yte)


def wrap(d):
    """signed toroidal displacement in (-L/2, L/2]"""
    d = d % L
    return torch.where(d > L // 2, d - L, d)


def tri_toroidal_distance():
    """Graph distance on a triangular lattice with periodic boundaries,
    in axial coordinates: d = (|a| + |b| + |a+b|) / 2."""
    pos = torch.arange(NCELL)
    r, c = pos // L, pos % L
    a = wrap(r[:, None] - r[None, :]).float()
    b = wrap(c[:, None] - c[None, :]).float()
    return (a.abs() + b.abs() + (a + b).abs()) / 2.0      # (196,196)


def toroidal_fourier(n_harm):
    pos = torch.arange(NCELL)
    r = (pos // L).float(); c = (pos % L).float()
    feats = []
    for k in range(1, n_harm + 1):
        for coord in (r, c):
            feats.append(torch.sin(2 * math.pi * k * coord / L))
            feats.append(torch.cos(2 * math.pi * k * coord / L))
    return torch.stack(feats, dim=-1)


# ------------------------------- model -------------------------------------
class ToroidalAttention(nn.Module):
    """Attention biased by toroidal displacement only, initialized local."""

    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.dk = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.pdrop = dropout

        pos = torch.arange(NCELL)
        r, c = pos // L, pos % L
        dr = (r[:, None] - r[None, :]) % L
        dc = (c[:, None] - c[None, :]) % L
        self.register_buffer("ridx", (dr * L + dc).reshape(-1).long(), persistent=False)

        # locality prior: one bias value per toroidal displacement per head,
        # initialized to -scale_h * lattice distance (different range per head)
        dist = tri_toroidal_distance()                       # (196,196)
        flat = torch.zeros(L * L)
        flat[self.ridx.reshape(NCELL, NCELL)[0]] = dist[0]   # displacement -> distance
        scales = torch.logspace(math.log10(2.0), math.log10(0.25), n_heads)
        self.rel = nn.Parameter(-scales[:, None] * flat[None, :])

    def forward(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        bias = self.rel[:, self.ridx].reshape(1, self.h, N, N)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=bias,
            dropout_p=self.pdrop if self.training else 0.0)
        out = out.transpose(1, 2).reshape(B, N, D)
        return self.proj(out)


class Block(nn.Module):
    def __init__(self, d_model, n_heads, ffn_mult, dropout):
        super().__init__()
        self.n1 = nn.LayerNorm(d_model)
        self.att = ToroidalAttention(d_model, n_heads, dropout)
        self.n2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, ffn_mult * d_model), nn.GELU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(ffn_mult * d_model, d_model))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.drop(self.att(self.n1(x)))
        x = x + self.drop(self.ffn(self.n2(x)))
        return x


class AttnPool(nn.Module):
    """Learned-query attention pooling. Permutation-invariant."""

    def __init__(self, d):
        super().__init__()
        self.q = nn.Parameter(torch.randn(d) * 0.02)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)

    def forward(self, z):
        a = (self.k(z) @ self.q) / math.sqrt(z.shape[-1])     # (B,N)
        w = a.softmax(dim=-1).unsqueeze(-1)
        return (w * self.v(z)).sum(dim=1)


class ToroidalTransformer(nn.Module):
    def __init__(self, d_model=128, n_heads=4, depth=4, ffn_mult=4,
                 dropout=0.05, pos="rel", n_harm=4, stem=0):
        super().__init__()
        self.pos_mode, self.n_stem = pos, stem
        if stem > 0:
            layers, cin = [], 4
            for _ in range(stem):
                layers += [nn.Conv2d(cin, d_model, 3, padding=1,
                                     padding_mode="circular"), nn.GELU()]
                cin = d_model
            self.stem = nn.Sequential(*layers)
        else:
            self.emb = nn.Embedding(4, d_model)
        if pos == "abs":
            self.abs_pos = nn.Parameter(torch.randn(NCELL, d_model) * 0.02)
            self.register_buffer("fourier", toroidal_fourier(n_harm), persistent=False)
            self.fproj = nn.Linear(4 * n_harm, d_model)
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, ffn_mult, dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(d_model)
        self.pool = AttnPool(d_model)
        self.head = nn.Sequential(nn.Linear(2 * d_model, 256), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(256, 3))

    def forward(self, x):
        if self.n_stem > 0:
            g = F.one_hot(x, 4).to(torch.float32).permute(0, 2, 1).reshape(-1, 4, L, L)
            z = self.stem(g).reshape(x.shape[0], -1, L * L).transpose(1, 2)
        else:
            z = self.emb(x)
        if self.pos_mode == "abs":
            z = z + self.abs_pos.unsqueeze(0) + self.fproj(self.fourier).unsqueeze(0)
        for b in self.blocks:
            z = b(z)
        z = self.norm(z)
        return self.head(torch.cat([z.mean(dim=1), self.pool(z)], dim=-1))


# ------------------------------- train -------------------------------------
def evaluate(net, X, bs=4096):
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
        log("  LABELS PERMUTED (null run, seed %d)" % a.seed, fh)

    n = Xtr.shape[0]
    cnt = np.bincount(ytr_np, minlength=3).astype(np.float64)
    cw = torch.tensor(n / (3.0 * np.maximum(cnt, 1.0)), dtype=torch.float32, device=dev)
    lossf = nn.CrossEntropyLoss(weight=cw)

    net = ToroidalTransformer(d_model=a.d_model, n_heads=a.n_heads, depth=a.depth,
                              ffn_mult=a.ffn_mult, dropout=a.dropout,
                              pos=a.pos, stem=a.stem).to(dev)
    nparam = sum(p.numel() for p in net.parameters())
    log("  device=%s n_train=%d params=%.2fM pos=%s stem=%d depth=%d d=%d h=%d"
        % (dev, n, nparam / 1e6, a.pos, a.stem, a.depth, a.d_model, a.n_heads), fh)

    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = math.ceil(n / a.batch) * a.epochs
    warm = max(1, int(0.02 * steps))

    def lr_at(s):
        if s < warm:
            return s / warm
        p = (s - warm) / max(1, steps - warm)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, p)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

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
            opt.step(); sched.step()
        vba = balanced_accuracy_score(yva_np, evaluate(net, Xva)); nep = ep + 1
        log("    epoch %2d  val_balacc %.4f  (%.1f s)" % (nep, vba, time.time() - te), fh)
        if vba > best + 1e-4:
            best, bad = vba, 0
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= a.patience:
                log("    early stop (patience %d)" % a.patience, fh); break
    if best_state is not None:
        net.load_state_dict(best_state)

    pred = evaluate(net, Xte)
    ba = balanced_accuracy_score(yte_np, pred)
    acc = accuracy_score(yte_np, pred)
    rec = recall_score(yte_np, pred, average=None, labels=[0, 1, 2], zero_division=0)
    secs = time.time() - t0

    row = [a.tag, "toroidal_transformer", a.pos + ("+stem%d" % a.stem if a.stem else ""),
           a.labels, a.frac, a.depth, a.d_model, a.n_heads,
           "%.6f" % ba, "%.6f" % acc, "%.6f" % rec[0], "%.6f" % rec[1], "%.6f" % rec[2],
           n, Xte.shape[0], a.seed, nep, "%.6f" % best, nparam, "%.1f" % secs,
           "locality-prior rel bias; attn+mean pool; bs%d lr%.1e drop%.2f"
           % (a.batch, a.lr, a.dropout)]

    with open(os.path.join(OUT, "shard_%s.csv" % a.tag), "w", newline="") as f:
        w = csv.writer(f); w.writerow(HDR); w.writerow(row)
    master = os.path.join(OUT, "f0_stress_results.csv")
    isnew = not os.path.exists(master)
    with open(master, "a", newline="") as f:
        w = csv.writer(f)
        if isnew: w.writerow(HDR)
        w.writerow(row)
    log("  ==> %s  TEST balacc = %.4f  (chance 0.3333)  [%.1f s]" % (a.tag, ba, secs), fh)
    return ba


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--frac", default="0.00")
    p.add_argument("--labels", default="real", choices=["real", "permuted"])
    p.add_argument("--pos", default="rel", choices=["rel", "abs"])
    p.add_argument("--stem", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_train", type=int, default=200000)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--ffn_mult", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--patience", type=int, default=4)
    a = p.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if os.path.exists(os.path.join(OUT, "shard_%s.csv" % a.tag)):
        print("skip %s (shard exists)" % a.tag, flush=True); return
    with open(os.path.join(OUT, "log_%s.txt" % a.tag), "w") as fh:
        log("=== %s : f=%s labels=%s pos=%s stem=%d seed=%d ==="
            % (a.tag, a.frac, a.labels, a.pos, a.stem, a.seed), fh)
        run_one(a, fh)


if __name__ == "__main__":
    main()
