#!/usr/bin/env python
# ---------------------------------------------------------------------------
# stress_hmm.py
#
# Hidden Markov model applied to the initial configuration (Reviewer 2's
# second request). CPU-only (vectorized numpy), so it can run concurrently
# with GPU work.
#
# HOW AN HMM IS APPLIED TO A STATIC 2D CONFIGURATION
# --------------------------------------------------
# An HMM models sequences, so the 14x14 lattice must first be serialized into
# a sequence of 196 discrete symbols. We use three scan orders:
#   row        : row-major raster scan
#   serpentine : boustrophedon scan (every other row reversed), so that
#                consecutive symbols are always spatial neighbours
#   col        : column-major scan
# A first-order chain along any scan captures along-scan adjacency well and
# across-scan adjacency poorly (the vertical neighbour sits 14 steps away in a
# row scan). This is an intrinsic limitation of applying an HMM to a 2D field
# and is reported as such, not hidden.
#
# CLASSIFIER
# ----------
# Standard generative Bayes construction: fit one HMM per fate class on that
# class's training sequences, then assign a test configuration to whichever
# class HMM gives it the higher log-likelihood. With uniform class priors this
# is the Bayes rule for balanced error, which matches balanced accuracy as the
# reported metric. Empirical-prior accuracy is recorded alongside it.
#
# CONTROLS
#   --labels permuted : training labels shuffled before fitting -> null
#   --frac 0.50       : positive control at mid-trajectory
#
# Labels: final_pattern_type {1,2,3} -> {0,1,2} = static, spiral, rectilinear
# Chance balanced accuracy = 1/3.
# ---------------------------------------------------------------------------

import os, csv, time, math, argparse
import numpy as np
from sklearn.metrics import balanced_accuracy_score, accuracy_score, recall_score

SPLITS = r"<PATH_TO_SPLITS>"
OUT    = r"<PATH_TO_RESULTS>"
L, NCELL, NSYM, NCLS = 14, 196, 4, 3
HDR = ["tag", "model", "scan", "labels", "fraction_f", "n_states",
       "balanced_accuracy", "accuracy", "recall_static", "recall_spiral",
       "recall_rectilinear", "n_train_per_class", "n_test", "seed",
       "em_iters", "mean_test_loglik", "pred_frac_static", "pred_frac_spiral",
       "pred_frac_rectilinear", "train_seconds", "notes"]


def log(m, fh=None):
    s = "[%s] %s" % (time.strftime("%H:%M:%S"), m)
    print(s, flush=True)
    if fh:
        fh.write(s + "\n"); fh.flush()


# ------------------------------ scan orders --------------------------------
def scan_index(kind):
    idx = np.arange(NCELL).reshape(L, L)
    if kind == "row":
        pass
    elif kind == "serpentine":
        idx = idx.copy()
        idx[1::2, :] = idx[1::2, ::-1]
    elif kind == "col":
        idx = idx.T
    else:
        raise ValueError(kind)
    return idx.reshape(-1)


# ------------------------------ data ---------------------------------------
def load(frac):
    d = os.path.join(SPLITS, "ml_splits_v260529_Ttrue_f%s" % frac)
    Xtr = np.load(d + "/X_train_pool.npy").astype(np.int64) - 1
    Xte = np.load(d + "/X_test.npy").astype(np.int64) - 1
    ytr = np.load(d + "/meta_train_pool.npz")["final_pattern_type"].astype(np.int64) - 1
    yte = np.load(d + "/meta_test.npz")["final_pattern_type"].astype(np.int64) - 1
    return Xtr, ytr, Xte, yte


# ------------------------- scaled forward-backward -------------------------
def forward_scaled(X, pi, A, B):
    """Return alpha (N,T,K) normalized per t, and c (N,T) scale factors."""
    N, T = X.shape
    K = pi.shape[0]
    alpha = np.empty((N, T, K))
    c = np.empty((N, T))
    a = pi[None, :] * B[:, X[:, 0]].T
    c[:, 0] = a.sum(1)
    alpha[:, 0, :] = a / c[:, 0:1]
    for t in range(1, T):
        a = (alpha[:, t - 1, :] @ A) * B[:, X[:, t]].T
        c[:, t] = a.sum(1)
        alpha[:, t, :] = a / np.maximum(c[:, t:t + 1], 1e-300)
    return alpha, c


def loglik_scaled(X, pi, A, B, chunk=4000):
    """Log P(x) per sequence, forward pass only (memory-light)."""
    N, T = X.shape
    K = pi.shape[0]
    out = np.empty(N)
    for s in range(0, N, chunk):
        Xc = X[s:s + chunk]
        n = Xc.shape[0]
        a = pi[None, :] * B[:, Xc[:, 0]].T
        cs = a.sum(1)
        ll = np.log(np.maximum(cs, 1e-300))
        a = a / np.maximum(cs[:, None], 1e-300)
        for t in range(1, T):
            a = (a @ A) * B[:, Xc[:, t]].T
            cs = a.sum(1)
            ll += np.log(np.maximum(cs, 1e-300))
            a = a / np.maximum(cs[:, None], 1e-300)
        out[s:s + n] = ll
    return out


def fit_hmm(X, K, n_iter, seed, chunk=2000, tol=1e-4, fh=None, label=""):
    """Baum-Welch for a categorical HMM, vectorized over sequences."""
    rng = np.random.default_rng(seed)
    N, T = X.shape
    pi = rng.dirichlet(np.ones(K))
    A = rng.dirichlet(np.ones(K), size=K)
    B = rng.dirichlet(np.ones(NSYM) * 5.0, size=K)
    prev = -np.inf
    for it in range(n_iter):
        pi_num = np.zeros(K)
        A_num = np.zeros((K, K))
        B_num = np.zeros((K, NSYM))
        gsum = np.zeros(K)
        total_ll = 0.0
        for s in range(0, N, chunk):
            Xc = X[s:s + chunk]
            n = Xc.shape[0]
            alpha, c = forward_scaled(Xc, pi, A, B)
            total_ll += np.log(np.maximum(c, 1e-300)).sum()
            # backward
            beta = np.empty_like(alpha)
            beta[:, T - 1, :] = 1.0
            for t in range(T - 2, -1, -1):
                bt = beta[:, t + 1, :] * B[:, Xc[:, t + 1]].T
                beta[:, t, :] = (bt @ A.T) / np.maximum(c[:, t + 1:t + 2], 1e-300)
            gamma = alpha * beta
            gamma /= np.maximum(gamma.sum(2, keepdims=True), 1e-300)
            # xi accumulation
            Bem = B[:, Xc[:, 1:]].transpose(1, 2, 0)          # (n, T-1, K)
            term = (beta[:, 1:, :] * Bem) / np.maximum(c[:, 1:, None], 1e-300)
            A_num += A * np.einsum("nti,ntj->ij", alpha[:, :-1, :], term)
            pi_num += gamma[:, 0, :].sum(0)
            gsum += gamma.sum((0, 1))
            for sym in range(NSYM):
                m = (Xc == sym)
                if m.any():
                    B_num[:, sym] += gamma[m].sum(0)
        pi = pi_num / max(pi_num.sum(), 1e-300)
        A = A_num / np.maximum(A_num.sum(1, keepdims=True), 1e-300)
        B = B_num / np.maximum(B_num.sum(1, keepdims=True), 1e-300)
        ll_per = total_ll / N
        if fh and (it % 4 == 0 or it == n_iter - 1):
            log("      %s EM iter %2d  loglik/seq %.4f" % (label, it + 1, ll_per), fh)
        if abs(ll_per - prev) < tol * abs(prev):
            break
        prev = ll_per
    return pi, A, B, it + 1, ll_per


# ------------------------------- main --------------------------------------
def run(a, fh):
    Xtr, ytr, Xte, yte = load(a.frac)
    order = scan_index(a.scan)
    Xtr, Xte = Xtr[:, order], Xte[:, order]

    if a.labels == "permuted":
        rng = np.random.default_rng(7000 + a.seed)
        ytr = ytr[rng.permutation(len(ytr))]
        log("  LABELS PERMUTED (null run)", fh)

    rng = np.random.default_rng(4242 + a.seed)
    t0 = time.time()
    models, iters = [], []
    for c in range(NCLS):
        idx = np.where(ytr == c)[0]
        rng.shuffle(idx)
        idx = idx[:a.n_per_class]
        Xc = np.ascontiguousarray(Xtr[idx])
        log("    fitting class %d HMM on %d sequences (K=%d, scan=%s)"
            % (c, Xc.shape[0], a.n_states, a.scan), fh)
        pi, A, B, ni, ll = fit_hmm(Xc, a.n_states, a.em_iters, a.seed * 100 + c,
                                  fh=fh, label="c%d" % c)
        models.append((pi, A, B)); iters.append(ni)

    LL = np.empty((Xte.shape[0], NCLS))
    for c, (pi, A, B) in enumerate(models):
        LL[:, c] = loglik_scaled(Xte, pi, A, B)
        log("    scored test set under class %d HMM" % c, fh)

    pred_uniform = LL.argmax(1)                       # uniform priors
    prior = np.log(np.bincount(ytr, minlength=NCLS) / len(ytr))
    pred_emp = (LL + prior[None, :]).argmax(1)

    ba = balanced_accuracy_score(yte, pred_uniform)
    acc = accuracy_score(yte, pred_emp)
    rec = recall_score(yte, pred_uniform, average=None, labels=[0, 1, 2], zero_division=0)
    pf = np.bincount(pred_uniform, minlength=NCLS) / len(pred_uniform)
    secs = time.time() - t0

    row = [a.tag, "hmm", a.scan, a.labels, a.frac, a.n_states,
           "%.6f" % ba, "%.6f" % acc, "%.6f" % rec[0], "%.6f" % rec[1], "%.6f" % rec[2],
           a.n_per_class, Xte.shape[0], a.seed, max(iters),
           "%.4f" % LL.max(1).mean(), "%.4f" % pf[0], "%.4f" % pf[1], "%.4f" % pf[2],
           "%.1f" % secs,
           "per-class categorical HMM, scaled Baum-Welch, uniform-prior argmax"]

    with open(os.path.join(OUT, "shard_%s.csv" % a.tag), "w", newline="") as f:
        w = csv.writer(f); w.writerow(HDR); w.writerow(row)
    log("  ==> %s  TEST balacc = %.4f  (chance 0.3333)  pred fracs %.2f/%.2f/%.2f  [%.1f s]"
        % (a.tag, ba, pf[0], pf[1], pf[2], secs), fh)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--scan", default="serpentine", choices=["row", "serpentine", "col"])
    p.add_argument("--n_states", type=int, default=8)
    p.add_argument("--frac", default="0.00")
    p.add_argument("--labels", default="real", choices=["real", "permuted"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_per_class", type=int, default=8000)
    p.add_argument("--em_iters", type=int, default=25)
    a = p.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if os.path.exists(os.path.join(OUT, "shard_%s.csv" % a.tag)):
        print("skip %s (shard exists)" % a.tag, flush=True); return
    with open(os.path.join(OUT, "log_%s.txt" % a.tag), "w") as fh:
        log("=== %s : HMM scan=%s K=%d f=%s labels=%s ==="
            % (a.tag, a.scan, a.n_states, a.frac, a.labels), fh)
        run(a, fh)


if __name__ == "__main__":
    main()
