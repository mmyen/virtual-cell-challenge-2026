"""
Local stand-in for the VCC 2026 leaderboard, scored on cells.

There is no held-out single-cell data, so the reference is a proxy: a held-out
screen's MEASURED effects applied to one context's real control cells. The
model is fit without that screen (LOCO), so it predicts it zero-shot. The
submission and the proxy reference then go through the official pipeline:

    pseudobulk profiles      -> pds_cosine, expr_mse_unbiased_capped_norm
    per-cell Wilcoxon + BH   -> fidelity, reach, sig_jaccard, lfc_nmae

and every metric is scaled as (u - b) / (r - b) with
    b = baseline:  the proxy reference's mean effect, the same for every target
    r = replicate: two disjoint 200-cell draws of the proxy reference, one
                   scored against the other

Formulas follow the metric brief:
https://github.com/ArcInstitute/cell-eval2/blob/main/docs/vcc2026_metrics/vcc2026-metrics-brief.md

The context's controls are split in two:
    comp -- the comparison group of every DE test, and the cells the
            submission is generated from (as in the real submission)
    pool -- the cells the proxy reference is generated from (real perturbed
            cells are different cells from the controls)

Known gaps vs. the real leaderboard: proxy reference cells shift every cell by
the same mean multiplier (real perturbed cells vary more, so n_real is
probably inflated); the truth is another cell line's pseudobulk; the panel is
the sampled targets, not 300 (PDS is easier with fewer candidates); and the
replicate's halves share one control group instead of each having its own.
"""

import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import norm

TS_BULK = 5e4  # bulk_target_sum, sections 1-2
FDR = 0.05     # p_adj_threshold
P0 = 0.9       # REACH purity floor
MIN_GATE = 10  # NMAE min_gate_size
METRICS = ["pds", "mse", "fid", "reach", "jac", "nmae"]


# ----------------------------------------------------------------------------
# DE test (section 3)
# ----------------------------------------------------------------------------
class FastWilcoxon:
    # Wilcoxon rank-sum of many small groups against one fixed, large control group.

    def __init__(self, ctrl_counts, min_cpm=5.0):
        X = sp.csr_matrix(ctrl_counts, dtype=np.float64)
        cpm = sp.diags(1e6 / np.asarray(X.sum(axis=1)).ravel()) @ X
        self.genes = np.where(np.asarray(cpm.mean(axis=0)).ravel() >= min_cpm)[0]  # global gene indices
        self.ctrl = np.sort(cpm[:, self.genes].toarray().astype(np.float32), axis=0)
        self.ctrl_zeros = (self.ctrl <= 0).sum(axis=0)
        self.ctrl_mean = self.ctrl.mean(axis=0)

    def test(self, counts):
        # counts: (cells, all genes) raw counts -> two-sided p, z (>0 = up), log2FC, per tested gene
        X = sp.csr_matrix(counts, dtype=np.float64)
        lib = np.maximum(np.asarray(X.sum(axis=1)).ravel(), 1)
        P = (sp.diags(1e6 / lib) @ X)[:, self.genes].toarray().astype(np.float32)
        n1, n2 = P.shape[0], self.ctrl.shape[0]
        U = np.empty(P.shape[1])
        for j in range(P.shape[1]):
            c = self.ctrl[:, j]
            lo = np.searchsorted(c, P[:, j], "left")
            hi = np.searchsorted(c, P[:, j], "right")
            U[j] = lo.sum() + 0.5 * (hi - lo).sum()
        t = (self.ctrl_zeros + (P <= 0).sum(axis=0)).astype(np.float64)
        N = n1 + n2
        var = n1 * n2 / 12.0 * ((N + 1) - (t ** 3 - t) / (N * (N - 1)))
        z = (U - n1 * n2 / 2.0) / np.sqrt(np.maximum(var, 1e-12))
        p = 2 * norm.sf(np.abs(z))
        lfc = np.log2((P.mean(axis=0) + 1e-9) / (self.ctrl_mean + 1e-9))
        return p, z, lfc


def bh(p):
    n = len(p)
    order = np.argsort(p)
    q = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1)
    return out


# ----------------------------------------------------------------------------
# pseudobulk profile + jackknife noise (sections 1-2)
# ----------------------------------------------------------------------------
def bulk_profile(counts, chunk=1000):
    """
    counts: (cells, genes) raw counts of one group.
    Returns (b, C): b = log1p(TS * P / S), the group's pseudobulk profile, and
    C = per-gene delete-one jackknife variance of b (sum over genes = C_p).
    Chunked so the ~9k-cell control group fits in memory.
    """
    X = sp.csr_matrix(counts, dtype=np.float64)
    n = X.shape[0]
    P = np.asarray(X.sum(axis=0)).ravel()
    S = P.sum()
    b = np.log1p(TS_BULK * P / S)
    if n < 2:
        return b, np.zeros_like(b)
    lib = np.asarray(X.sum(axis=1)).ravel()
    s1 = np.zeros_like(P)
    s2 = np.zeros_like(P)
    for start in range(0, n, chunk):
        y = X[start:start + chunk].toarray()
        v = np.log1p(TS_BULK * (P - y) / (S - lib[start:start + chunk])[:, None])
        s1 += v.sum(axis=0)
        s2 += (v ** 2).sum(axis=0)
    C = (n - 1) / n * np.maximum(s2 - s1 ** 2 / n, 0)
    return b, C


# ----------------------------------------------------------------------------
# cells
# ----------------------------------------------------------------------------
def cells_from_effect(control_X, effect, n_cells, rng, cells=None):
    """
    n_cells real control cells (rows of control_X, sampled without
    replacement unless `cells` gives the row indices) with every gene scaled
    by 2**effect and stochastically rounded -- the same recipe as
    Model._predicted_cells, but for an arbitrary log2 effect vector.
    """
    idx = rng.choice(control_X.shape[0], n_cells, replace=False) if cells is None else cells
    X = control_X[idx]
    X = X.toarray() if sp.issparse(X) else np.asarray(X)
    scaled = X * (2.0 ** np.asarray(effect, dtype=np.float64))
    out = np.floor(scaled)
    out += rng.random(scaled.shape) < (scaled - out)
    return sp.csr_matrix(np.clip(out, 0, None).astype(np.float32))


# ----------------------------------------------------------------------------
# scorer
# ----------------------------------------------------------------------------
class ProxyScorer:
    """
    comp_counts: (cells, genes) raw counts of the comparison control group.
    Summarize every group with .summarize(), then score a submission's
    summaries against the reference's with .score().
    """

    def __init__(self, comp_counts):
        self.tester = FastWilcoxon(comp_counts)
        self.pos = {g: i for i, g in enumerate(self.tester.genes)}
        self.b_ctrl, self.C_ctrl = bulk_profile(comp_counts)

    def summarize(self, counts):
        b, C = bulk_profile(counts)
        p, _, lfc = self.tester.test(counts)
        padj = bh(p)
        return {"b": b, "C": C, "p": p, "padj": padj, "lfc": lfc, "sig": padj < FDR}

    def score(self, sub, ref, target_gis):
        """
        sub, ref: lists of summaries, one per target, same order.
        target_gis: each target's global gene index (or -1 if not a gene).
        Returns the raw panel aggregate of every metric (not yet scaled).
        """
        n = len(ref)
        G = len(self.b_ctrl)
        panel = np.array([g for g in target_gis if g >= 0])
        tpos = [self.pos.get(g, -1) for g in target_gis]  # position among DE-tested genes

        # --- 1. PDS: cosine distance of effects, all panel target genes removed
        keep = np.ones(G, bool)
        keep[panel] = False
        D_hat = np.stack([s["b"][keep] - self.b_ctrl[keep] for s in sub])
        D = np.stack([r["b"][keep] - self.b_ctrl[keep] for r in ref])
        nh = np.linalg.norm(D_hat, axis=1)[:, None]
        nr = np.linalg.norm(D, axis=1)[None, :]
        with np.errstate(invalid="ignore", divide="ignore"):
            cos = (D_hat @ D.T) / (nh * nr)
        dist = np.where((nh == 0) | (nr == 0), 1.0, 1 - np.nan_to_num(cos))
        own = np.diag(dist)[:, None]
        k = (dist < own).sum(axis=1) + 0.5 * ((dist == own).sum(axis=1) - 1)
        pds = float(np.mean(1 - k / (n - 1)))

        # --- 2. MSE: unbiased, capped, normalized; own target gene removed
        M = np.ones((n, G), bool)
        for i, g in enumerate(target_gis):
            if g >= 0:
                M[i, g] = False
        Gp = M.sum(axis=1)
        Bh = np.stack([s["b"] for s in sub])
        Ch = np.stack([s["C"] for s in sub])
        B = np.stack([r["b"] for r in ref])
        Cr = np.stack([r["C"] for r in ref])
        minC = (np.minimum(Ch, Cr) * M).sum(axis=1)
        Cp = (Cr * M).sum(axis=1)
        w = (M / Gp[:, None])
        mean_w = (w * Bh).sum(axis=0) / np.maximum(w.sum(axis=0), 1e-300)
        spread = (w * (Bh - mean_w) ** 2).sum()
        denom_rho = (minC / Gp).sum()
        rho = min(1.0, spread / denom_rho) if denom_rho > 0 else 1.0
        N = (((Bh - B) ** 2 * M).sum(axis=1) - rho * minC - Cp) / Gp
        Dn = (((B - self.b_ctrl) ** 2 * M).sum(axis=1) - Cp - (self.C_ctrl * M).sum(axis=1)) / Gp
        mse = float(N.sum() / Dn.sum())  # unclamped here; the clamp is on the scaled score

        # --- 4-7. DE metrics
        fid, reach, jac, nmae = [], [], [], []
        for s, r, tp in zip(sub, ref, tpos):
            not_target = np.ones(len(r["sig"]), bool)
            if tp >= 0:
                not_target[tp] = False
            r_sig = r["sig"] & not_target
            s_sig = s["sig"] & not_target
            adj = np.isfinite(r["lfc"]) & (r["lfc"] != 0) | np.isinf(r["lfc"])
            same = np.sign(s["lfc"]) == np.sign(r["lfc"])
            n_real = int(r_sig.sum())

            # 4. fidelity: k / max(n_pred, n_real)
            calls = s_sig & adj
            n_pred, k = int(calls.sum()), int((calls & same).sum())
            if n_real > 0:
                fid.append(k / max(n_pred, n_real))
            elif n_pred > 0:
                fid.append(k / n_pred)

            # 5. reach: deepest prefix of the submission-ordered pool with purity >= P0
            pool = np.where(r_sig)[0]
            if len(pool):
                order = np.lexsort((pool, -np.abs(s["lfc"][pool]), s["p"][pool],
                                    s["padj"][pool], ~s["sig"][pool]))
                walk = pool[order]
                walk = walk[adj[walk]]
                hits = np.cumsum(same[walk])
                purity = hits / np.arange(1, len(walk) + 1)
                ok = np.where(purity >= P0)[0]
                reach.append((ok[-1] + 1 if len(ok) else 0) / n_real)

            # 6. jaccard of significant sets (empty union = 1)
            union = (r_sig | s_sig).sum()
            jac.append((r_sig & s_sig).sum() / union if union else 1.0)

            # 7. NMAE over the reference's significant genes with finite lfc
            gate = r_sig & np.isfinite(r["lfc"])
            if gate.sum() >= MIN_GATE:
                den = np.abs(r["lfc"][gate]).sum()
                if den > 0 and np.isfinite(den):
                    nmae.append(np.abs(s["lfc"][gate] - r["lfc"][gate]).sum() / den)

        mean = lambda x: float(np.mean(x)) if len(x) else np.nan
        return {"pds": pds, "mse": mse, "fid": mean(fid), "reach": mean(reach),
                "jac": mean(jac), "nmae": mean(nmae),
                "n_sig": float(np.median([int((s["sig"]).sum()) for s in sub]))}


def scale(raw, base, rep):
    """
    (u - b) / (r - b) per metric, with the brief's clamps on the SCALED score:
    MSE to [0, 1] (so MSE can never go below the baseline's 0), NMAE floored at -6.
    """
    out = {m: (raw[m] - base[m]) / (rep[m] - base[m]) for m in METRICS}
    out["mse"] = float(np.clip(out["mse"], 0, 1))
    out["nmae"] = max(out["nmae"], -6.0)
    out["overall"] = float(np.mean([out[m] for m in METRICS]))
    return out


# ----------------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------------
def proxy_leaderboard(model, ctrl_adata, truth, targets, gains, gene_index,
                      kd=0.9, alpha=1.5, n_cells=400, seed=0, verbose=True):
    """
    model:      a fitted Model whose set_context() was already called for the
                held-out screen (fit WITHOUT it).
    ctrl_adata: one context's control cells (raw counts, fully loaded).
    truth:      DataFrame, targets x global gene index, the held-out screen's
                measured log2FC; genes it didn't measure are treated as 0.
    targets:    target symbols to score (must be rows of truth).
    gains:      gains to try for the model's cells.
    Returns (scaled, raw): DataFrames indexed by submission
    ("model gain g", "baseline", "replicate"), one column per metric.
    """
    rng = np.random.default_rng(seed)
    G = ctrl_adata.n_vars
    half = rng.permutation(ctrl_adata.n_obs)
    comp_idx, pool_idx = np.sort(half[: len(half) // 2]), np.sort(half[len(half) // 2:])
    comp = ctrl_adata[comp_idx].copy()
    pool_X = ctrl_adata.X[pool_idx]

    t0 = time.time()
    scorer = ProxyScorer(comp.X)
    if verbose:
        print(f"scorer: {len(scorer.tester.genes)} DE-tested genes, "
              f"{comp.n_obs} comparison / {len(pool_idx)} pool cells ({time.time() - t0:.0f}s)")

    # measured effects as full-length log2 vectors; own gene gets the knockdown
    E = np.zeros((len(targets), G))
    cols = np.asarray(truth.columns)
    E[:, cols] = truth.loc[targets].to_numpy()
    target_gis = [gene_index.get(t, -1) for t in targets]
    for i, g in enumerate(target_gis):
        if g >= 0:
            E[i, g] = np.log2(1 - kd)

    # proxy reference (400 cells) and replicate halves (2 x 200, disjoint)
    t0 = time.time()
    ref, rep_a, rep_b = [], [], []
    for e in E:
        ref.append(scorer.summarize(cells_from_effect(pool_X, e, n_cells, rng)))
        two = rng.choice(len(pool_idx), n_cells, replace=False)
        rep_a.append(scorer.summarize(cells_from_effect(pool_X, e, None, rng, cells=two[: n_cells // 2])))
        rep_b.append(scorer.summarize(cells_from_effect(pool_X, e, None, rng, cells=two[n_cells // 2:])))
    if verbose:
        print(f"reference + replicate: {time.time() - t0:.0f}s, "
              f"median n_real {np.median([int(r['sig'].sum()) for r in ref]):.0f}")

    # baseline: the brief scores the mean per-construct count VECTOR as a
    # submission, so its "cells" are n_cells identical copies of the expected
    # counts (control mean x mean multiplier) -- no sampling noise, which is why
    # it calls many genes in the DE test and gets no noise credit in MSE
    mu_ctrl = np.asarray(comp.X.mean(axis=0)).ravel()
    base_vec = mu_ctrl * np.mean(2.0 ** E, axis=0)
    base_cells = scorer.summarize(np.tile(base_vec, (n_cells, 1)))
    raw = {"baseline": scorer.score([base_cells] * len(targets), ref, target_gis),
           "replicate": scorer.score(rep_a, rep_b, target_gis)}

    for g in gains:
        t0 = time.time()
        sub = [scorer.summarize(model._predicted_cells(comp, t, n_cells, rng, kd, alpha,
                                                       gain=g, rounding="stochastic"))
               for t in targets]
        raw[f"model gain {g}"] = scorer.score(sub, ref, target_gis)
        if verbose:
            print(f"gain {g}: {time.time() - t0:.0f}s")

    raw = pd.DataFrame(raw).T
    scaled = pd.DataFrame({k: scale(raw.loc[k], raw.loc["baseline"], raw.loc["replicate"])
                           for k in raw.index}).T
    scaled["n_sig"] = raw["n_sig"]
    scaled.loc["proxy reference", "n_sig"] = float(np.median([int(r["sig"].sum()) for r in ref]))
    return scaled, raw
