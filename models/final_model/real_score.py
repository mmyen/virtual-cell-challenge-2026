"""
Score a model against REAL held-out Perturb-seq lines with the VCC 2026 metrics.

Same scoring code as proxy_score.py (official formulas, scaled against the
mean-vector baseline and a replicate), but the reference is a line's real
perturbed cells instead of measured effects pasted onto control cells.

Lines (neither is a training line, so fit the model on all four training screens):
    HCT116 -- X-Atlas/Orion subset (data/xaira/), 298 of the 300 VCC 2026 test
              targets, every one also perturbed in K562 -> exercises direct transfer.
              Few cells per target (median ~133), so fewer real DE calls than VCC.
              Its replicate halves (~90 cells) call almost nothing, so the replicate
              is barely better than the baseline and SCALED scores blow up (FID, MSE
              hit the clamps). Compare settings on HCT116 with the RAW table.
    H1     -- VCC 2025 test set (data/vcc2025/adata_Test.h5ad), 100 targets; 92 of
              the 97 kept ones were perturbed in K562, so like the VCC 2026 panel
              (272/300 covered) it exercises direct transfer AND the embedding
              (SLIM) path. Thousands of cells per target; ~54.6k UMI/cell, so load
              it with target_umi=20000 to match the VCC contexts' depth.

Setup mirrors the real leaderboard: the predicted cells are sampled from the
line's control cells, and the DE test compares every group against those same
controls. The reference is up to `max_ref_cells` real cells per target (VCC's
ground truth has ~460), and the replicate is two disjoint halves of them.

Typical use (notebook in this folder):

    from real_score import load_line, set_line_context, score_line
    line = load_line("HCT116", gene_names)
    set_line_context(model, line, dp, train_names)
    scaled, raw = score_line(model, line, {"gain 2.5": {"gain": 2.5},
                                           "N200 top2.5": {"gain": 1.0, "n_call": 200, "gain_top": 2.5}})
"""

import time
from dataclasses import dataclass, field

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from proxy_score import ProxyScorer, scale

DATA = "../../data"
CONTROL = "non-targeting"


@dataclass
class RealLine:
    name: str
    ctrl: ad.AnnData                       # control cells, raw counts, VCC gene order
    cells: dict = field(repr=False)        # target -> CSR (cells, VCC genes) raw counts
    dropped: dict = field(default_factory=dict, repr=False)  # target -> reason

    @property
    def targets(self):
        return list(self.cells)


# ----------------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------------
def _to_vcc_genes(X, src_genes, vcc_genes):
    """Reorder/subset the columns of X (cells x src_genes) to vcc_genes; genes the
    source didn't measure become all-zero columns."""
    col = pd.Index(src_genes).get_indexer(vcc_genes)          # vcc position -> source column
    ok = col >= 0
    S = sp.csr_matrix((np.ones(ok.sum(), np.float32), (col[ok], np.where(ok)[0])),
                      shape=(len(src_genes), len(vcc_genes)))
    return sp.csr_matrix(X, dtype=np.float32) @ S


def _thin(X, p, rng):
    """Binomial thinning: every UMI is kept independently with probability p. This is
    what sequencing the same library p times as deeply would give, so DE power drops to
    the shallower regime while each cell's relative depth is preserved."""
    X = sp.csr_matrix(X, dtype=np.float32, copy=True)
    X.data = rng.binomial(np.rint(X.data).astype(np.int64), p).astype(np.float32)
    X.eliminate_zeros()
    return X


def _pick_rows(obs, n_ctrl, max_ref_cells, min_cells, targets, rng):
    """Row indices of the controls (up to n_ctrl) and of each target (up to max_ref_cells)."""
    tg = obs["target_gene"].astype(str).to_numpy()
    ctrl_rows = np.flatnonzero(tg == CONTROL)
    if len(ctrl_rows) > n_ctrl:
        ctrl_rows = np.sort(rng.choice(ctrl_rows, n_ctrl, replace=False))
    groups = pd.Series(np.arange(len(tg))).groupby(tg).indices
    keep, dropped = {}, {}
    for t in (targets if targets is not None else sorted(groups)):
        if t == CONTROL:
            continue
        rows = groups.get(t)
        if rows is None:
            dropped[t] = "not in data"
        elif len(rows) < min_cells:
            dropped[t] = f"only {len(rows)} cells"
        else:
            keep[t] = np.sort(rng.choice(rows, min(len(rows), max_ref_cells), replace=False))
    return ctrl_rows, keep, dropped


def load_line(name, vcc_genes, n_ctrl=18400, max_ref_cells=400, min_cells=100,
              targets=None, seed=0, target_umi=None):
    """
    name:          "HCT116" or "H1".
    vcc_genes:     the 18,533 VCC gene symbols (data/controls/gene_names.csv), = model.gene_names.
    n_ctrl:        control cells to keep (VCC contexts have 18,400).
    max_ref_cells: real cells per target used as the reference (VCC ground truth ~460).
    min_cells:     targets with fewer real cells are dropped (their DE calls and
                   replicate are too noisy to score).
    targets:       optional list to restrict to; default = every target in the line.
    target_umi:    if set, binomially thin all cells (controls and perturbed) by ONE
                   keep-fraction p = target_umi / median control UMI/cell (on the VCC
                   genes), so the controls' median depth becomes target_umi. One shared p
                   keeps the cell-to-cell depth spread and any perturbed-vs-control
                   library-size difference. VCC 2026 contexts A-F all have a median of
                   ~20k UMI/cell; H1 is ~54.6k, which inflates its DE power, so use
                   target_umi=20000 for H1. None = native depth (also if already shallower).
    """
    rng = np.random.default_rng(seed)
    vcc_genes = list(vcc_genes)
    t0 = time.time()
    if name == "HCT116":
        ctrl = ad.read_h5ad(f"{DATA}/xaira/HCT116_controls.h5ad")
        pert = ad.read_h5ad(f"{DATA}/xaira/HCT116_perturbed.h5ad")
        assert list(ctrl.var_names) == vcc_genes and list(pert.var_names) == vcc_genes, \
            "HCT116 files are expected in VCC gene order"
        obs = pd.concat([ctrl.obs[["target_gene"]], pert.obs[["target_gene"]]])
        ctrl_rows, keep, dropped = _pick_rows(obs, n_ctrl, max_ref_cells, min_cells, targets, rng)
        Xc = sp.csr_matrix(ctrl.X, dtype=np.float32)
        Xp = sp.csr_matrix(pert.X, dtype=np.float32)
        nc = ctrl.n_obs
        ctrl_X = Xc[ctrl_rows]
        cells = {t: Xp[rows - nc] for t, rows in keep.items()}
    elif name == "H1":
        a = ad.read_h5ad(f"{DATA}/vcc2025/adata_Test.h5ad", backed="r")
        ctrl_rows, keep, dropped = _pick_rows(a.obs, n_ctrl, max_ref_cells, min_cells, targets, rng)
        all_rows = np.concatenate([ctrl_rows] + list(keep.values()))
        order = np.argsort(all_rows)
        sub = a[all_rows[order]].to_memory()                   # one sorted read of the 12 GB file
        X = _to_vcc_genes(sub.X, a.var_names, vcc_genes)
        a.file.close()
        X = X[np.argsort(order)]                               # back to all_rows order
        ctrl_X = X[: len(ctrl_rows)]
        cells, start = {}, len(ctrl_rows)
        for t, rows in keep.items():
            cells[t] = X[start:start + len(rows)]
            start += len(rows)
    else:
        raise ValueError(f"unknown line {name!r} (HCT116 or H1)")

    depth = ""
    if target_umi is not None:
        before = np.median(np.asarray(ctrl_X.sum(axis=1)).ravel())
        p = min(1.0, target_umi / before)
        if p < 1.0:
            ctrl_X = _thin(ctrl_X, p, rng)
            cells = {t: _thin(m, p, rng) for t, m in cells.items()}
        after = np.median(np.asarray(ctrl_X.sum(axis=1)).ravel())
        depth = f", control UMI/cell median {before:.0f} -> {after:.0f} (p={p:.3f})"

    # targets the model can't take (target gene not in the VCC gene list)
    gset = set(vcc_genes)
    for t in [t for t in cells if t not in gset]:
        dropped[t] = "target not in VCC gene list"
        del cells[t]

    ctrl = ad.AnnData(X=sp.csr_matrix(ctrl_X), var=pd.DataFrame(index=vcc_genes))
    ctrl.obs["target_gene"] = CONTROL
    n = [m.shape[0] for m in cells.values()]
    print(f"{name}: {ctrl.n_obs} controls, {len(cells)} targets "
          f"(cells per target median {np.median(n):.0f}, min {min(n)}), "
          f"{len(dropped)} dropped{depth} ({time.time() - t0:.0f}s)")
    return RealLine(name, ctrl, cells, dropped)


def set_line_context(model, line, dp, train_names, tau=0.05, mix=0.3):
    """Point the model's context-aware mean (b_c) at this line, as write_submission does
    for a VCC context: control profile -> similarity weights, panel = the line's targets."""
    profile = pd.Series(dp.context_ctrl_mean(line.ctrl), index=model.gene_names)
    model.set_context(profile, dp.source_ctrls(train_names), tau=tau, mix=mix, panel=line.targets)
    print(line.name, dict(zip(model.source_names, np.round(model.weights, 2))))


# ----------------------------------------------------------------------------
# scoring
# ----------------------------------------------------------------------------
def score_line(model, line, settings, kd=0.9, alpha=1.5, n_cells=400, seed=0, verbose=True):
    """
    settings: {name: kwargs for model._predicted_cells}, e.g.
              {"gain 2.5": {"gain": 2.5},
               "N200 top2.5": {"gain": 1.0, "n_call": 200, "gain_top": 2.5}}
    Returns (scaled, raw) like proxy_leaderboard: rows = settings + "baseline" +
    "replicate"; scaled has 0 = mean-vector baseline, 1 = replicate. Column n_sig
    is the median number of DE calls per target; the "real reference" row holds
    the real cells' own median, the call count to aim for.
    """
    rng = np.random.default_rng(seed)
    targets = line.targets
    target_gis = [model.gene_index.get(t, -1) for t in targets]

    t0 = time.time()
    scorer = ProxyScorer(line.ctrl.X)
    if verbose:
        print(f"scorer: {len(scorer.tester.genes)} DE-tested genes, {line.ctrl.n_obs} controls "
              f"({time.time() - t0:.0f}s)")

    # reference = the real cells; replicate = two disjoint halves of them
    t0 = time.time()
    ref, rep_a, rep_b, means = [], [], [], []
    for t in targets:
        X = line.cells[t]
        ref.append(scorer.summarize(X))
        perm = rng.permutation(X.shape[0])
        h = X.shape[0] // 2
        rep_a.append(scorer.summarize(X[perm[:h]]))
        rep_b.append(scorer.summarize(X[perm[h:2 * h]]))
        means.append(np.asarray(X.mean(axis=0)).ravel())
    if verbose:
        print(f"reference + replicate: {time.time() - t0:.0f}s, "
              f"median real DE calls {np.median([int(r['sig'].sum()) for r in ref]):.0f}")

    # baseline: the mean per-construct count vector, as n_cells identical cells
    base_cells = scorer.summarize(np.tile(np.mean(means, axis=0), (n_cells, 1)))
    raw = {"baseline": scorer.score([base_cells] * len(targets), ref, target_gis),
           "replicate": scorer.score(rep_a, rep_b, target_gis)}

    for name, kw in settings.items():
        t0 = time.time()
        sub = [scorer.summarize(model._predicted_cells(line.ctrl, t, n_cells, rng, kd, alpha,
                                                       rounding="stochastic", **kw))
               for t in targets]
        raw[name] = scorer.score(sub, ref, target_gis)
        if verbose:
            print(f"{name}: {time.time() - t0:.0f}s")

    raw = pd.DataFrame(raw).T
    scaled = pd.DataFrame({k: scale(raw.loc[k], raw.loc["baseline"], raw.loc["replicate"])
                           for k in raw.index}).T
    scaled["n_sig"] = raw["n_sig"]
    scaled.loc["real reference", "n_sig"] = float(np.median([int(r["sig"].sum()) for r in ref]))
    return scaled, raw


# ----------------------------------------------------------------------------
# sign diagnostic: where do wrong-direction DE calls come from?
# ----------------------------------------------------------------------------
def sign_diagnostic(model, line, setting, kd=0.9, alpha=1.5, n_cells=400, seed=0, verbose=True):
    """
    For one setting (kwargs for model._predicted_cells), take each target's
    predicted DE calls and ask whose sign was wrong. The predicted log2 effect is
    E = b + d, with b = the shared context mean (model.b_c, from the training lines)
    and d = the target-specific part (SLIM embedding and/or direct transfer).
    Signs are compared with the real cells' log2FC (CPM means vs the same controls).

    "shift" is the line's TRUE shared response: the lfc of the mean real perturbed
    cell (what the mean-vector baseline predicts). It is an oracle the model can't
    see at test time; it shows how good a perfect b would be.

    Per-target columns (fractions over that target's calls, target gene excluded):
      n_calls, n_real
      acc_model    sign(predicted cells' lfc) == sign(real lfc)   (= raw FID when n_calls >= n_real)
      acc_E        sign(E) == real        sign the model intended
      realized     sign(predicted lfc) == sign(E)   (< 1: sampling/CPM normalisation flipped it)
      acc_b, acc_d sign(b) / sign(d) == real   (b = model.b_c; d = E - b, everything else)
      acc_ddir     sign(lam * d_direct) == real, on genes where it is nonzero (covered targets)
      acc_dslim    sign((1 - lam) * SLIM part) == real
                   (before 2026-11-03 "b" here wrongly included lam * d_direct for TransferModel)
      acc_shift    sign(true shift) == real          (oracle b)
      b_vs_shift   sign(b) == sign(true shift)       (is the transferred b right for this line?)
      d_vs_resid   sign(d) == sign(real - shift)     (is d right about the target-specific part?)
      b_dom        fraction of calls with |b| > |d|
      acc_bdom, acc_ddom   acc_model within b-dominated / d-dominated calls
      lib_ratio    median predicted UMI/cell / median control UMI/cell
    Also returns the same accuracies over the REAL significant genes (prefix real_),
    i.e. what a caller that picked exactly the right genes would get from each sign.
    Returns (per_target DataFrame, summary Series of per-target means).
    """
    rng = np.random.default_rng(seed)
    t0 = time.time()
    scorer = ProxyScorer(line.ctrl.X)
    tg = scorer.tester.genes                                # global indices of DE-tested genes
    ctrl_lib = np.median(np.asarray(line.ctrl.X.sum(axis=1)).ravel())

    means = np.mean([np.asarray(m.mean(axis=0)).ravel() for m in line.cells.values()], axis=0)
    shift = scorer.summarize(np.tile(means, (n_cells, 1)))["lfc"]

    sgn = np.sign
    frac = lambda x: float(np.mean(x)) if len(x) else np.nan
    rows = []
    for t in line.targets:
        real = scorer.summarize(line.cells[t])
        X = model._predicted_cells(line.ctrl, t, n_cells, rng, kd, alpha,
                                   rounding="stochastic", **setting)
        pred = scorer.summarize(X)
        # b straight from b_c: predict_effects(alpha=0) is NOT b for a TransferModel,
        # because it still adds the direct-transfer part lam * d_direct.
        row_g = model.row_in_G[tg]
        b = np.where(row_g >= 0, model.b_c[np.maximum(row_g, 0)], 0.0) / np.log(2)
        E0 = model.predict_effects([t], None, alpha=0.0)[:, 0][tg].astype(np.float64)
        E = model.predict_effects([t], None, alpha=alpha)[:, 0][tg].astype(np.float64)
        d = E - b                  # whole target-specific part
        d_dir = E0 - b             # lam * d_direct (0 for uncovered targets / plain Model)
        d_slim = E - E0            # (1 - lam) * alpha * SLIM embedding part

        keep = np.ones(len(tg), bool)
        gi = model.gene_index.get(t, -1)
        keep[tg == gi] = False
        adj = (np.isfinite(real["lfc"]) & (real["lfc"] != 0)) | np.isinf(real["lfc"])
        r = sgn(real["lfc"])
        calls = pred["sig"] & adj & keep
        rsig = real["sig"] & adj & keep
        resid = sgn(real["lfc"] - shift)

        c, bd = calls, np.abs(b) > np.abs(d)
        row = {"target": t, "n_calls": int(c.sum()), "n_real": int(rsig.sum()),
               "acc_model": frac(sgn(pred["lfc"][c]) == r[c]),
               "acc_E": frac(sgn(E[c]) == r[c]),
               "realized": frac(sgn(pred["lfc"][c]) == sgn(E[c])),
               "acc_b": frac(sgn(b[c]) == r[c]),
               "acc_d": frac(sgn(d[c]) == r[c]),
               "acc_ddir": frac(sgn(d_dir[c & (d_dir != 0)]) == r[c & (d_dir != 0)]),
               "acc_dslim": frac(sgn(d_slim[c & (d_slim != 0)]) == r[c & (d_slim != 0)]),
               "acc_shift": frac(sgn(shift[c]) == r[c]),
               "b_vs_shift": frac(sgn(b[c]) == sgn(shift[c])),
               "d_vs_resid": frac(sgn(d[c]) == resid[c]),
               "b_dom": frac(bd[c]),
               "acc_bdom": frac(sgn(pred["lfc"][c & bd]) == r[c & bd]),
               "acc_ddom": frac(sgn(pred["lfc"][c & ~bd]) == r[c & ~bd]),
               "lib_ratio": float(np.median(np.asarray(X.sum(axis=1)).ravel()) / ctrl_lib)}
        s = rsig
        row.update({"real_acc_E": frac(sgn(E[s]) == r[s]),
                    "real_acc_b": frac(sgn(b[s]) == r[s]),
                    "real_acc_d": frac(sgn(d[s]) == r[s]),
                    "real_acc_ddir": frac(sgn(d_dir[s & (d_dir != 0)]) == r[s & (d_dir != 0)]),
                    "real_acc_dslim": frac(sgn(d_slim[s & (d_slim != 0)]) == r[s & (d_slim != 0)]),
                    "real_ddir_vs_resid": frac(sgn(d_dir[s & (d_dir != 0)]) == resid[s & (d_dir != 0)]),
                    "real_acc_shift": frac(sgn(shift[s]) == r[s]),
                    "real_d_vs_resid": frac(sgn(d[s]) == resid[s])})
        rows.append(row)

    df = pd.DataFrame(rows).set_index("target")
    summary = df.mean(numeric_only=True)
    summary["n_calls"], summary["n_real"] = df["n_calls"].median(), df["n_real"].median()
    if verbose:
        print(f"sign_diagnostic: {len(df)} targets ({time.time() - t0:.0f}s)")
    return df, summary


# ----------------------------------------------------------------------------
# d vs b check: is direct transfer's d just "minus the average response" for
# weak targets?
# ----------------------------------------------------------------------------
def d_vs_b_check(model, targets, diag=None, n_bins=4):
    """
    Hypothesis: d_direct(t) = y_K562(t) - mean K562 response. For a target with
    little effect in K562, y_K562(t) ~ 0, so d ~ -(mean response), which points
    against b. Then d cancels part of b and its signs come out below chance
    (HCT116 sign_diagnostic: real_acc_d 0.42).

    Needs only the fitted model (call set_context / set_line_context first, so
    b_c and the line weights match the context), so it also runs on the VCC test
    panel, which has no real cells.

    targets: target genes to check; uncovered ones (no K562 measurement) are skipped.
    diag:    optional per-target DataFrame from sign_diagnostic on a real line
             (e.g. HCT116); its acc_d / real_acc_d / real_acc_b get joined on.
    Per-target columns:
      strength     ||y_K562(t)|| / ||mean K562 response||, on genes K562 measured
                   (~1 or below = the target's own effect is no bigger than the average one)
      n_cells, lam lam = n / (n + transfer_c), the weight d_direct gets
      corr_d_b     Pearson corr of d_direct and b_c over genes
      cos_d_negm   cosine of d_direct and -(mean K562 response); ~1 = d is just -mean
    Returns (per_target DataFrame, per-strength-bin summary DataFrame).
    """
    if "K562" not in model._source_code:
        raise ValueError("model has no K562 source line")
    k = model._source_code["K562"]
    g = np.arange(model.G.shape[0])                         # every row of G
    meas = model._measured[:, k]
    mean_k = model.source_means[:, k]
    b = model.b_c
    c = model.transfer_c

    def corr(x, y):
        x, y = x - x.mean(), y - y.mean()
        den = np.sqrt((x @ x) * (y @ y))
        return float(x @ y / den) if den > 0 else np.nan

    def cos(x, y):
        den = np.linalg.norm(x) * np.linalg.norm(y)
        return float(x @ y / den) if den > 0 else np.nan

    rows = []
    for t in targets:
        cols = [(j, n) for j, kk, n in model._direct_cols.get(t, []) if kk == k]
        if not cols:
            continue
        got = model._direct(t, g)
        if got is None:
            continue
        d, n_eff, has = got
        m = meas & has
        y = np.mean([model._Y[:, j] for j, _ in cols], axis=0)
        rows.append({"target": t,
                     "strength": float(np.linalg.norm(y[m]) / np.linalg.norm(mean_k[m])),
                     "n_cells": n_eff,
                     "lam": n_eff / (n_eff + c) if c is not None else 0.0,
                     "corr_d_b": corr(d[m], b[m]),
                     "cos_d_negm": cos(d[m], -mean_k[m])})
    df = pd.DataFrame(rows).set_index("target")
    if diag is not None:
        df = df.join(diag[[col for col in ("acc_d", "real_acc_d", "real_acc_b") if col in diag]])

    df["strength_bin"] = pd.qcut(df["strength"], n_bins, labels=[f"Q{i + 1}" for i in range(n_bins)])
    agg = {"strength": ("strength", "median"), "n_targets": ("strength", "size"),
           "lam": ("lam", "median"), "corr_d_b": ("corr_d_b", "mean"),
           "cos_d_negm": ("cos_d_negm", "mean")}
    for col in ("acc_d", "real_acc_d", "real_acc_b"):
        if col in df:
            agg[col] = (col, "mean")
    summary = df.groupby("strength_bin", observed=True).agg(**agg)
    return df, summary
