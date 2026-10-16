"""
SLIM + direct transfer for covered targets.

A target that was perturbed in a training screen ("covered") has a measured
effect there. SLIM only uses it indirectly, through a K-component ridge map
from the STRING embedding, which smooths away most of what is specific to that
target. TransferModel uses the measurement directly:

    E(t) = b_c + d(t)
    d(t) = lam_t * d_direct(t) + (1 - lam_t) * d_slim(t)

    d_slim(t)   = alpha * G W p(t)                 (SLIM's target-specific part)
    d_direct(t) = sum_s w_s y_s(t) / sum_s w_s     (over source lines s that perturbed t)
    y_s(t)      = Y[:, t in s] - source_means[:, s]  (the effect minus line s's mean
                                                    response; the shared part is b_c's job)
    lam_t       = n_t / (n_t + c)                  (n_t = cells behind the measurement)

w_s are set_context's similarity weights, renormalised over the lines that
perturbed t and, per gene, over the lines that measured that gene. A gene no
source line measured falls back to SLIM. Uncovered targets have n_t = 0, so
they get SLIM unchanged, and transfer_c=None turns the transfer off entirely
(plain SLIM, for comparisons).

All 267 covered VCC validation targets were perturbed in K562 only, so the
default source_lines=("K562",). Validate with the same restriction: proxy and
LOCO targets that were also perturbed in RPE1/HepG2/Jurkat should still only be
transferred from K562.
"""

import numpy as np

from model import Model


class TransferModel(Model):

    def __init__(self, gene_names, embeddings_path, config, source_lines=("K562",),
                 transfer_c=200.0, transfer_scale=1.0):
        """
        source_lines:   training lines whose measured effects get transferred.
        transfer_c:     shrinkage, in cells: lam = n / (n + c). Smaller = trust the
                        measurement more. None = plain SLIM.
        transfer_scale: multiplies d_direct (measured effects aren't ridge-shrunk,
                        so they may want a different scale than alpha gives SLIM).
        Stored on the instance because write_submission() / _predicted_cells()
        call predict_effects() without these arguments.
        """
        super().__init__(gene_names, embeddings_path, config)
        self.source_lines = tuple(source_lines)
        self.transfer_c = transfer_c
        self.transfer_scale = transfer_scale

    def fit(self, Y, genes, perts, measured=None):
        """
        Same as Model.fit, plus the lookup for direct transfer.

        measured: {line: gene symbols that line measured}, e.g.
                  {n: dp.load_dataset(n).var_names for n in train_names}.
                  With build_Y(genes="union") an unmeasured gene is stored as 0,
                  which would read as "no effect"; the mask keeps those genes
                  out of d_direct. None = treat every gene as measured.
        """
        super().fit(Y, genes, perts)

        lines = perts["cell_line"].to_numpy()
        n_cells = perts["n_cells"].to_numpy().astype(float)
        self._source_code = {n: k for k, n in enumerate(self.source_names)}

        # target -> [(column of Y, line code, n_cells)], source lines only
        self._direct_cols = {}
        for j, (t, line) in enumerate(zip(self._targets, lines)):
            if line in self.source_lines:
                self._direct_cols.setdefault(t, []).append((j, self._source_code[line], n_cells[j]))

        # (genes in G, lines) measured mask
        self._measured = np.ones((len(genes), len(self.source_names)), bool)
        if measured is not None:
            for line, gset in measured.items():
                if line in self._source_code:
                    self._measured[:, self._source_code[line]] = np.isin(genes, list(gset))

        n_cov = sum(1 for t in set(self._targets) if t in self._direct_cols)
        print(f"  direct transfer from {self.source_lines}: {n_cov} targets have a measurement")

    def _direct(self, target, g_idx):
        """
        (d_direct over G rows g_idx in natural log, n_eff cells, per-gene "has data")
        for one target, or None if no source line perturbed it.
        """
        cols = self._direct_cols.get(target)
        if not cols:
            return None
        # similarity weights from set_context, uniform before it is called
        w_line = getattr(self, "weights", None)
        num = np.zeros(len(g_idx))
        den = np.zeros(len(g_idx))
        w_tot = n_eff = 0.0
        for j, k, n in cols:
            w = 1.0 if w_line is None else float(w_line[k])
            m = self._measured[g_idx, k]
            y = self._Y[g_idx, j] - self.source_means[g_idx, k]
            num += w * m * y
            den += w * m
            w_tot += w
            n_eff += w * n
        has = den > 0
        d = np.where(has, num / np.where(has, den, 1.0), 0.0)
        return d, (n_eff / w_tot if w_tot > 0 else 0.0), has

    def predict_effects(self, target_genes, knockdown_fraction=0.9, gene_rows=None, alpha=1.5,
                        transfer_c="default"):
        """Model.predict_effects with covered targets blended toward their measured effect (log2)."""
        c = self.transfer_c if transfer_c == "default" else transfer_c
        full = super().predict_effects(target_genes, None, gene_rows=gene_rows, alpha=alpha)
        if c is None:
            E = full
        else:
            base = super().predict_effects(target_genes, None, gene_rows=gene_rows, alpha=0.0)
            E = full.astype(np.float64)
            rows = np.arange(self.N_genes) if gene_rows is None else np.asarray(gene_rows)
            idx = self.row_in_G[rows]
            ok = idx >= 0
            for j, t in enumerate(target_genes):
                got = self._direct(t, idx[ok])
                if got is None:
                    continue  # uncovered: SLIM as is
                d_direct, n_eff, has = got
                lam = n_eff / (n_eff + c) if (n_eff + c) > 0 else 1.0
                lam_g = lam * has  # genes no source line measured stay SLIM
                d_slim = full[ok, j] - base[ok, j]
                d = lam_g * self.transfer_scale * d_direct / np.log(2) + (1 - lam_g) * d_slim
                E[ok, j] = base[ok, j] + d

        if knockdown_fraction is not None:  # same override as Model.predict_effects
            rows = np.arange(self.N_genes) if gene_rows is None else np.asarray(gene_rows)
            pos = {g: i for i, g in enumerate(rows)}
            kd = np.broadcast_to(np.asarray(knockdown_fraction, dtype=float), (len(target_genes),))
            for j, t in enumerate(target_genes):
                gi = self.gene_index.get(t)
                if gi in pos:
                    E[pos[gi], j] = np.log2(1 - kd[j])
        return np.asarray(E, dtype=np.float32)
