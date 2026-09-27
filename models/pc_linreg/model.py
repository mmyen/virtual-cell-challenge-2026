import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import svds


class Model:
    """
    Linear model from Ahlmann-Eltze, Huber & Anders, "Deep-learning-based
    gene perturbation effect prediction does not yet outperform simple
    linear baselines" (Nat. Methods, 2025): predicted expression
    Y_hat = G @ W @ P.T + b, where G/P are PCA-derived low-rank gene/
    perturbation embeddings and W is fit by ridge-regularized least
    squares (Methods, eqs. for W).

    Adapted for our multi-dataset, zero-shot setting: instead of one
    clean training matrix, we pool log2 fold-changes (vs. each dataset's
    own control) across all of data/replogle_nadig/, so values are
    comparable across cell lines before the PCA/ridge fit -- the paper
    works from a single dataset's raw (log-transformed) profile matrix,
    which we don't have. Log space, not raw (X/ctrl - 1): the latter
    blows up for lowly-expressed genes and a handful of huge outliers
    were dominating the SVD below.

    P is literally G's rows for genes that were perturbed in training
    (per the paper), not an independent table -- this is what lets
    predict() produce an effect for target genes that were never
    directly perturbed in any reference dataset, as long as they were
    measured as a response gene somewhere.
    """

    DEFAULT_TRAINING_PATHS = (
        "../../data/replogle_nadig/K562_gwps_raw_bulk_01.h5ad",
        "../../data/replogle_nadig/rpe1_raw_bulk_01.h5ad",
        "../../data/replogle_nadig/nadig_hepg2_pseudobulk.h5ad",
        "../../data/replogle_nadig/nadig_jurkat_pseudobulk.h5ad",
    )

    def __init__(self, gene_names):
        """
        gene_names: the 18,533 challenge gene symbols, in the same order
                    as data/controls/gene_names.csv / every context's
                    var_names. Fixes the index space G, P, and b live in.
        """
        self.gene_names = np.asarray(gene_names)
        self.gene_index = {g: i for i, g in enumerate(self.gene_names)}
        self.N_genes = len(self.gene_names)

        # set by _preprocess_training_data(); cached so fit() can be
        # re-run at different K/lam without re-reading the h5ad files
        self._Y = None  # (N_genes, N_perts) sparse, pooled effect matrix
        self._pert_genes = None  # length N_perts, gene symbol per column of _Y
        self._Y_effect_space = None  # "log2" or "ratio" -- which _Y was built with

        # set by fit()
        self.K = None
        self.lam = None
        self.effect_space = None  # "log2" or "ratio", see fit()
        self.G = None  # (N_genes, K)
        self.W = None  # (K, K)
        self.b = None  # (N_genes,)

    # ------------------------------------------------------------------
    # 1. preprocessing: replogle_nadig -> one partially-observed
    #    (N_genes, N_perturbations) relative-effect matrix, in the
    #    challenge's gene index space. Cached on self._Y.
    # ------------------------------------------------------------------
    def _preprocess_training_data(self, dataset_paths, effect_space="log2"):
        """
        Pool the reference Perturb-seq/pseudobulk files into a sparse
        (N_genes, N_perts) matrix Y where:
            Y[g, p] ~ mean effect on gene g's expression when gene p was
                      perturbed, averaged over every dataset and sgRNA
                      construct that tested p.
        Unobserved (g, p) pairs are left at 0 -- that's "missing"
        (no training signal), not "no effect"; fit() doesn't distinguish
        the two, which is a simplification worth revisiting later.

        effect_space: "log2" (default) -- log2((X+eps)/(ctrl+eps)),
                      bounded, matches the paper's log-transformed space.
                      "ratio" -- the original X/ctrl - 1 formulation.
                      Kept as a toggle for comparison: it's unbounded and
                      a handful of near-zero-ctrl_mean genes blow it up to
                      ~1000x, which dominates the SVD in fit() -- see the
                      class docstring. "log2" is the one worth using;
                      "ratio" exists to reproduce/compare against the
                      earlier version of this model.

        NOTE: this assumes each file's .X is already control-normalized
        per dataset (true for all four current files -- values sit in
        the ~0-1.6 / ~0-5.5 range, not raw counts).
        """
        if effect_space not in ("log2", "ratio"):
            raise ValueError(f"effect_space must be 'log2' or 'ratio', got {effect_space!r}")

        row_idx, col_pert, values = [], [], []
        pert_order, pert_col = [], {}

        def col_for(p_sym):
            if p_sym not in pert_col:
                pert_col[p_sym] = len(pert_order)
                pert_order.append(p_sym)
            return pert_col[p_sym]

        for path in dataset_paths:
            a = ad.read_h5ad(path)

            # keep only symbols that are unambiguous within this dataset
            # and present in the challenge's gene panel
            unambiguous = ~a.var["gene_name"].duplicated(keep=False)
            usable = unambiguous & a.var["gene_name"].isin(self.gene_index)
            local_cols = np.where(usable.to_numpy())[0]
            global_rows = np.array(
                [self.gene_index[s] for s in a.var["gene_name"].to_numpy()[local_cols]]
            )

            ctrl_mask = (
                a.obs["core_control"] if "core_control" in a.obs.columns
                else a.obs["is_control"]
            ).to_numpy()
            ctrl_mean = np.asarray(a.X[ctrl_mask][:, local_cols].mean(axis=0)).ravel()
            ctrl_mean = np.clip(ctrl_mean, 1e-6, None)

            if "target_ensembl" in a.obs.columns:
                target_ensembl = a.obs["target_ensembl"]
            else:
                target_ensembl = a.obs_names.str.extract(r"(ENSG\d+)$")[0]
            sym_by_ensembl = (
                a.var.set_index("ensembl_id")["gene_name"]
                if "ensembl_id" in a.var.columns
                else a.var["gene_name"]  # var already indexed by Ensembl id
            )
            target_sym = target_ensembl.map(sym_by_ensembl)

            X = a.X[:, local_cols]
            X = X.toarray() if sp.issparse(X) else np.asarray(X)
            if effect_space == "log2":
                pseudocount = 0.1
                rel_effect = np.log2((X + pseudocount) / (ctrl_mean + pseudocount))
            else:
                rel_effect = X / ctrl_mean - 1.0

            df = pd.DataFrame(rel_effect)
            df["target_sym"] = target_sym.to_numpy()
            df = df[(~ctrl_mask) & df["target_sym"].isin(self.gene_index)]
            pooled = df.groupby("target_sym").mean()  # average replicate constructs

            p_cols = np.array([col_for(s) for s in pooled.index])
            block = pooled.to_numpy()  # (n_perts_in_dataset, n_matched_cols)

            r, c = np.meshgrid(global_rows, p_cols, indexing="ij")
            row_idx.append(r.ravel())
            col_pert.append(c.ravel())
            values.append(block.T.ravel())

        row_idx = np.concatenate(row_idx)
        col_pert = np.concatenate(col_pert)
        values = np.concatenate(values).astype(np.float32)
        n_perts = len(pert_order)

        sums = sp.coo_matrix(
            (values, (row_idx, col_pert)), shape=(self.N_genes, n_perts)
        ).tocsr()
        counts = sp.coo_matrix(
            (np.ones_like(values), (row_idx, col_pert)),
            shape=(self.N_genes, n_perts),
        ).tocsr()
        counts.data = 1.0 / counts.data  # avg over datasets/constructs that agree

        self._Y = sums.multiply(counts).tocsr()
        self._pert_genes = np.array(pert_order)
        self._Y_effect_space = effect_space
        return self._Y

    # ------------------------------------------------------------------
    # 2. fit: PCA embedding (G, P) + ridge-regularized W, following
    #    Ahlmann-Eltze et al. 2025 (Methods).
    # ------------------------------------------------------------------
    def fit(self, dataset_paths=DEFAULT_TRAINING_PATHS, K=10, lam=0.1,
            effect_space="log2", force_reprocess=False):
        """
        K and lam are cheap to sweep: the expensive step
        (_preprocess_training_data, reading all of data/replogle_nadig/)
        only reruns if it hasn't been cached yet, force_reprocess=True,
        or effect_space differs from what the cached matrix was built
        with. Re-calling fit(K=...) with a new K (same effect_space)
        reuses the cached pooled matrix, so trying several K values
        across runs doesn't mean re-reading the h5ad files each time.

        effect_space: "log2" (default, recommended) or "ratio" (the
        original, outlier-prone version -- kept so you can flip back and
        compare the two). See _preprocess_training_data for what each
        means; predict() reads self.effect_space to reconstruct
        correctly, so there's nothing else to keep in sync by hand.
        """
        if self._Y is None or force_reprocess or self._Y_effect_space != effect_space:
            self._preprocess_training_data(dataset_paths, effect_space=effect_space)
        Y, pert_genes = self._Y, self._pert_genes
        n_perts = Y.shape[1]

        row_sums = np.asarray(Y.sum(axis=1)).ravel()
        row_nnz = Y.getnnz(axis=1)
        b = np.divide(
            row_sums, row_nnz,
            out=np.zeros(self.N_genes, dtype=np.float32),
            where=row_nnz > 0,
        )

        # G: top-K left singular vectors of Y (a PCA-style embedding of
        # each gene's pattern of relative effects across perturbations).
        # Unlike the paper, we do NOT mean-center by b before the SVD --
        # Y is already in "deviation from each gene's own control" units
        # (near 0 = no effect), and centering here would force a dense
        # (N_genes x n_perts) copy for comparatively little benefit. b is
        # still fit below and added back into every prediction.
        U, S, _ = svds(Y, k=K)
        order = np.argsort(-S)
        G = U[:, order]

        pert_rows = np.array([self.gene_index[s] for s in pert_genes])
        P = G[pert_rows]  # (n_perts, K) -- literally a subset of G's rows

        GtY = (Y.T @ G).T  # (K, n_perts), avoids densifying Y
        Gtb = G.T @ b  # (K,)
        GtY_centered = GtY - np.outer(Gtb, np.ones(n_perts))

        lhs = G.T @ G + lam * np.eye(K)
        mid = P.T @ P + lam * np.eye(K)
        rhs = GtY_centered @ P  # (K, K)
        A = np.linalg.solve(lhs, rhs)  # lhs^-1 @ rhs
        W = np.linalg.solve(mid.T, A.T).T  # A @ mid^-1

        self.K, self.lam = K, lam
        self.effect_space = effect_space
        self.G, self.W, self.b = G, W, b
        return self

    # ------------------------------------------------------------------
    # 3. predict: apply fitted G, W, b to real control cells for a context
    # ------------------------------------------------------------------
    def predict(self, control_adata, target_genes, cells_per_pert=400, rng=None,
                knockdown_fraction=0.9):
        """
        control_adata: one context's real control AnnData
                       (data/controls/context_{A,B,C}.h5ad).
        target_genes: the target_gene symbols to predict for this context.
        knockdown_fraction: the target gene's own predicted expression is
                       forced to (1 - knockdown_fraction) of control, same as
                       the target-scaling baseline, rather than trusting
                       G/W's learned/extrapolated self-effect -- diagnostics
                       showed that self-effect is weak and inconsistent (only
                       ~60% of sampled genes even came out negative), so this
                       recovers the one signal every CRISPRi perturbation is
                       reliably expected to show. G/W is still used for every
                       *other* gene -- that's the actual value pc_linreg adds
                       over the baseline, which only ever touches this one
                       column.
        Returns a per-cell AnnData for just this context.
        """
        if self.G is None or self.W is None:
            raise RuntimeError("call fit() before predict()")

        rng = np.random.default_rng() if rng is None else rng
        n_cells = control_adata.n_obs
        assert control_adata.n_vars == self.N_genes

        blocks, targets = [], []
        for p_sym in target_genes:
            p_gi = self.gene_index[p_sym]
            effect = self.b + self.G @ (self.W @ self.G[p_gi])  # (N_genes,)

            idx = rng.choice(n_cells, size=cells_per_pert, replace=False)
            cells = control_adata.X[idx]
            cells = cells.toarray() if sp.issparse(cells) else np.asarray(cells)

            if self.effect_space == "log2":
                effect[p_gi] = np.log2(1.0 - knockdown_fraction)
                multiplier = 2.0 ** effect
            else:
                effect[p_gi] = -knockdown_fraction
                multiplier = 1.0 + effect

            predicted = np.clip(np.round(cells * multiplier), 0, None).astype(np.float32)
            blocks.append(sp.csr_matrix(predicted))  # sparsify per-block, not at the end
            targets.extend([p_sym] * cells_per_pert)

        X = sp.vstack(blocks).tocsr()
        obs = pd.DataFrame(
            {
                "target_gene": targets,
                "context": control_adata.obs["context"].iloc[0],
            }
        )
        return ad.AnnData(X=X, obs=obs, var=control_adata.var.copy())

    def predict_submission(self, context_controls, target_genes_by_context):
        """
        context_controls: {"A": adata_A, "B": adata_B, "C": adata_C}
        target_genes_by_context: {"A": [...300 genes...], ...} (or one
                                  shared list reused for every context)
        Returns the full 3-context AnnData ready for vcc prep.
        """
        parts = [
            self.predict(control_adata, target_genes_by_context[ctx])
            for ctx, control_adata in context_controls.items()
        ]
        return ad.concat(parts, join="outer")
