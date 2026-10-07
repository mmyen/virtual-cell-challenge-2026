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

        # set by _load_dataset_effects(); one entry per (path, effect_space),
        # so any subset of datasets can be re-pooled without re-reading the
        # h5ad files (what leave-one-cell-line-out validation needs)
        self._dataset_cache = {}
        self._self_ratio_cache = {}  # path -> target gene's own expr / ctrl

        # set by _preprocess_training_data(); cached so fit() can be
        # re-run at different K/lam without re-pooling
        self._Y = None  # (N_genes, N_perts) sparse, pooled effect matrix
        self._pert_genes = None  # length N_perts, gene symbol per column of _Y
        self._Y_key = None  # (paths, effect_space, excluded perts) _Y was built from

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
    def _load_dataset_effects(self, path, effect_space="log2"):
        """
        Read ONE reference file and return a dict with:
            "effects":   DataFrame, index = target gene symbol (one row per
                         perturbed gene, sgRNA constructs averaged),
                         columns = global gene index (only the genes this
                         dataset measured), values = relative effect in
                         `effect_space` (see _preprocess_training_data).
            "ctrl_mean": Series indexed like effects.columns -- this
                         dataset's mean control expression per gene.
            "self_ratio": Series, index = target symbol -- the target
                         gene's own expression / control (0.1 = a 90%
                         knockdown). Only for targets the dataset measured.

        Cached per (path, effect_space): leave-one-cell-line-out validation
        re-pools many different subsets of datasets, and re-reading the
        h5ad files each time would dominate the runtime.

        NOTE: this assumes each file's .X is already control-normalized
        per dataset (true for all four current files -- values sit in
        the ~0-1.6 / ~0-5.5 range, not raw counts).
        """
        if effect_space not in ("log2", "ratio"):
            raise ValueError(f"effect_space must be 'log2' or 'ratio', got {effect_space!r}")
        key = (path, effect_space)
        if key in self._dataset_cache:
            return self._dataset_cache[key]

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
        ).to_numpy().astype(bool)
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
        target_sym = np.asarray(target_ensembl.map(sym_by_ensembl))

        X = a.X[:, local_cols]
        X = X.toarray() if sp.issparse(X) else np.asarray(X)
        if effect_space == "log2":
            pseudocount = 0.1
            rel_effect = np.log2((X + pseudocount) / (ctrl_mean + pseudocount))
        else:
            rel_effect = X / ctrl_mean - 1.0

        keep = (~ctrl_mask) & pd.Series(target_sym).isin(self.gene_index).to_numpy()
        df = pd.DataFrame(rel_effect[keep].astype(np.float32), columns=global_rows)
        df["target_sym"] = target_sym[keep]
        pooled = df.groupby("target_sym").mean()  # average replicate constructs

        # target gene's own remaining expression, for knockdown_from_training()
        local_pos = {g: j for j, g in enumerate(global_rows)}
        rows = [(i, local_pos[self.gene_index[s]]) for i, s in
                zip(np.where(keep)[0], target_sym[keep]) if self.gene_index[s] in local_pos]
        r_i = np.array([i for i, _ in rows])
        r_j = np.array([j for _, j in rows])
        self_ratio = (
            pd.Series(X[r_i, r_j] / ctrl_mean[r_j], index=target_sym[r_i])
            .groupby(level=0).mean()
        )

        out = {
            "effects": pooled,
            "ctrl_mean": pd.Series(ctrl_mean, index=global_rows),
            "self_ratio": self_ratio,
        }
        self._dataset_cache[key] = out
        self._self_ratio_cache[path] = self_ratio
        return out

    def _preprocess_training_data(self, dataset_paths, effect_space="log2",
                                  exclude_perts=()):
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
        exclude_perts: gene symbols whose perturbations are dropped from
                      every dataset (they stay in Y as *response* genes).
                      Used by validation to simulate targets that were
                      never perturbed in training.
        """
        exclude = set(exclude_perts)
        row_idx, col_pert, values = [], [], []
        pert_order, pert_col = [], {}

        def col_for(p_sym):
            if p_sym not in pert_col:
                pert_col[p_sym] = len(pert_order)
                pert_order.append(p_sym)
            return pert_col[p_sym]

        for path in dataset_paths:
            pooled = self._load_dataset_effects(path, effect_space)["effects"]
            pooled = pooled[~pooled.index.isin(exclude)]
            global_rows = pooled.columns.to_numpy()

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
        self._Y_key = (tuple(dataset_paths), effect_space, frozenset(exclude))
        return self._Y

    def knockdown_from_training(self, dataset_paths=DEFAULT_TRAINING_PATHS,
                                weighting="n_perts"):
        """
        Knockdown fraction to force on each target gene in predict():
        1 - (weighted average of each dataset's median remaining target
        expression). weighting="n_perts" weights each dataset by how many
        perturbations it measured (K562 dominates, ~0.84); "equal" gives
        each cell line the same say (~0.88).
        """
        medians, n = [], []
        for path in dataset_paths:
            if path not in self._self_ratio_cache:
                self._load_dataset_effects(path)
            r = self._self_ratio_cache[path].dropna()
            medians.append(r.median())
            n.append(len(r))
        if weighting == "n_perts":
            weights = np.asarray(n, dtype=float)
        elif weighting == "equal":
            weights = np.ones(len(n))
        else:
            raise ValueError(f"weighting must be 'n_perts' or 'equal', got {weighting!r}")
        return float(1.0 - np.average(medians, weights=weights))

    # ------------------------------------------------------------------
    # 2. fit: PCA embedding (G, P) + ridge-regularized W, following
    #    Ahlmann-Eltze et al. 2025 (Methods).
    # ------------------------------------------------------------------
    def fit(self, dataset_paths=DEFAULT_TRAINING_PATHS, K=10, lam=0.1,
            effect_space="log2", exclude_perts=(), force_reprocess=False):
        """
        K and lam are cheap to sweep: the pooled matrix
        (_preprocess_training_data) is only rebuilt if it hasn't been built
        yet, force_reprocess=True, or dataset_paths / effect_space /
        exclude_perts differ from what the cached matrix was built with.
        Each h5ad file is read at most once per effect_space either way
        (see _load_dataset_effects), so switching dataset subsets for
        validation only costs the re-pooling. If the pooled matrix, K, lam
        and effect_space are all unchanged, the existing fit is kept, so
        sweeping a predict-time setting like alpha costs no refits.

        effect_space: "log2" (default, recommended) or "ratio" (the
        original, outlier-prone version -- kept so you can flip back and
        compare the two). See _preprocess_training_data for what each
        means; predict() reads self.effect_space to reconstruct
        correctly, so there's nothing else to keep in sync by hand.
        exclude_perts: perturbations to leave out of training (validation
        only; see _preprocess_training_data).
        """
        key = (tuple(dataset_paths), effect_space, frozenset(exclude_perts))
        if self._Y is None or force_reprocess or self._Y_key != key:
            self._preprocess_training_data(
                dataset_paths, effect_space=effect_space, exclude_perts=exclude_perts
            )
        elif self.G is not None and (self.K, self.lam, self.effect_space) == (K, lam, effect_space):
            return self  # same data and hyperparameters: G, W, b are already fit
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
    def predict_effects(self, target_genes, knockdown_fraction=0.9, gene_rows=None,
                        alpha=1.0):
        """
        Predicted relative effect (in self.effect_space) of knocking down
        each target -- the same vector predict() multiplies control cells
        by, before any cells are involved. Validation scores these directly.

        target_genes: symbols to predict.
        knockdown_fraction: scalar, or one value per target. Each target's
                       own gene is forced to (1 - knockdown_fraction) of
                       control; see predict() for why. None = no override:
                       the target is predicted by G W G[p] like any other
                       gene, as in Ahlmann-Eltze et al.
        gene_rows: optional global gene indices to restrict the output to
                   (saves memory when scoring thousands of perturbations).
        alpha: shrinkage on the target-specific term, E = b + alpha * G W G[p].
               1 = the fitted model, 0 = the training mean effect b for every
               target. LOCO validation showed G W G[p] adds a little
               target-specific signal but often gets the sign of the
               top changed genes wrong, so a value in between may do best.
        Returns (len(gene_rows) or N_genes, len(target_genes)) float32.
        """
        if self.G is None or self.W is None:
            raise RuntimeError("call fit() before predict_effects()")
        rows = np.arange(self.N_genes) if gene_rows is None else np.asarray(gene_rows)
        p_idx = np.array([self.gene_index[s] for s in target_genes])
        E = self.b[rows, None] + alpha * (self.G[rows] @ (self.W @ self.G[p_idx].T))

        if knockdown_fraction is None:
            return E.astype(np.float32)

        kd = np.broadcast_to(np.asarray(knockdown_fraction, dtype=float), p_idx.shape)
        self_value = np.log2(1.0 - kd) if self.effect_space == "log2" else -kd
        pos = {g: i for i, g in enumerate(rows)}
        for j, g in enumerate(p_idx):
            if g in pos:
                E[pos[g], j] = self_value[j]
        return E.astype(np.float32)

    def predict(self, control_adata, target_genes, cells_per_pert=400, rng=None,
                knockdown_fraction=0.9, alpha=1.0):
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
                       column. None = no override (see predict_effects).
                       NOTE: the 2026 metrics exclude each perturbation's
                       target gene from scoring (PDS excludes all panel
                       targets), so this value only matters for realism and
                       via per-cell library-size normalization, not directly.
        alpha: shrinkage on the target-specific term, see predict_effects.
        Returns a per-cell AnnData for just this context.
        """
        rng = np.random.default_rng() if rng is None else rng
        assert control_adata.n_vars == self.N_genes

        blocks, targets = [], []
        for p_sym in target_genes:
            blocks.append(self._predicted_cells(control_adata, p_sym, cells_per_pert, rng,
                                                knockdown_fraction, alpha))
            targets.extend([p_sym] * cells_per_pert)

        X = sp.vstack(blocks).tocsr()
        obs = pd.DataFrame(
            {
                "target_gene": targets,
                "context": control_adata.obs["context"].iloc[0],
            }
        )
        return ad.AnnData(X=X, obs=obs, var=control_adata.var.copy())

    def _predicted_cells(self, control_adata, p_sym, cells_per_pert, rng,
                         knockdown_fraction, alpha, gain=1.0, rounding="nearest"):
        """
        One perturbation's predicted cells: cells_per_pert real control cells
        (sampled without replacement), each gene scaled by the predicted
        effect and rounded to counts. Returns a (cells_per_pert, N_genes)
        float32 CSR block. Shared by predict() and write_submission() so the
        in-memory and streamed outputs are generated identically.

        gain: multiplies every non-target gene's predicted effect before it
              is applied to cells (the target keeps its forced knockdown).
              Only affects the per-cell output, not predict_effects(): the
              DE metrics (FID/JAC/REACH) need effects large enough for a
              Wilcoxon test to detect, and the fitted effects are shrunken
              (ridge + averaging over screens). See test_fid_fix.ipynb.
        rounding: "nearest" -- np.round; deterministic, so any multiplier in
              (0.5, 1.5) leaves a count of 1 unchanged, which erases most
              predicted effects on low-count genes. "stochastic" -- round
              up with probability equal to the fractional part, so every
              gene's expected count is cells * multiplier exactly.
        """
        effect = self.predict_effects([p_sym], knockdown_fraction, alpha=alpha)[:, 0]  # (N_genes,)
        if gain != 1.0:
            p_gi = self.gene_index[p_sym]
            target_effect = effect[p_gi]
            effect = effect * gain
            effect[p_gi] = target_effect

        idx = rng.choice(control_adata.n_obs, size=cells_per_pert, replace=False)
        cells = control_adata.X[idx]
        cells = cells.toarray() if sp.issparse(cells) else np.asarray(cells)

        multiplier = 2.0 ** effect if self.effect_space == "log2" else np.clip(1.0 + effect, 0, None)

        scaled = cells * multiplier
        if rounding == "nearest":
            predicted = np.round(scaled)
        elif rounding == "stochastic":
            predicted = np.floor(scaled)
            predicted += rng.random(scaled.shape) < (scaled - predicted)
        else:
            raise ValueError(f"rounding must be 'nearest' or 'stochastic', got {rounding!r}")
        predicted = np.clip(predicted, 0, None).astype(np.float32)
        return sp.csr_matrix(predicted)  # sparsify per-block, not at the end

    def write_submission(self, path, context_controls, target_genes_by_context,
                         knockdown_fraction=0.9, alpha=1.0, cells_per_pert=400,
                         rng=None, compression="gzip", gain=1.0, rounding="nearest"):
        """
        Same output as predict_submission(...).write_h5ad(path), but streamed
        to disk one perturbation at a time instead of built in memory.

        The full submission is 3 contexts x 300 targets x 400 cells, each a
        rescaled real control cell with ~6k nonzero genes: ~2 billion
        nonzeros, ~17 GB as an in-memory CSR matrix, and ~2x that at peak
        while predict()/ad.concat copy it. Here only one 400-cell block is
        in memory at a time, so peak RAM is roughly the fitted model plus
        the loaded control cells.

        How: write obs/var with anndata (X=None), then append X's CSR
        arrays (data, indices, indptr) into resizable HDF5 datasets in the
        layout anndata reads back as a csr_matrix.

        Arguments as in predict_submission(); rng is shared across contexts
        (None = fresh, unseeded, like predict()). gain and rounding: see
        _predicted_cells (defaults reproduce the earlier submissions).
        """
        import h5py

        rng = np.random.default_rng() if rng is None else rng
        contexts = list(context_controls)
        var_names = context_controls[contexts[0]].var_names
        for ctx in contexts:
            assert context_controls[ctx].n_vars == self.N_genes
            assert context_controls[ctx].var_names.equals(var_names), f"context {ctx}: gene order differs"

        # obs is known up front: every target gets exactly cells_per_pert cells
        obs = pd.concat([
            pd.DataFrame({
                "target_gene": np.repeat(target_genes_by_context[ctx], cells_per_pert),
                "context": context_controls[ctx].obs["context"].iloc[0],
            })
            for ctx in contexts
        ], ignore_index=True)
        obs.index = obs.index.astype(str)
        obs = obs.astype("category")
        n_obs = len(obs)

        # var as predict_submission()'s ad.concat leaves it: gene names only
        ad.AnnData(obs=obs, var=pd.DataFrame(index=var_names)).write_h5ad(path)

        indptr = np.zeros(n_obs + 1, dtype=np.int64)  # int64: total nnz is close to 2**31
        with h5py.File(path, "a") as f:
            if "X" in f:
                del f["X"]
            X = f.create_group("X")
            X.attrs["encoding-type"] = "csr_matrix"
            X.attrs["encoding-version"] = "0.1.0"
            X.attrs["shape"] = np.array([n_obs, self.N_genes])
            chunk = 1 << 20
            data = X.create_dataset("data", shape=(0,), maxshape=(None,), dtype=np.float32,
                                    chunks=(chunk,), compression=compression)
            indices = X.create_dataset("indices", shape=(0,), maxshape=(None,), dtype=np.int32,
                                       chunks=(chunk,), compression=compression)

            row, nnz = 0, 0
            for ctx in contexts:
                for p_sym in target_genes_by_context[ctx]:
                    block = self._predicted_cells(context_controls[ctx], p_sym, cells_per_pert,
                                                  rng, knockdown_fraction, alpha,
                                                  gain=gain, rounding=rounding)
                    block.sort_indices()
                    n = block.nnz
                    data.resize((nnz + n,))
                    data[nnz:] = block.data
                    indices.resize((nnz + n,))
                    indices[nnz:] = block.indices
                    indptr[row + 1: row + cells_per_pert + 1] = nnz + block.indptr[1:]
                    row, nnz = row + cells_per_pert, nnz + n
            assert row == n_obs
            X.create_dataset("indptr", data=indptr, compression=compression)
        return path

    def predict_submission(self, context_controls, target_genes_by_context,
                           knockdown_fraction=0.9, alpha=1.0):
        """
        context_controls: {"A": adata_A, "B": adata_B, "C": adata_C}
        target_genes_by_context: {"A": [...300 genes...], ...} (or one
                                  shared list reused for every context)
        knockdown_fraction: passed to predict() (e.g. knockdown_from_training()).
        alpha: passed to predict(); see predict_effects.
        Returns the full 3-context AnnData ready for vcc prep.
        Builds everything in memory (~33 GB peak for the full panel); use
        write_submission() to stream it to disk instead.
        """
        parts = [
            self.predict(control_adata, target_genes_by_context[ctx],
                         knockdown_fraction=knockdown_fraction, alpha=alpha)
            for ctx, control_adata in context_controls.items()
        ]
        return ad.concat(parts, join="outer")