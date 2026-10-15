import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


class DataProcessor:
    """
    Turns the reference screens in data/replogle_nadig/ into the training
    matrix Y that SLIM (Hu et al., 2026; github.com/RasmussenLab/SLIM)
    builds its gene basis from:

        Y : (n_genes, n_perts) mean expression per perturbation
        b  = Y.mean(axis=1)                   (SLIM default, "train_mean")
        G  = PCA(K).fit_transform(Y - b)      (n_genes, K)

    SLIM fits one dataset at a time on normalize(1e4) -> log1p -> mean
    pseudobulk. Ours come in two different spaces, so load_dataset() puts
    every screen into that same log1p space first:

      * Replogle (K562, RPE1) *_raw_bulk_01: mean RAW UMI per cell, one
        row per sgRNA construct. Each row is normalized to target_sum and
        log1p'd. That is log(mean), not mean(log) like Nadig -- the
        single-cell data would be needed to match exactly. The file also
        only keeps ~8k genes, so the normalization total differs from
        Nadig's. Both are mostly a per-dataset offset that cancels when
        each screen is centered on its own control (build_Y center="ctrl").
      * Nadig (HepG2, Jurkat) *_pseudobulk: already normalize(1e4) ->
        log1p -> mean (see uns["processing"]), used as is.

    Genes live in the challenge's symbol space (data/controls/
    gene_names.csv), so G's rows line up with the context controls.
    Perturbations are keyed by target gene SYMBOL, which is what the
    STRING embedding file (P) is keyed by.
    """

    DATASETS = {
        "K562": "../../data/replogle_nadig/K562_gwps_raw_bulk_01.h5ad",
        "RPE1": "../../data/replogle_nadig/rpe1_raw_bulk_01.h5ad",
        "HepG2": "../../data/replogle_nadig/nadig_hepg2_pseudobulk.h5ad",
        "Jurkat": "../../data/replogle_nadig/nadig_jurkat_pseudobulk.h5ad",
    }

    def __init__(self, gene_names, target_sum=1e4, min_cells=30, datasets=None):
        """
        gene_names: the 18,533 challenge gene symbols, in the order of
                    data/controls/gene_names.csv.
        target_sum: per-row normalization total for the Replogle files
                    (matches Nadig's normalize(1e4)).
        min_cells:  drop targets with fewer cells than this, summed over
                    constructs. 30 is SLIM's own cutoff for HepG2/Jurkat
                    (and the cutoff Nadig's pseudobulk already applied).
        datasets:   {name: path}; defaults to DATASETS.
        """
        self.gene_names = np.asarray(gene_names)
        self.gene_index = {g: i for i, g in enumerate(self.gene_names)}
        self.target_sum = target_sum
        self.min_cells = min_cells
        self.datasets = dict(self.DATASETS if datasets is None else datasets)
        self._cache = {}  # name -> AnnData from load_dataset()

    # ------------------------------------------------------------------
    # 1. one screen -> (targets x genes) log1p pseudobulk + control
    # ------------------------------------------------------------------
    def load_dataset(self, name):
        """
        Read one screen and return an AnnData with:
            obs:  one row per target gene symbol (constructs averaged,
                  weighted by cell count); columns n_cells, n_constructs,
                  target_measured (target is one of this screen's genes,
                  i.e. its own knockdown is visible in X).
            var:  challenge gene symbols this screen measured, in
                  challenge order; column ctrl_mean = this screen's mean
                  control expression (same log1p space as X).
            X:    dense float32 mean log1p-normalized expression.
            uns:  cell_line, ctrl_n_cells.
        Cached per name, so validation can re-combine screens cheaply.
        """
        if name in self._cache:
            return self._cache[name]

        a = ad.read_h5ad(self.datasets[name])
        X = a.X.toarray() if sp.issparse(a.X) else np.asarray(a.X)
        X = X.astype(np.float64)

        already_log = "log1p" in str(a.uns.get("processing", {}).get("order", ""))
        if not already_log:
            # Replogle: mean raw UMI per cell -> normalize each row -> log1p
            X = np.log1p(X / X.sum(axis=1, keepdims=True) * self.target_sum)

        if "core_control" in a.obs.columns:  # Replogle
            ctrl = a.obs["core_control"].to_numpy().astype(bool)
            n_cells = a.obs["num_cells_filtered"].to_numpy().astype(float)
        else:  # Nadig
            ctrl = a.obs["is_control"].to_numpy().astype(bool)
            n_cells = a.obs["n_cells"].to_numpy().astype(float)

        # obs names are {idx}_{SYMBOL}_{promoter}_{ENSG}; the promoter field
        # can itself contain "_" (e.g. NM_001159524), so take field 1 only
        sym = a.obs_names.str.split("_").str[1].to_numpy()
        # non-targeting guides that aren't core controls belong in neither group
        is_pert = ~ctrl & (sym != "non-targeting") & pd.notna(sym)

        # genes: unambiguous symbols that are in the challenge panel
        unambiguous = ~a.var["gene_name"].duplicated(keep=False)
        usable = (unambiguous & a.var["gene_name"].isin(self.gene_index)).to_numpy()
        gene_sym = a.var["gene_name"].to_numpy()[usable]
        order = np.argsort([self.gene_index[g] for g in gene_sym])  # challenge order
        cols = np.where(usable)[0][order]
        gene_sym = gene_sym[order]

        w_ctrl = n_cells[ctrl]
        ctrl_mean = (w_ctrl @ X[ctrl][:, cols]) / w_ctrl.sum()

        # average constructs of the same target, weighted by cell count
        rows = np.where(is_pert)[0]
        codes, targets = pd.factorize(sym[rows])
        w = n_cells[rows]
        agg = sp.csr_matrix((w, (codes, np.arange(len(rows)))), shape=(len(targets), len(rows)))
        tot = np.asarray(agg.sum(axis=1)).ravel()
        Xt = (agg @ X[rows][:, cols]) / tot[:, None]
        n_constructs = np.bincount(codes, minlength=len(targets))

        keep = tot >= self.min_cells
        measured = set(a.var["gene_name"])
        out = ad.AnnData(
            X=Xt[keep].astype(np.float32),
            obs=pd.DataFrame(
                {
                    "n_cells": tot[keep],
                    "n_constructs": n_constructs[keep],
                    "target_measured": [t in measured for t in targets[keep]],
                },
                index=pd.Index(targets[keep], name="target_gene"),
            ),
            var=pd.DataFrame({"ctrl_mean": ctrl_mean.astype(np.float32)},
                             index=pd.Index(gene_sym, name="gene_name")),
        )
        out.uns["cell_line"] = name
        out.uns["ctrl_n_cells"] = float(w_ctrl.sum())
        self._cache[name] = out
        return out

    # ------------------------------------------------------------------
    # 2. several screens -> SLIM's Y
    # ------------------------------------------------------------------
    def build_Y(self, names=None, genes="intersection", center="ctrl", exclude_perts=()):
        """
        Stack screens into SLIM's training matrix.

        names:   cell lines to include (default: all). For LOCO, pass the
                 three training lines.
        genes:   "intersection" -- only genes every included screen
                 measured (fully observed, what SLIM's PCA assumes).
                 "union" -- every gene any of them measured; a gene a
                 screen didn't measure is filled with 0 = "no change from
                 that screen's center", so it requires center != None.
        center:  what each screen is centered on, separately, BEFORE
                 stacking (removes cell-line baseline differences, which
                 would otherwise dominate the PCA):
                 "ctrl" -- its own control mean (Y = effect vs. control).
                 "mean" -- its own mean over its perturbations (SLIM's
                           train_mean bias, applied per screen).
                 None   -- raw log1p expression (single screen only makes
                           sense; mixing lines then mixes baselines).
        exclude_perts: target symbols left out of every screen (validation:
                 simulates targets never perturbed in training). Applied
                 before center="mean" so the excluded targets can't leak in.

        Returns (Y, genes, perts):
            Y:     (n_genes, n_perts) float32, ready for SLIM's PCA.
            genes: challenge symbols, one per row of Y (challenge order).
            perts: DataFrame, one row per column of Y: cell_line,
                   target_gene, n_cells, target_measured. The same target
                   in several screens is kept as separate columns (each
                   is a training sample sharing one STRING embedding).
        """
        if center not in ("ctrl", "mean", None):
            raise ValueError(f"center must be 'ctrl', 'mean' or None, got {center!r}")
        if genes not in ("intersection", "union"):
            raise ValueError(f"genes must be 'intersection' or 'union', got {genes!r}")
        if genes == "union" and center is None:
            raise ValueError("genes='union' fills missing genes with 0, which needs center != None")

        names = list(self.datasets) if names is None else list(names)
        exclude = set(exclude_perts)
        screens = [self.load_dataset(n) for n in names]

        gene_sets = [set(s.var_names) for s in screens]
        chosen = set.intersection(*gene_sets) if genes == "intersection" else set.union(*gene_sets)
        gene_list = sorted(chosen, key=self.gene_index.__getitem__)
        row_of = {g: i for i, g in enumerate(gene_list)}

        blocks, meta = [], []
        for name, s in zip(names, screens):
            s = s[~s.obs_names.isin(exclude)]
            E = s.X.astype(np.float64)
            if center == "ctrl":
                E = E - s.var["ctrl_mean"].to_numpy()
            elif center == "mean":
                E = E - E.mean(axis=0)

            block = np.zeros((len(gene_list), s.n_obs), dtype=np.float32)
            here = [j for j, g in enumerate(s.var_names) if g in row_of]
            block[[row_of[s.var_names[j]] for j in here]] = E[:, here].T
            blocks.append(block)
            meta.append(pd.DataFrame({
                "cell_line": name,
                "target_gene": s.obs_names.to_numpy(),
                "n_cells": s.obs["n_cells"].to_numpy(),
                "target_measured": s.obs["target_measured"].to_numpy(),
            }))

        Y = np.concatenate(blocks, axis=1)
        perts = pd.concat(meta, ignore_index=True)
        return Y, gene_list, perts

    # ------------------------------------------------------------------
    # 3. control profiles, for weighting source screens by similarity
    # ------------------------------------------------------------------
    def source_ctrls(self, names=None):
        """
        Mean control profile of each screen, side by side, for comparing
        a new context's controls against the training screens.

        names: screens to include (default: all), e.g. the training lines
               of a LOCO fold. Columns come back in this order.

        Returns a DataFrame: index = challenge gene symbols every listed
        screen measured (challenge order, so no NaNs), columns = screen
        names, values = load_dataset(name).var["ctrl_mean"] (log1p space).
        The caller still has to intersect with the target context's genes
        and reorder columns to match the model's source order.
        """
        names = list(self.datasets) if names is None else list(names)
        ctrls = [self.load_dataset(n).var["ctrl_mean"] for n in names]
        shared = set.intersection(*(set(c.index) for c in ctrls))
        genes = sorted(shared, key=self.gene_index.__getitem__)
        return pd.DataFrame({n: c.loc[genes].to_numpy() for n, c in zip(names, ctrls)},
                            index=pd.Index(genes, name="gene_name"))

    def context_ctrl_mean(self, control_adata, chunk=2000):
        """
        Mean log1p-normalized control profile of one challenge context
        (data/controls/context_{A,B,C}.h5ad, raw counts per cell), in the
        same normalize(target_sum) -> log1p -> mean space as Nadig. This is
        the context's b when Y was built with center="ctrl".
        Returns a (n_genes,) float64 array in control_adata's gene order.
        Works on backed AnnData; reads chunk cells at a time.
        """
        total = np.zeros(control_adata.n_vars)
        for start in range(0, control_adata.n_obs, chunk):
            X = control_adata.X[start:start + chunk]
            X = sp.csr_matrix(X, dtype=np.float64)
            lib = np.asarray(X.sum(axis=1)).ravel()
            X = sp.diags(self.target_sum / lib) @ X
            X.data = np.log1p(X.data)  # log1p(0) = 0, so sparsity is preserved
            total += np.asarray(X.sum(axis=0)).ravel()
        return total / control_adata.n_obs
