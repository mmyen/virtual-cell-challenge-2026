# TODO: implement baseline slim
# Then put in context-aware mean?

# IMPORT STATEMENTS

# Code adapted from RasmussenLab/SLIM (MIT license, Copyright (c) 2026 RasmussenLab)
# See LICENSE-SLIM file

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn

from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, TensorDataset

# from basis.py
def pca_basis(Y_centered, K = 10):
    """PCA basis of the centered pseudobulk matrix. Default SLIM basis.

    Args:
        Y_centered: (n_genes, n_perts) pseudobulk expression, centered by bias.
        K: target number of components (capped at n_perts - 1).

    Returns:
        G: (n_genes, n_comp) basis matrix.
    """
    n_comp = min(K, Y_centered.shape[1] - 1)
    pca = PCA(n_components=n_comp)
    return pca.fit_transform(Y_centered)

# from bias.py
def train_mean_bias(Y):
    """Bias = mean pseudobulk across all training perturbations. Default.

    Args:
        Y: (n_genes, n_perts) pseudobulk expression.

    Returns:
        b: (n_genes, 1) bias vector.
    """
    return Y.mean(axis=1, keepdims=True)

# from embeddings.py
# I want to modify this later so I can test different types of embeddings (side offshoot)
def load_h5_embeddings(filepath, dim=None):
    """Load a gene -> embedding-vector table from an HDF5 file.

    Args:
        filepath: path to an HDF5 file with one dataset per gene symbol.
        dim: if given, truncate each embedding vector to its first `dim`
            entries (used to cap STRING/SPACE embeddings at a working size).

    Returns:
        (genes2idx, embeddings): `genes2idx` maps gene symbol -> row index,
        `embeddings` is an (n_genes, dim) array.
    """
    with h5py.File(filepath, "r") as f:
        genes = list(f.keys())
        emb = np.array([f[g][:] for g in genes])
    if dim:
        emb = emb[:, :dim]
    return {g: i for i, g in enumerate(genes)}, emb

def select_embeddings(genes, genes2idx, embeddings, zero_fill=False):
    """Look up embedding rows for a list of gene symbols.

    Args:
        genes: list of gene symbols to look up.
        genes2idx: gene -> row index, as returned by `load_h5_embeddings`.
        embeddings: (n_genes, dim) embedding matrix.
        zero_fill: if True, genes missing from `genes2idx` get a zero vector
            instead of being dropped (used at inference time so every
            requested perturbation gets a prediction).

    Returns:
        (rows, valid_idx): `rows` is an (n_selected, dim) array, `valid_idx`
        are the positions in `genes` that were included in `rows`.
    """
    rows, valid_idx = [], []
    dim = embeddings.shape[1]
    for i, gene in enumerate(genes):
        if gene in genes2idx:
            rows.append(embeddings[genes2idx[gene]])
            valid_idx.append(i)
        elif zero_fill:
            rows.append(np.zeros(dim))
            valid_idx.append(i)
    return np.array(rows), valid_idx

# From model.py
def solve_W(G, P, Y, lambda_reg):
    """Closed-form Ridge solve: W = (G'G + lambda*I)^-1 G'Y P (P'P + lambda*I)^-1."""
    K, D = G.shape[1], P.shape[1]
    Z = np.linalg.solve(G.T @ G + lambda_reg * np.eye(K), G.T @ Y)
    return np.linalg.solve(P.T @ P + lambda_reg * np.eye(D), (Z @ P).T).T

class _DecoderMLP(nn.Module):
    """MLP mapping a STRING embedding directly to a K-dim basis code.

    Used in place of the closed-form Ridge solve when
    config.solver_method == "mlp_decoder": prediction is
    `G @ decoder(string_emb) + b`, trained end-to-end with MSE loss on
    gene space instead of solving for a linear W.
    """

    def __init__(self, string_dim, K, hidden_dim=512, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(string_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, K),
        )

    def forward(self, string_emb):
        return self.net(string_emb)

class Model:
    """Fit and predict with the SLIM bilinear model for single perturbations."""

    def __init__(self, gene_names, embeddings_path, config):
        self.gene_names = np.asarray(gene_names)
        self.config = config
        self.N_genes = len(self.gene_names)
        self.gene_index = {g: i for i, g in enumerate(self.gene_names)}
        self.genes2idx, self.embeddings = load_h5_embeddings(embeddings_path, config.string_dim)

    def fit(self, Y, genes, perts):
        """Fit on training perturbations from a loaded DatasetAdapter.

        Args:
            Y: placeholder (already centered)
            genes: placeholder
            perts: placeholder
        """
        self.b_c = None
        cfg = self.config
        self.var_names = genes
        self.bias_method = cfg.bias_method  # read by pipeline.save_results

        self._Y = Y                                   # the Y passed in (relative to control), NOT Y_centered

        self._targets = perts["target_gene"].to_numpy()   # array, not list, so np.isin works well
        if perts is None:
            raise ValueError("Provide list of perturbations.")
        targets = perts['target_gene'].tolist()

        self.row_in_G = np.full(self.N_genes, -1)
        self.row_in_G[[self.gene_index[g] for g in genes]] = np.arange(len(genes)) 

        # Center screens on means- context aware
        lines = perts['cell_line'].to_numpy()
        codes, names = pd.factorize(lines)
        self._codes = codes                           # integer line id per column
        self.source_means = np.column_stack([Y[:, codes == k].mean(axis=1) for k in range(len(names))])
        self.source_names = list(names)
        Y_centered = Y - self.source_means[:, codes]

        # # Bias
        self.b = train_mean_bias(Y).ravel()
        # Y_centered = Y - self.b[:, None]



        # Basis
        if cfg.basis_method == "pca":
            self.G = pca_basis(Y_centered, cfg.K)
        else:
            raise ValueError(f"Unknown basis_method: {cfg.basis_method!r} (As is, this code only supports PCA)")

        # Solver
        P, valid_emb_idx = select_embeddings(targets, self.genes2idx, self.embeddings)
        if cfg.solver_method == "ridge_closed_form":
            self.W = solve_W(self.G, P, Y_centered[:, valid_emb_idx], cfg.lambda_reg)
            self.decoder = None
        elif cfg.solver_method == "mlp_decoder":
            self.W = None
            self.decoder = self._fit_decoder(targets, Y_centered, cfg)
        else:
            raise ValueError(f"Unknown solver_method: {cfg.solver_method!r}")

        n_covered = len([p for p in targets if p in self.genes2idx])
        print(f"  Fitted on {len(targets)} perts, "
              f"STRING coverage: {n_covered}/{len(targets)}")

    def _fit_decoder(self, train_perts_valid, Y_centered, cfg):
        """Train an MLP decoder (string_emb -> basis code) in place of solve_W.

        Only training perturbations with STRING coverage are used (the
        decoder needs a real embedding as input); trained with mini-batch
        Adam + MSE loss on gene space, matching run_slim_decoder.py.
        """
        emb_list, shift_list = [], []
        for gene, y_col in zip(train_perts_valid, Y_centered.T):
            if gene not in self.genes2idx:
                continue
            emb_list.append(self.embeddings[self.genes2idx[gene]])
            shift_list.append(y_col)
        print(f"  STRING coverage: {len(emb_list)}/{len(train_perts_valid)} training perts")

        emb_t = torch.tensor(np.array(emb_list), dtype=torch.float32)
        shift_t = torch.tensor(np.array(shift_list), dtype=torch.float32)  # (n, n_genes)
        G_t = torch.tensor(self.G, dtype=torch.float32)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        decoder = _DecoderMLP(
            string_dim=self.embeddings.shape[1], K=self.G.shape[1],
            hidden_dim=cfg.decoder_hidden_dim, dropout=cfg.decoder_dropout,
        ).to(device)
        G_t = G_t.to(device)

        optimizer = torch.optim.Adam(decoder.parameters(), lr=cfg.decoder_lr)
        criterion = nn.MSELoss()
        loader = DataLoader(TensorDataset(emb_t, shift_t),
                             batch_size=cfg.decoder_batch_size, shuffle=True)

        for epoch in range(1, cfg.decoder_epochs + 1):
            decoder.train()
            epoch_loss = 0.0
            for emb_b, shift_b in loader:
                emb_b, shift_b = emb_b.to(device), shift_b.to(device)
                optimizer.zero_grad()
                z = decoder(emb_b)                 # (batch, K)
                pred = (G_t @ z.t()).t()            # (batch, n_genes)
                loss = criterion(pred, shift_b)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * emb_b.size(0)
            if epoch % 50 == 0 or epoch == 1:
                print(f"  Epoch {epoch:4d}/{cfg.decoder_epochs}  loss={epoch_loss / len(loader.dataset):.4f}")

        decoder.eval()
        return decoder

    def predict_effects(self, target_genes, knockdown_fraction = 0.9, gene_rows = None, alpha = 1.5):
        """Predict mean expression for each perturbation gene.

        Returns:
            dict {gene_name: predicted_mean_expression (n_genes,)}
        """
        if gene_rows is None:
            rows = np.arange(self.N_genes)
        else:
            rows = np.asarray(gene_rows)
        idx = self.row_in_G[rows]
        ok = (idx >= 0)

        P_new, _ = select_embeddings(target_genes, self.genes2idx, self.embeddings, zero_fill=True)

        if self.config.solver_method == "ridge_closed_form":
            delta = self.G @ self.W @ P_new.T                  # (n_G_genes, n_targets)
        else:
            device = next(self.decoder.parameters()).device
            P_t = torch.tensor(P_new, dtype=torch.float32).to(device)
            G_t = torch.tensor(self.G, dtype=torch.float32).to(device)
            with torch.no_grad():
                delta = (G_t @ self.decoder(P_t).t()).cpu().numpy()

        E = np.zeros((len(rows), len(target_genes)))
        if self.b_c is not None:
            E[ok] = self.b_c[idx[ok], None] + alpha * delta[idx[ok]]
        else:
            E[ok] = self.b[idx[ok], None] + alpha * delta[idx[ok]]

        E /= np.log(2) # Y i snatural log

        if knockdown_fraction is not None:
            pos = {g: i for i, g in enumerate(rows)}
            kd = np.broadcast_to(np.asarray(knockdown_fraction, dtype=float), (len(target_genes),))
            for j, t in enumerate(target_genes):
                gi = self.gene_index.get(t)
                if gi in pos:
                    E[pos[gi], j] = np.log2(1 - kd[j])
        return E.astype(np.float32)

    def _predicted_cells(self, control_adata, p_sym, cells_per_pert, rng,
                         knockdown_fraction, alpha, gain=1.0, rounding="nearest"):
        """
        One perturbation's predicted cells: cells_per_pert real control cells
        (sampled without replacement), each gene scaled by the predicted
        effect and rounded to counts. Returns a (cells_per_pert, N_genes)
        float32 CSR block. Shared by predict_effects() and write_submission() so the
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

        multiplier = 2.0 ** effect

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

    def write_submission(self, path, context_controls, context_ctrl_profiles, source_ctrls,
                         panel_by_context, target_genes_by_context, tau = 0.05, mix = 0.3,
                         knockdown_fraction=0.9, alpha=1.5, cells_per_pert=400,
                         rng=None, compression="gzip", gain=1.2, rounding="stochastic"):
        """
        placeholder documentation
        """
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
                self.set_context(context_ctrl_profiles[ctx], source_ctrls, tau = tau, mix = mix, panel = panel_by_context[ctx])
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

    def set_context(self, new_context_control, old_context_controls, tau = 0.05, mix = 0.3, panel = None, min_n = 30):
        shared = new_context_control.index.intersection(old_context_controls.index)

        # call calculate_similarity_weights
        self.weights = calculate_similarity_weights(new_context_control.loc[shared].to_numpy(),
                                                    old_context_controls.loc[shared, self.source_names].to_numpy(),
                                                    tau,
                                                    mix)
        # start from each line's full average; overwrite with panel averages where possible
        M_panel = self.source_means.copy()                  # (G genes × lines)

        if panel is not None:
            for k, name in enumerate(self.source_names):
                mask = (self._codes == k) & np.isin(self._targets, panel)
                if mask.sum() >= min_n:                     # enough panel targets in this line
                    M_panel[:, k] = self._Y[:, mask].mean(axis=1)
                # else: column k keeps line k's full average

        self.b_c = M_panel @ self.weights

def calculate_similarity_weights(new_context_control, old_context_controls, tau = 0.05, mix = 0.0):
    correlations = []
    num_sources = old_context_controls.shape[1]
    for i in range(num_sources):
        correlations.append(np.corrcoef(new_context_control, old_context_controls[:,i])[0,1])

    correlations = np.array(correlations)
    w = np.exp((correlations - correlations.max())/tau)

    # normalize weights
    w /= w.sum()

    # blend towards uniform
    w = (1 - mix) * w + mix / num_sources

    return w

