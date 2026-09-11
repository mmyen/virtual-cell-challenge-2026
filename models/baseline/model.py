import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy


class Model:
    def __init__(self):
        # 0.9 is a placeholder value. If 83% of cells have >80% knockdown, then
        # that means the penetrance coefficient should be at least 80%. I decided
        # to split the difference between 0.8 and 1.
        self.c = -0.9
        self.target_sum = 10000
        self.cells_per_pert = 400
    
    def log_normalize_data(self, data):
        # The original Signal, Bounds, and Baselines paper normalizes library
        # size to a target sum and log 1p normalized using Scanpy, so that is what
        # I am doing here.
        d = data.X.tocsr().astype(np.float32) # astype returns a copy
        library_sizes = np.asarray(d.sum(axis = 1)).ravel() # row total
        normalized_data = ad.AnnData(d)
        sc.pp.normalize_total(normalized_data, target_sum = self.target_sum)
        sc.pp.log1p(normalized_data)
        return normalized_data, library_sizes

    def convert_data_to_raw_counts(self, data):
        return np.expm1(data).astype(int)

    def calculate_means(self, data):
        # Calculate mean
        return np.asarray(data.X.mean(axis=0)).ravel().astype(np.float64)

    def align_gene_names(self, data, gene_names):
        # make sure gene names in predictions and original data max
        if list(data.var_names) != gene_names:
            data = data[:, gene_names].copy()
        return data

    def predict_gene(self, means, gene_names, pert_gene):
        # gene_names should be a list of genes
        if pert_gene not in gene_names:
            return means
        else:
            pert_ind = gene_names.index(pert_gene)
            pert_means = means.copy()
            pert_means[pert_ind] = pert_means[pert_ind] * (1 + self.c)
            return pert_means

    def predict_profiles(self, data, gene_names, pert_counts):
        # Normalize the data
        normalized_data, lib_sizes = self.log_normalize_data(data)

        # Calculate the mean gene expression values
        gene_exp_means = self.calculate_means(normalized_data)

        # Make list of gene names (incase list is not already passed in)
        # as well as list of genes to perturb
        if type(gene_names) != list:
            gene_names = list(gene_names)
        if type(pert_counts) != list:
            pert_counts = list(pert_counts)

        # Predict perturbations across all genes
        X_counts = [] # predicted counts
        pred_labels = [] # corresponding gene labels
        for p in pert_counts:
            raw_predictions = self.predict_gene(gene_exp_means, gene_names, p)
            prediction_counts = self.convert_data_to_raw_counts(raw_predictions)
            row = scipy.sparse.csr_matrix(prediction_counts.astype(np.float32).reshape(1, -1))
            X_counts.append(scipy.sparse.vstack([row] * self.cells_per_pert, format = 'csr'))
            pred_labels.extend([p] * self.cells_per_pert)

        X = scipy.sparse.vstack(X_counts, format = 'csr')

        predictions = ad.AnnData(
            X = X,
            obs = pd.DataFrame({'target_gene': pred_labels}),
            var = pd.DataFrame(index = list(data.var_names))
        )
        predictions = self.align_gene_names(predictions, gene_names)

        return predictions




