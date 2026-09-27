import anndata as ad
import decoupler as dc
import numpy as np
import pandas as pd
import scanpy as sc
import scipy

class DataProcessor:
    def __init__(self):
        self.target_sum = 10000
        self.cells_per_pert = 400

    def log_counts(self, data):
        return sc.pp.log1p(data, copy = True)

    def pseudobulk_data(self, data, sample, groups):
        # data should be an AnnData object
        pdata = dc.pp.pseudobulk(
            adata = data,
            sample_col = sample,
            groups_col = groups,
            mode = "mean",
            min_cells = 1,
            min_counts = 0
        )
        return pdata