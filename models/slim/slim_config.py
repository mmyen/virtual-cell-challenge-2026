"""SlimConfig: the single knob-set shared by all BilinearModel/BilinearComboModel runs.

Replaces the CONDITION_COL/CONTROL_TAG/STRING_DIM/K_PCA/LAMBDA_REG module-level
constants that used to be duplicated (with occasional drift) at the top of
every run_slim*.py script.
"""

# Code adapted from RasmussenLab/SLIM (MIT license, Copyright (c) 2026 RasmussenLab)
# See LICENSE-SLIM file

from dataclasses import dataclass


@dataclass
class SlimConfig:
    condition_col: str = "condition"   # obs column holding perturbation labels
    control_tag: str = "ctrl"          # value in condition_col identifying control cells
    string_dim: int = 100_000          # embedding dimensions to use (STRING/SPACE default: 100k)
    K: int = 10                        # basis rank (PCA/NMF/AE components)
    lambda_reg: float = 0.1            # Ridge regularization strength for solve_W
    basis_method: str = "pca"          # "pca" | "nmf" | "autoencoder" | "random"
    bias_method: str = "train_mean"    # "train_mean" | "ctrl_mean"
                                        # (BilinearComboModel also accepts "two_bias" | "shared")
    solver_method: str = "ridge_closed_form"  # "ridge_closed_form" | "mlp_decoder"
    population_method: str = "rescale"  # "rescale" | "gaussian" (passed to scaffold.build_result_h5ad)
    seed: int = 1                      # random seed for sampling / stochastic basis methods

    # autoencoder-basis-specific (only read when basis_method == "autoencoder")
    ae_hidden_dim: int = 32
    ae_epochs: int = 500
    ae_lr: float = 1e-3

    # mlp_decoder-solver-specific (only read when solver_method == "mlp_decoder")
    decoder_hidden_dim: int = 512
    decoder_dropout: float = 0.2
    decoder_epochs: int = 200
    decoder_batch_size: int = 64
    decoder_lr: float = 1e-3