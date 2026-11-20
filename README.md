# virtual-cell-challenge-2026

Repo for the [2026 Virtual Cell Challenge](https://virtualcellchallenge.org/), which focuses on zero-shot prediction.

Models included (`models` folder):
1. `baseline` folder: Target Scaling Baseline with constant c ("scuffed baseline"), as presented in Vollenweider and Bühlmann (2026).
2. `pc_linreg` folder: Simple Linear Baseline, adapted from Ahlmann-Eltze, et al. (2025). Performs linear regression on a principal-component gene embedding (K configurable, default 10), with effects computed in either raw fold-change or log2 fold-change space.
3. `slim` folder: Adapted from Hu et al. (2026). https://www.biorxiv.org/content/10.64898/2026.08.07.743481v1 to include context-aware mean and direct transfer for genes perturbed in training. The transfer term was tuned to increase FID (varying mileage).
4. `final_model` folder: My final model. An extension of the context-aware mean and direct transfer SLIM model with no shared mean response and a smaller shrinkage term compared to the validation tests.

Scripts included (`scripts` folder):
1. `prep_xaira.py`: Turn the Xaira therapeutics HCT116 data into data similar to the VCC validation set.
2. `xaira_subset.py`: Download a subset of the Xaira therapeutics data for use in the proxy validation leaderboard.

Data used include:
1. The VCC 2026 Validation Data
1. The VCC 2026 Final Test Set Data
2. Replogle et al., 2022
3. Nadig et al., 2025
4. STRING embeddings 12.0 (Szklarczyk et al., 2023)
5. The VCC 2025 Test Set Data
6. Xaira Therapeutics HCT116 Test Data
After the validation phase ended, datasets (5) and (6) were used to create a proxy validation board used in the `final_model` tuning, with various seeds tested for robustness.

The VCC validation and test data can be found in the submission portal.
Certain files (datasets, .vcc files for submission) have been omitted due to large file size.

The SLIM model with direct transfer scored ~40th percentile (0.0265) on the validation, while my final model with the smaller shrinkage term exhibited similar performance on the final leaderboard (0.0654).