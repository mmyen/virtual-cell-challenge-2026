# virtual-cell-challenge-2026

Repo for the [2026 Virtual Cell Challenge](https://virtualcellchallenge.org/), which focuses on zero-shot prediction.

Models included (`models` folder):
1. `baseline` folder: Target Scaling Baseline with constant c ("scuffed baseline"), as presented in Vollenweider and Bühlmann (2026).
2. `pc_linreg` folder: Simple Linear Baseline, adapted from Ahlmann-Eltze, et al. (2025). Performs linear regression on a principal-component gene embedding (K configurable, default 10), with effects computed in either raw fold-change or log2 fold-change space.

Data used include:
1. The VCC Validation Data
2. Replogle et al., 2022
3. Nadig et al., 2025

The VCC validation data can be found in the submission portal.
Certain files (datasets, .vcc files for submission) have been omitted due to large file size.