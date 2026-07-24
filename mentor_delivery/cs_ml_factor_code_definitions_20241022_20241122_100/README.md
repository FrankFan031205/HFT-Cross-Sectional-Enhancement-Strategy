# Cross-Sectional ML Factor Definitions

This package contains the code definitions of the raw factors used for cross-sectional ML factor generation.

## Files

- factor_info.yaml  
  Factor configuration and dependency information.

- formula_factor.py  
  Python definitions of raw fwz factors.

- formula_additional_feature.py  
  Python definitions of intermediate additional features used by raw factors.

- feature_cols_20241022_20241122_100.yaml  
  The actual list of 52 raw factors used as ML model inputs.

- train_cs_ml_factors_20.py  
  Training script for generating 20 cross-sectional ML hidden factors.

- eval_cs_ml_factors_20_20241022_20241122_100.csv  
  Evaluation summary of the generated 20 ML factors.

## Data Period

20241022 to 20241122

## ML Factor Design

The generated ML factors are based on 5 non-deep-learning models:

1. Ridge
2. Lasso SGD
3. ElasticNet SGD
4. LightGBM
5. ExtraTrees

Each model is trained on 4 horizons:

- h30
- h60
- h90
- h120

Total: 5 models × 4 horizons = 20 cross-sectional ML factors.

## Note

The large factor value matrix and generated hidden factor CSV are not included in this package.
This package focuses on the raw factor code definitions and evaluation summary.
