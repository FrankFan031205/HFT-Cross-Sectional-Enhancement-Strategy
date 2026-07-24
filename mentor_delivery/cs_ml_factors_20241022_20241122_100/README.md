# Cross-Sectional ML Factors

## Data Period

20241022 to 20241122

## Objective

Generate 20 cross-sectional machine learning factors using one-month high-frequency microstructure factor data.

## Models

5 non-deep-learning models:

1. Ridge
2. Lasso SGD
3. ElasticNet SGD
4. LightGBM
5. ExtraTrees

Each model is trained on 4 prediction horizons:

- h30
- h60
- h90
- h120

Total number of generated factors:

5 models × 4 horizons = 20 factors

## Target

The training target is cross-sectionally normalized future return:

label_h_cs = zscore(label_h by datetime)

This makes the model focus on relative stock strength at each timestamp.

## Main Files

- ml_cs_hidden_factors_20_20241022_20241122_100.csv  
  Merged file containing all 20 generated ML factors.

- eval_cs_ml_factors_20_20241022_20241122_100.csv  
  Evaluation summary of the 20 factors.

- train_cs_ml_factors_20.py  
  Training script.

- feature_cols_20241022_20241122_100.yaml  
  Input raw factor list.

## Key Result

The best model family is LightGBM.

Top factors on the out-of-sample test set:

1. hidden_factor_cs_lgbm_h30
2. hidden_factor_cs_lgbm_h60
3. hidden_factor_cs_lgbm_h90
4. hidden_factor_cs_lgbm_h120

The best factor is hidden_factor_cs_lgbm_h30, with test mean cross-sectional RankIC around 0.215.
