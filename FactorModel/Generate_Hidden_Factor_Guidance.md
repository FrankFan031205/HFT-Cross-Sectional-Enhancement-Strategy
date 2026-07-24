# Guide: Generate `hidden_factor_xx.csv`

This guide explains how to generate hidden factor prediction files in the `FactorModel` stage.

A hidden factor file looks like:

```text
outputs/hidden_factor_lgbm_h60_202410_100.csv
outputs/hidden_factor_attention_h60_202410_100.csv
outputs/hidden_factor_mlp2_h60_202410_100.csv
outputs/hidden_factor_lookback_attention_h60_202410_100.csv
```

Each file contains a supervised model prediction signal, usually trained to predict:

```text
label_60
```

`label_60` means the future 60-horizon mid-price return.

---

## 1. What the Script Does

The script `generate_hidden_factor.py` is a dispatcher. It reads a model YAML config, checks `model.type`, calls the correct training script, and verifies that `prediction.output_path` was created.

Supported mapping:

| `model.type` | Training script |
|---|---|
| `ridge` | `src/train_model.py` |
| `lightgbm` | `src/train_model.py` |
| `feature_attention_nn` | `src/train_attention_nn.py` |
| `mlp` | `src/train_mlp.py` |
| `residual_mlp` | `src/train_mlp.py` |
| `lookback_attention_nn` | `src/train_lookback_attention.py` |

---

## 2. Put the Script Here

Copy the script to:

```text
/home/fwz/projects/HFT_010-dev_fwz/FactorModel/src/generate_hidden_factor.py
```

Then run all commands from the `FactorModel` root:

```bash
cd /home/fwz/projects/HFT_010-dev_fwz/FactorModel
```

---

## 3. Required Input Files

Before generating a hidden factor, make sure these files exist:

```text
data/raw/factor_features_202410_100.csv
data/raw/feature_cols_202410_100.yaml
data/processed/factor_features_202410_100_processed.csv
```

Check:

```bash
ls -lh data/raw/factor_features_202410_100.csv
ls -lh data/raw/feature_cols_202410_100.yaml
ls -lh data/processed/factor_features_202410_100_processed.csv
```

If processed data does not exist, run preprocessing first:

```bash
python src/preprocess.py --config configs/factor_model_lgbm.yaml
```

---

## 4. Config Requirements

Each model config should include:

```yaml
project:
  name: factor_model_xxx_h60
  version: v1
  seed: 42

data:
  processed_data_path: data/processed/factor_features_202410_100_processed.csv
  date_col: date
  datetime_col: datetime
  symbol_col: securityid
  feature_cols_path: data/raw/feature_cols_202410_100.yaml
  label_col: label_60

  train_start: 20241022
  train_end: 20241028
  valid_start: 20241029
  valid_end: 20241029
  test_start: 20241030
  test_end: 20241031

model:
  type: xxx

prediction:
  output_col: hidden_factor_xxx_h60
  output_path: outputs/hidden_factor_xxx_h60_202410_100.csv
```

The most important fields are:

```text
model.type
prediction.output_col
prediction.output_path
```

---

## 5. Generate One Hidden Factor

### LGBM

```bash
python src/generate_hidden_factor.py \
  --config configs/factor_model_lgbm.yaml
```

Expected output:

```text
outputs/hidden_factor_lgbm_h60_202410_100.csv
```

### Feature AttentionNN

```bash
python src/generate_hidden_factor.py \
  --config configs/factor_model_attention.yaml
```

Expected output:

```text
outputs/hidden_factor_attention_h60_202410_100.csv
```

### 2-layer MLP

```bash
python src/generate_hidden_factor.py \
  --config configs/factor_model_mlp2.yaml
```

Expected output:

```text
outputs/hidden_factor_mlp2_h60_202410_100.csv
```

### 60-layer Residual MLP

```bash
python src/generate_hidden_factor.py \
  --config configs/factor_model_mlp60.yaml
```

Expected output:

```text
outputs/hidden_factor_mlp60_h60_202410_100.csv
```

### Lookback AttentionNN

```bash
python src/generate_hidden_factor.py \
  --config configs/factor_model_lookback_attention.yaml
```

Expected output:

```text
outputs/hidden_factor_lookback_attention_h60_202410_100.csv
```

---

## 6. Generate Multiple Hidden Factors Sequentially

```bash
python src/generate_hidden_factor.py \
  --config configs/factor_model_lgbm.yaml \
  --config configs/factor_model_attention.yaml \
  --config configs/factor_model_mlp2.yaml
```

The script runs them one by one.

---

## 7. Run with `nohup`

For long jobs:

```bash
nohup python -u src/generate_hidden_factor.py \
  --config configs/factor_model_lookback_attention.yaml \
  > logs/generate_lookback_attention.log 2>&1 &
```

Check progress:

```bash
tail -f logs/generate_lookback_attention.log
```

Exit log view:

```text
Ctrl + C
```

This does not stop the process.

---

## 8. Skip Existing Outputs

If the output already exists and you do not want to retrain:

```bash
python src/generate_hidden_factor.py \
  --config configs/factor_model_lgbm.yaml \
  --skip_if_exists
```

The script will check the existing file and print a summary.

---

## 9. Dry Run

To print the command without running it:

```bash
python src/generate_hidden_factor.py \
  --config configs/factor_model_lgbm.yaml \
  --dry_run
```

---

## 10. Verify the Output File

The script automatically checks:

```text
1. Output file exists
2. hidden_factor_* column exists
3. Prediction non-null ratio
4. Date range
5. Number of stocks
6. Split counts
```

Manual check:

```bash
python - <<'EOF'
import pandas as pd

path = "outputs/hidden_factor_lgbm_h60_202410_100.csv"
df = pd.read_csv(path, nrows=5)

print(df.columns.tolist())
print(df.head())
EOF
```

Expected columns:

```text
date
datetime
securityid
label_60
hidden_factor_lgbm_h60
split
```

---

## 11. Evaluate After Generation

After generating a hidden factor, run evaluation:

```bash
python src/evaluate_hidden_factor.py \
  --config configs/eval_hidden_factor_lgbm.yaml
```

Output:

```text
outputs/eval_hidden_factor_lgbm_h60_202410_100.csv
outputs/eval_hidden_factor_lgbm_h60_202410_100_timeseries.csv
```

Compare all hidden factors:

```bash
python src/compare_all_hidden_factors.py
```

Output:

```text
outputs/compare_all_hidden_factors_h60_202410_100.csv
```

---

## 12. Common Issues

### `empty train set`

The config date range does not overlap with processed data.

Current processed data only has:

```text
20241022 to 20241031
```

Use:

```yaml
train_start: 20241022
train_end: 20241028
valid_start: 20241029
valid_end: 20241029
test_start: 20241030
test_end: 20241031
```

### `ModuleNotFoundError`

Install missing packages:

```bash
pip install pyyaml pandas numpy scipy scikit-learn joblib lightgbm torch
```

or:

```bash
conda install -c conda-forge scikit-learn scipy joblib lightgbm -y
```

### Output file missing

Check the generation log:

```bash
tail -100 logs/generate_hidden_factor_<project_name>.log
```

### Training still running

```bash
ps -ef | grep train_ | grep -v grep
```

Detailed process status:

```bash
ps -p <PID> -o pid,stat,etime,time,pcpu,pmem,cmd
```

---

## 13. Recommended Workflow

```text
1. Generate raw factor data from FactorGenerate.
2. Generate feature_cols_202410_100.yaml.
3. Run preprocessing.
4. Generate hidden_factor_xx.csv with generate_hidden_factor.py.
5. Evaluate the hidden factor.
6. Compare all hidden factors.
7. Merge the best hidden factor into PricingModel.
```

Recommended primary model currently:

```text
hidden_factor_lookback_attention_h60
```

because it had the best out-of-sample cross-sectional RankIC in the current comparison.
