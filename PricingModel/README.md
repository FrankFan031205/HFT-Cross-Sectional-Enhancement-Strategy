# PricingModel — Pricing Stage README

## 1. Purpose

`PricingModel` is the **fair-price engine** of the HFT pipeline.

At the current stage, this module does **not** perform maker quoting, order placement, fill simulation, or PnL backtesting.  
Its only purpose is to convert model signals from `FactorModel` into calibrated expected returns and fair-price estimates.

The core workflow is:

```text
FactorModel hidden factors
        ↓
linear calibration to future return scale
        ↓
predicted return
        ↓
fair price
        ↓
alpha price / alpha ticks
        ↓
pricing evaluation metrics
```

In other words, this module answers:

```text
Given the current mid price and model signal, what is the model-implied fair price?
```

It does **not** yet answer:

```text
Should I post a bid?
Should I post an ask?
At what quote price should I place an order?
Will the order be filled?
What is the PnL?
```

Those belong to the later **maker / quoting / backtesting stage**.

---

## 2. Relationship with Other Modules

The project pipeline is organized as:

```text
FactorGenerate
    → generate raw microstructure factors and future return labels

FactorModel
    → train models such as Ridge, LGBM, AttentionNN
    → output hidden factors / prediction signals

PricingModel
    → calibrate model signals to predicted returns
    → convert predicted returns into fair prices

Quote / Maker / Backtest stage
    → use fair prices to generate bid / ask quotes
    → simulate fills and calculate PnL
```

Current `PricingModel` input comes from a prepared pricing dataset, for example:

```text
PricingModel/data/pricing_dataset_h60_202410_100.csv
```

This dataset should already contain:

```text
market data:
    date
    datetime
    securityid
    bid1
    ask1
    mid_price

model signals:
    hidden_factor_lgbm_h60
    hidden_factor_ridge_h60
    hidden_factor_attention_h60
    hidden_factor_lookback_attention_h60  # optional future signal

target labels:
    ret_60

split information:
    split
```

---

## 3. Current Directory Structure

Recommended structure:

```text
PricingModel/
├── config/
│   ├── pricing_config.yaml
│   └── export_market_return_202410_100.yaml
│
├── data/
│   ├── market_return_202410_100.csv
│   └── pricing_dataset_h60_202410_100.csv
│
├── pricing/
│   ├── __init__.py
│   ├── calibration.py        # optional modular calibration logic
│   ├── fair_price.py         # fair price transformation
│   ├── ensemble.py           # optional ensemble logic
│   └── metrics.py            # optional pricing metrics
│
├── scripts/
│   ├── run_pricing.py        # main pricing-stage entry point
│   └── run_backtest.py       # reserved for future maker/backtest stage
│
├── output/
│   ├── pricing/
│   └── reports/
│
├── logs/
│
├── quote/                    # reserved for later maker stage
└── backtest/                 # reserved for later maker stage
```

For the current stage, the main script is:

```bash
python scripts/run_pricing.py
```

---

## 4. Input Dataset

The main input is configured in:

```text
config/pricing_config.yaml
```

Example:

```yaml
paths:
  pricing_dataset_path: "data/pricing_dataset_h60_202410_100.csv"
  output_dir: "output"

data:
  date_col: "date"
  code_col: "securityid"
  timestamp_col: "datetime"
  mid_col: "mid_price"
  target_col: "ret_60"
  split_col: "split"

  signals:
    - name: "lgbm_h60"
      col: "hidden_factor_lgbm_h60"

    - name: "ridge_h60"
      col: "hidden_factor_ridge_h60"

    - name: "attention_h60"
      col: "hidden_factor_attention_h60"

    # Future signal example:
    # - name: "lookback_attention_h60"
    #   col: "hidden_factor_lookback_attention_h60"

pricing:
  return_type: "simple"
  tick_size: 0.01

calibration:
  enabled: true
  method: "linear"
  calibrate_splits: ["train", "valid"]

ensemble:
  enabled: true
  method: "equal_weight"
```

---

## 5. Required Columns

The pricing dataset must include at least the following columns:

| Column | Meaning |
|---|---|
| `date` | Trading date |
| `datetime` | Timestamp |
| `securityid` | Stock code |
| `mid_price` | Current mid price |
| `split` | Train / valid / test split |
| `ret_60` | Future 60-step realized return |
| signal columns | Model outputs from `FactorModel` |

Common signal columns:

```text
hidden_factor_lgbm_h60
hidden_factor_ridge_h60
hidden_factor_attention_h60
hidden_factor_lookback_attention_h60
```

Optional but useful columns:

```text
bid1
ask1
ret_30
ret_90
ret_120
```

`ret_60` is used as the calibration and evaluation target.  
It must **not** be used directly as a prediction signal, otherwise it creates look-ahead bias.

---

## 6. Core Pricing Logic

### 6.1 Signal

A signal is a model output from `FactorModel`, for example:

```text
hidden_factor_attention_h60
```

This signal may have predictive information, but it may not be in the correct return scale.

For example, a large positive signal usually means the model is bullish, but the raw number itself may not directly mean a return of the same magnitude.

Therefore, the signal needs calibration.

---

### 6.2 Linear Calibration

For each signal, `PricingModel` fits a linear mapping on train / validation data:

```text
ret_60 ≈ intercept + beta × signal
```

Then the calibrated predicted return is:

```text
pred_ret_i = intercept_i + beta_i × signal_i
```

where `i` represents one model signal, such as:

```text
lgbm_h60
ridge_h60
attention_h60
lookback_attention_h60
```

Calibration is trained only on:

```text
split in ["train", "valid"]
```

The test split is used only for evaluation.

This avoids look-ahead bias.

---

### 6.3 Fair Price

Once the predicted return is obtained, the fair price is calculated as:

```text
fair_price_i = mid_price × (1 + pred_ret_i)
```

when `return_type = "simple"`.

If log return is used, the formula would be:

```text
fair_price_i = mid_price × exp(pred_ret_i)
```

but the current configuration uses simple return.

---

### 6.4 Alpha Price

The price adjustment relative to the current mid price is:

```text
alpha_price_i = fair_price_i - mid_price
```

This measures how much the model-implied fair price differs from the current mid price.

---

### 6.5 Alpha Ticks

To make the fair-price adjustment more interpretable in a tick-size market, the module also calculates:

```text
alpha_ticks_i = alpha_price_i / tick_size
```

For China A-shares, the common tick size is:

```text
tick_size = 0.01
```

Example:

```text
mid_price = 20.00
pred_ret = 0.00030
fair_price = 20.006
alpha_price = 0.006
alpha_ticks = 0.6
```

This means the model-implied fair price is about 0.6 ticks above the current mid price.

---

## 7. Ensemble Fair Price

When multiple signals are available, `PricingModel` can combine calibrated predicted returns.

For equal-weight ensemble:

```text
ensemble_pred_ret = average(pred_ret_i)
```

Then:

```text
ensemble_fair_price = mid_price × (1 + ensemble_pred_ret)
ensemble_alpha_price = ensemble_fair_price - mid_price
ensemble_alpha_ticks = ensemble_alpha_price / tick_size
```

The current supported ensemble method is:

```text
equal_weight
```

Later, this can be extended to:

```text
IC-weighted ensemble
Sharpe-weighted ensemble
validation-MSE-weighted ensemble
regularized linear ensemble
```

---

## 8. Output Files

After running:

```bash
python scripts/run_pricing.py
```

the module generates:

```text
output/pricing/priced_dataset.csv
output/reports/pricing_metrics_by_signal.csv
output/reports/pricing_calibration.csv
output/reports/pricing_ensemble_weights.csv
```

---

### 8.1 `output/pricing/priced_dataset.csv`

This is the main output of the pricing stage.

It contains one row per stock-time observation and includes model-specific pricing results.

Example columns:

```text
date
datetime
securityid
split
mid_price
target_ret
bid1
ask1

signal_lgbm_h60
pred_ret_lgbm_h60
fair_price_lgbm_h60
alpha_price_lgbm_h60
alpha_ticks_lgbm_h60

signal_ridge_h60
pred_ret_ridge_h60
fair_price_ridge_h60
alpha_price_ridge_h60
alpha_ticks_ridge_h60

signal_attention_h60
pred_ret_attention_h60
fair_price_attention_h60
alpha_price_attention_h60
alpha_ticks_attention_h60

ensemble_pred_ret
ensemble_fair_price
ensemble_alpha_price
ensemble_alpha_ticks
```

Interpretation:

| Column | Meaning |
|---|---|
| `signal_*` | Raw model signal from `FactorModel` |
| `pred_ret_*` | Calibrated predicted return |
| `fair_price_*` | Model-implied fair price |
| `alpha_price_*` | Fair price minus current mid price |
| `alpha_ticks_*` | Alpha price measured in ticks |
| `ensemble_*` | Combined pricing result across signals |
| `target_ret` | Future realized return, used only for evaluation |

---

### 8.2 `output/reports/pricing_calibration.csv`

This file records the calibration coefficients for each signal.

Example columns:

```text
signal
raw_col
calibrate
calibrate_splits
intercept
beta
n_calibration
```

Interpretation:

```text
pred_ret_signal = intercept + beta × raw_signal
```

If `beta` is positive, a larger signal implies a higher expected future return.

If `beta` is negative, a larger signal implies a lower expected future return, which may indicate signal direction reversal or a model issue.

---

### 8.3 `output/reports/pricing_metrics_by_signal.csv`

This file evaluates pricing quality for each signal and split.

Example columns:

```text
signal
split
n
signal_rank_ic
pred_ret_rank_ic
pred_ret_pearson_ic
mse
mae
pred_ret_mean
pred_ret_std
target_mean
target_std
```

Important metrics:

| Metric | Meaning |
|---|---|
| `signal_rank_ic` | Spearman correlation between raw signal and future return |
| `pred_ret_rank_ic` | Spearman correlation between calibrated predicted return and future return |
| `pred_ret_pearson_ic` | Pearson correlation between predicted return and future return |
| `mse` | Mean squared error |
| `mae` | Mean absolute error |
| `pred_ret_std` | Standard deviation of predicted return |
| `target_std` | Standard deviation of realized return |

The most important split is:

```text
test
```

A good pricing signal should have:

```text
positive test RankIC
reasonable pred_ret scale
stable metrics across train / valid / test
```

---

### 8.4 `output/reports/pricing_ensemble_weights.csv`

This file records how individual signals are combined.

For equal-weight ensemble, each valid signal receives:

```text
weight = 1 / number_of_signals
```

Example:

```text
signal,weight
lgbm_h60,0.3333
ridge_h60,0.3333
attention_h60,0.3333
```

---

## 9. How to Run

From the `PricingModel/` directory:

```bash
cd /home/fwz/projects/HFT_010-dev_fwz/PricingModel
python scripts/run_pricing.py
```

Expected terminal output:

```text
Pricing finished.
saved: output/pricing/priced_dataset.csv
saved: output/reports/pricing_metrics_by_signal.csv
saved: output/reports/pricing_calibration.csv
saved: output/reports/pricing_ensemble_weights.csv
```

Check outputs:

```bash
ls output/pricing
ls output/reports
```

View metrics:

```bash
cat output/reports/pricing_metrics_by_signal.csv
cat output/reports/pricing_calibration.csv
cat output/reports/pricing_ensemble_weights.csv
```

Quick inspection with Python:

```bash
python - <<'PY'
import pandas as pd

priced = pd.read_csv("output/pricing/priced_dataset.csv")
metrics = pd.read_csv("output/reports/pricing_metrics_by_signal.csv")
calib = pd.read_csv("output/reports/pricing_calibration.csv")

print(priced.head())
print(metrics)
print(calib)
PY
```

---

## 10. How to Add a New Signal

Suppose a future `FactorModel` produces:

```text
hidden_factor_lookback_attention_h60
```

To include it in pricing, only add it to `config/pricing_config.yaml`:

```yaml
data:
  signals:
    - name: "lgbm_h60"
      col: "hidden_factor_lgbm_h60"

    - name: "ridge_h60"
      col: "hidden_factor_ridge_h60"

    - name: "attention_h60"
      col: "hidden_factor_attention_h60"

    - name: "lookback_attention_h60"
      col: "hidden_factor_lookback_attention_h60"
```

Then rerun:

```bash
python scripts/run_pricing.py
```

The output will automatically include:

```text
signal_lookback_attention_h60
pred_ret_lookback_attention_h60
fair_price_lookback_attention_h60
alpha_price_lookback_attention_h60
alpha_ticks_lookback_attention_h60
```

and it will also be included in the ensemble if `ensemble.enabled = true`.

---

## 11. Important Notes

### 11.1 Do Not Use Future Returns as Signals

Columns such as:

```text
ret_30
ret_60
ret_90
ret_120
```

are realized future returns.

They are labels, not tradable signals.

Do not configure:

```yaml
signal_col: "ret_60"
```

or:

```yaml
signals:
  - name: "future_return"
    col: "ret_60"
```

That would cause look-ahead bias.

---

### 11.2 Calibration Should Not Use Test Data

Correct:

```yaml
calibration:
  calibrate_splits: ["train", "valid"]
```

Incorrect:

```yaml
calibration:
  calibrate_splits: ["train", "valid", "test"]
```

The test split should only be used for out-of-sample evaluation.

---

### 11.3 Fair Price Is Not the Same as Quote Price

The pricing stage outputs:

```text
fair_price
```

It does not output:

```text
bid_quote
ask_quote
```

Fair price means:

```text
the model-implied reasonable value of the stock
```

Quote price means:

```text
the actual passive bid / ask order price submitted to the market
```

Quote generation requires additional logic:

```text
spread
fee
adverse selection buffer
inventory penalty
queue position
fill probability
risk limits
```

That belongs to the maker stage.

---

### 11.4 Alpha Ticks Help Decide Whether Pricing Is Actionable

Even if a signal has positive RankIC, the implied fair-price shift may be less than one tick.

Example:

```text
alpha_ticks = 0.25
```

This means the model fair price is only 0.25 ticks away from mid price.

Such a signal may still be useful, but it may not be strong enough to directly move quotes unless combined with other information.

Useful checks:

```bash
python - <<'PY'
import pandas as pd

df = pd.read_csv("output/pricing/priced_dataset.csv")

cols = [c for c in df.columns if "alpha_ticks" in c]

for c in cols:
    print("\n", c)
    print(df[c].describe(percentiles=[0.01,0.05,0.25,0.5,0.75,0.95,0.99]))
    print("abs quantiles:")
    print(df[c].abs().quantile([0.5,0.75,0.9,0.95,0.99]))
PY
```

---

## 12. Recommended Evaluation Checklist

After each run, check the following:

### 12.1 Calibration

Open:

```text
output/reports/pricing_calibration.csv
```

Check:

```text
intercept
beta
n_calibration
```

Questions:

```text
Is beta positive or negative?
Is beta magnitude reasonable?
Does any signal have zero or near-zero variance?
```

---

### 12.2 Test RankIC

Open:

```text
output/reports/pricing_metrics_by_signal.csv
```

Focus on:

```text
split = test
pred_ret_rank_ic
```

A useful signal should have positive test RankIC.

---

### 12.3 Stability Across Splits

Compare:

```text
train RankIC
valid RankIC
test RankIC
```

Good behavior:

```text
train, valid, and test RankIC are similar
```

Bad behavior:

```text
train RankIC is high but test RankIC collapses
```

This may indicate overfitting.

---

### 12.4 Predicted Return Scale

Compare:

```text
pred_ret_std
target_std
```

Usually:

```text
pred_ret_std < target_std
```

because model predictions are smoother than realized high-frequency returns.

If `pred_ret_std` is extremely large, fair prices may become unrealistic.

---

### 12.5 Alpha Ticks

Check:

```text
alpha_ticks
```

Useful interpretation:

```text
|alpha_ticks| < 0.5
    weak fair-price adjustment

0.5 <= |alpha_ticks| < 1
    moderate fair-price adjustment

|alpha_ticks| >= 1
    potentially actionable pricing signal
```

---

## 13. Troubleshooting

### 13.1 Missing Column Error

Error example:

```text
KeyError: Missing column: hidden_factor_lookback_attention_h60
```

Reason:

```text
The column is listed in pricing_config.yaml, but it does not exist in the input dataset.
```

Fix:

```text
Either remove this signal from config or regenerate the pricing dataset with this column.
```

---

### 13.2 Empty Calibration Data

Error example:

```text
No valid calibration data for hidden_factor_attention_h60
```

Possible reasons:

```text
No rows with split = train / valid
Signal column is all NaN
Target column is all NaN
Wrong split names
```

Fix:

```bash
python - <<'PY'
import pandas as pd

df = pd.read_csv("data/pricing_dataset_h60_202410_100.csv")
print(df["split"].value_counts())
print(df[["hidden_factor_attention_h60", "ret_60"]].isna().mean())
PY
```

---

### 13.3 Signal Variance Is Zero

Error example:

```text
Signal variance is zero
```

Reason:

```text
The signal has the same value for all calibration rows.
```

Fix:

```text
Check FactorModel output. The model may not have produced a valid signal.
```

---

### 13.4 Wrong Return Scale

If `alpha_ticks` is extremely large, for example:

```text
100 ticks
1000 ticks
```

then the signal may not be properly calibrated, or the input signal may not be a return-like value.

Check:

```bash
python - <<'PY'
import pandas as pd

df = pd.read_csv("output/pricing/priced_dataset.csv")
print(df[[c for c in df.columns if "pred_ret" in c]].describe().T)
print(df[[c for c in df.columns if "alpha_ticks" in c]].describe().T)
PY
```

---

## 14. Conceptual Summary

The pricing stage is a **multi-signal fair-price engine**.

For each model signal:

```text
signal_i
    ↓ calibration
pred_ret_i
    ↓ fair-price transformation
fair_price_i
    ↓ tick normalization
alpha_ticks_i
```

For multiple signals:

```text
pred_ret_lgbm
pred_ret_ridge
pred_ret_attention
pred_ret_lookback_attention
        ↓
ensemble_pred_ret
        ↓
ensemble_fair_price
```

The final output is not a trading decision yet.  
It is the model-implied fair value that will later be used by the maker / quoting stage.

---

## 15. Current Status

Current implemented stage:

```text
Pricing stage:
    FactorModel hidden factors
    → calibrated predicted returns
    → fair prices
    → alpha ticks
    → pricing metrics
```

Future stage:

```text
Maker stage:
    fair prices
    → bid / ask quote generation
    → fill simulation
    → PnL and risk metrics
```

This separation keeps the system modular:

```text
FactorModel decides what the signal is.
PricingModel decides what the fair price is.
MakerModel decides how to trade around that fair price.
```
