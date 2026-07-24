# MarketMakingModel

## 1. Overview

`MarketMakingModel` converts short-horizon prediction signals and high-frequency microstructure data into maker-style quote decisions.

The current stable version is **V5: Microstructure-Aware Fair Value Market Maker**.

The model does **not** simulate fills or calculate PnL. Its output is a full quote-decision CSV, which is designed to be consumed by the downstream `BacktestingModel`.

The research flow is:

```text
FactorModel / PredictionModel
        ↓
Hidden factor / predicted return
        ↓
MarketMakingModel V5
        ↓
Quote-decision CSV
        ↓
BacktestingModel
        ↓
Fill simulation / PnL / toxic fill / inventory analysis
```

The main idea of V5 is to build a fair value using both model alpha and real-time microstructure state:

```text
model alpha
+ microprice pressure
+ book pressure
+ trade-flow pressure
+ cancel pressure
+ inventory adjustment
+ adverse selection buffer
        ↓
quote bid / quote ask decision
```

---

## 2. Project Structure

Current structure:

```text
MarketMakingModel/
├── config/
│   ├── market_making_attention_h60_v5.yaml
│   ├── market_making_mlp2_h60_v5.yaml
│   ├── market_making_mlp60_h60_v5.yaml
│   ├── market_making_attention_h60.yaml
│   ├── market_making_mlp2_h60.yaml
│   └── market_making_mlp60_h60.yaml
├── logs/
│   ├── generate_quotes_attention_v5.log
│   ├── generate_quotes_mlp2_v5.log
│   ├── generate_quotes_mlp60_v5.log
│   └── analyze_v5_outputs.log
├── outputs/
│   ├── calibration/
│   │   ├── alpha_calibration_attention_h60_202410_100.json
│   │   ├── alpha_calibration_mlp2_h60_202410_100.json
│   │   └── alpha_calibration_mlp60_h60_202410_100.json
│   └── quote_decisions/
│       ├── quote_decisions_attention_h60_202410_100_v5.csv
│       ├── quote_decisions_mlp2_h60_202410_100_v5.csv
│       ├── quote_decisions_mlp60_h60_202410_100_v5.csv
│       └── v5_model_summary.csv
├── scripts/
│   ├── generate_quotes.py
│   ├── fit_alpha_calibration.py
│   └── analyze_v5_outputs.py
├── src/
│   ├── __init__.py
│   └── strategy.py
└── README.md
```

---

## 3. Input Data

### 3.1 Market / Microstructure Data

V5 uses the full pricing dataset:

```text
/home/fwz/projects/HFT_010-dev_fwz/PricingModel/data/pricing_dataset_h60_202410_100_full.csv
```

This file contains market snapshots and microstructure features. It is used as `market_data_path` in the V5 YAML configs.

Important columns include:

```text
date
datetime
securityid
bid1
ask1
mid_price
spread
spread_ticks
bid1_volume
ask1_volume
limit_up_price
limit_down_price
microprice
microprice_shift_ticks
book_imbalance_1
book_imbalance_5
trade_imbalance
cancel_pressure_bid
cancel_pressure_ask
cancel_pressure_imbalance
volatility_ticks
liquidity_state
```

If a microstructure column is missing, the strategy has fallbacks for several fields. For example:

```text
microprice can be computed from bid1, ask1, bid1_volume, ask1_volume.
book_imbalance_5 can be computed from depth columns when available.
trade_imbalance and cancel pressure fall back to 0 if missing.
```

However, if too many microstructure fields are missing, V5 will partially degrade toward a simpler alpha-driven strategy.

---

### 3.2 Prediction Data

V5 currently supports three prediction models:

```text
Attention:
/home/fwz/projects/HFT_010-dev_fwz/FactorModel/outputs/hidden_factor_attention_h60_202410_100.csv
prediction column: hidden_factor_attention_h60

MLP2:
/home/fwz/projects/HFT_010-dev_fwz/FactorModel/outputs/hidden_factor_mlp2_h60_202410_100.csv
prediction column: hidden_factor_mlp2_h60

MLP60:
/home/fwz/projects/HFT_010-dev_fwz/FactorModel/outputs/hidden_factor_mlp60_h60_202410_100.csv
prediction column: hidden_factor_mlp60_h60
```

All three prediction files contain:

```text
date
datetime
securityid
label_60
hidden_factor_xxx_h60
split
```

The quote generation script merges market data and prediction data by:

```text
datetime, securityid
```

The script also supports aligning 500ms market rows to 1-second prediction rows. For example:

```text
20241022_093000000 → 20241022_093000000
20241022_093000500 → 20241022_093000000
20241022_093001500 → 20241022_093001000
```

---

## 4. Calibration

The raw hidden factor may not be directly interpretable as an expected return. Therefore, each model uses a linear calibration:

```text
calibrated_pred = a * raw_pred + b
```

Calibration is fitted using `label_60` as the target.

### 4.1 Calibration Outputs

```text
outputs/calibration/alpha_calibration_attention_h60_202410_100.json
outputs/calibration/alpha_calibration_mlp2_h60_202410_100.json
outputs/calibration/alpha_calibration_mlp60_h60_202410_100.json
```

### 4.2 Fit Calibration

Attention:

```bash
python scripts/fit_alpha_calibration.py \
  --input /home/fwz/projects/HFT_010-dev_fwz/FactorModel/outputs/hidden_factor_attention_h60_202410_100.csv \
  --output /home/fwz/projects/HFT_010-dev_fwz/MarketMakingModel/outputs/calibration/alpha_calibration_attention_h60_202410_100.json \
  --pred-col hidden_factor_attention_h60 \
  --target-col label_60 \
  --datetime-col datetime \
  --max-fair-shift-ticks 5
```

MLP2:

```bash
python scripts/fit_alpha_calibration.py \
  --input /home/fwz/projects/HFT_010-dev_fwz/FactorModel/outputs/hidden_factor_mlp2_h60_202410_100.csv \
  --output /home/fwz/projects/HFT_010-dev_fwz/MarketMakingModel/outputs/calibration/alpha_calibration_mlp2_h60_202410_100.json \
  --pred-col hidden_factor_mlp2_h60 \
  --target-col label_60 \
  --datetime-col datetime \
  --max-fair-shift-ticks 5
```

MLP60:

```bash
python scripts/fit_alpha_calibration.py \
  --input /home/fwz/projects/HFT_010-dev_fwz/FactorModel/outputs/hidden_factor_mlp60_h60_202410_100.csv \
  --output /home/fwz/projects/HFT_010-dev_fwz/MarketMakingModel/outputs/calibration/alpha_calibration_mlp60_h60_202410_100.json \
  --pred-col hidden_factor_mlp60_h60 \
  --target-col label_60 \
  --datetime-col datetime \
  --max-fair-shift-ticks 5
```

---

## 5. V5 Strategy Logic

### 5.1 Pre-Trade Filters

Each row first passes basic filters:

```text
valid order book
valid prediction
spread filter
liquidity filter
limit price filter
time filter
volatility filter, if enabled
```

The main time filter is:

```text
trading_start_time = 09:35:00
trading_end_time   = 14:55:00
```

Rows before 09:35 are marked as:

```text
before_trading_time
```

These rows are kept in the output CSV but do not quote.

---

### 5.2 Model Alpha

The calibrated prediction is converted into tick units:

```text
model_alpha_ticks = mid_price * calibrated_pred / tick_size
```

This represents the model-implied fair value shift in ticks.

The model alpha is clipped by:

```text
max_fair_shift_ticks = 5.0
```

---

### 5.3 Microprice

Microprice estimates the fair value implied by L1 book imbalance:

```text
microprice =
    (ask1 * bid1_volume + bid1 * ask1_volume)
    / (bid1_volume + ask1_volume)
```

If bid-side volume is large, the microprice moves closer to the ask price, indicating upward pressure.

The shift is:

```text
microprice_shift_ticks = (microprice - mid_price) / tick_size
```

---

### 5.4 Book Pressure

Book pressure is based on order book depth imbalance:

```text
book_imbalance_5 =
    (bid_depth_5 - ask_depth_5)
    / (bid_depth_5 + ask_depth_5)
```

Then:

```text
book_pressure_shift_ticks = book_pressure_scale * book_imbalance_5
```

Current value:

```text
book_pressure_scale = 0.25
```

---

### 5.5 Trade Pressure

Trade pressure captures active buy / sell imbalance:

```text
trade_pressure_shift_ticks = trade_pressure_scale * trade_imbalance
```

Current value:

```text
trade_pressure_scale = 0.15
```

Positive trade imbalance pushes fair value upward. Negative trade imbalance pushes fair value downward.

---

### 5.6 Cancel Pressure

Cancel pressure captures disappearing liquidity on each side of the book.

Important fields:

```text
cancel_pressure_bid
cancel_pressure_ask
cancel_pressure_imbalance
```

Fair value shift:

```text
cancel_pressure_shift_ticks = cancel_pressure_scale * cancel_pressure_imbalance
```

Current value:

```text
cancel_pressure_scale = 0.10
```

Cancel pressure is also used in the side-specific adverse selection buffer.

---

### 5.7 Microstructure-Aware Fair Value

V5 fair value is computed as:

```text
microstructure_fair_price
= microprice
+ model_alpha_ticks * tick_size
+ book_pressure_shift_ticks * tick_size
+ trade_pressure_shift_ticks * tick_size
+ cancel_pressure_shift_ticks * tick_size
```

Then:

```text
alpha_ticks = (microstructure_fair_price - mid_price) / tick_size
alpha_ticks = clip(alpha_ticks, -max_fair_shift_ticks, max_fair_shift_ticks)
fair_price = mid_price + alpha_ticks * tick_size
```

This means V5 does not rely only on model prediction. It also uses current order book and flow conditions.

---

### 5.8 Inventory Adjustment

The quote fair value is adjusted by inventory:

```text
quote_fair_price = fair_price - inventory_skew
```

Inventory skew:

```text
inventory_skew
= inventory_skew_ticks
* sign(position_ratio)
* abs(position_ratio) ^ inventory_skew_power
* tick_size
```

Current values:

```text
inventory_skew_ticks = 2.0
inventory_skew_power = 2.0
```

Interpretation:

```text
Long inventory  → lower quote_fair_price → more aggressive ask, less aggressive bid.
Short inventory → higher quote_fair_price → more aggressive bid, less aggressive ask.
```

---

### 5.9 Side-Specific Adverse Selection Buffer

V5 uses separate risk buffers for bid and ask.

Bid edge:

```text
bid_edge = quote_fair_price - bid_price - bid_fee - bid_adverse_buffer
```

Ask edge:

```text
ask_edge = ask_price - quote_fair_price - ask_fee - ask_adverse_buffer
```

Bid is risky when final alpha is negative. Ask is risky when final alpha is positive.

```text
alpha_bid_risk_ticks = max(0, -alpha_ticks) * alpha_toxic_buffer_multiplier
alpha_ask_risk_ticks = max(0,  alpha_ticks) * alpha_toxic_buffer_multiplier
```

The risk scores also include trade imbalance and cancel pressure:

```text
bid_risk_score = alpha_bid_risk_ticks + bid_trade_risk_ticks + bid_cancel_risk_ticks
ask_risk_score = alpha_ask_risk_ticks + ask_trade_risk_ticks + ask_cancel_risk_ticks
```

Finally:

```text
bid_adverse_buffer = base_buffer + bid_risk_score * tick_size
ask_adverse_buffer = base_buffer + ask_risk_score * tick_size
```

---

### 5.10 Quote Decision

The model quotes when edge is above threshold:

```text
quote_bid = bid_edge >= bid_threshold
quote_ask = ask_edge >= ask_threshold
```

Default quote prices are:

```text
bid_price = bid1
ask_price = ask1
```

When alpha is strong, the strategy can improve one tick:

```text
alpha_ticks >= strong_alpha_ticks  → bid_price = bid1 + tick_size
alpha_ticks <= -strong_alpha_ticks → ask_price = ask1 - tick_size
```

Current value:

```text
strong_alpha_ticks = 1.0
```

---

## 6. V5 Output Files

The final quote-decision outputs are:

```text
outputs/quote_decisions/quote_decisions_attention_h60_202410_100_v5.csv
outputs/quote_decisions/quote_decisions_mlp2_h60_202410_100_v5.csv
outputs/quote_decisions/quote_decisions_mlp60_h60_202410_100_v5.csv
```

Each row corresponds to one market snapshot and one stock:

```text
datetime, securityid
```

No-quote rows are also kept, with reason fields such as:

```text
before_trading_time
missing_prediction
spread_too_small
low_liquidity
weak_alpha
normal
strong_alpha
```

Important output fields:

```text
raw_pred
calibrated_pred
model_alpha_ticks
alpha_ticks
alpha_bucket
microprice
microprice_shift_ticks
book_imbalance_5
book_pressure_shift_ticks
trade_imbalance
trade_pressure_shift_ticks
cancel_pressure_bid
cancel_pressure_ask
cancel_pressure_imbalance
cancel_pressure_shift_ticks
volatility_ticks
volatility_regime
liquidity_state
microstructure_fair_price
fair_price
quote_fair_price
quote_bid
quote_ask
bid_price
ask_price
bid_size
ask_size
bid_edge
ask_edge
bid_adverse_buffer
ask_adverse_buffer
risk_state
quote_style
```

---

## 7. Full-Sample Results

All three models were run on the full October 2024 100-stock dataset.

Each model produced:

```text
22,026,447 rows
```

Summary:

| Model | Rows | Bid Quote Rate | Ask Quote Rate | Any Quote Rate | Tradable-Like Rate | Quoted NaN Fair/Edge Count |
|---|---:|---:|---:|---:|---:|---:|
| attention | 22,026,447 | 0.413472 | 0.380987 | 0.785025 | 0.939056 | 0 |
| mlp2 | 22,026,447 | 0.409501 | 0.384106 | 0.784267 | 0.939056 | 0 |
| mlp60 | 22,026,447 | 0.410556 | 0.380613 | 0.781867 | 0.939056 | 0 |

---

## 8. Sanity Checks

### 8.1 No NaN in Quoted Rows

For all three models:

```text
quoted_nan_fair_edge_count = 0
```

This means all quoted rows have valid:

```text
fair_price
quote_fair_price
bid_edge
ask_edge
```

This is the most important output integrity check.

---

### 8.2 Correct Side Selection

For all models, bid quotes are associated with positive final alpha and ask quotes are associated with negative final alpha.

| Model | Alpha Mean When Quote Bid | Alpha Mean When Quote Ask |
|---|---:|---:|
| attention | +0.819258 | -0.842625 |
| mlp2 | +0.827455 | -0.829721 |
| mlp60 | +0.803171 | -0.814153 |

Interpretation:

```text
quote_bid=True means fair value is above the bid quote price.
quote_ask=True means fair value is below the ask quote price.
```

This confirms the strategy is not quoting in the wrong direction.

---

### 8.3 Model Alpha vs Final Alpha

Model alpha is slightly negative on average, while final alpha is slightly positive after adding microstructure components.

| Model | Model Alpha Mean | Final Alpha Mean |
|---|---:|---:|
| attention | -0.008594 | +0.020687 |
| mlp2 | -0.005822 | +0.022564 |
| mlp60 | -0.005073 | +0.023530 |

This shows that V5 does not simply copy the model prediction. It actively adjusts fair value using market microstructure.

---

### 8.4 Microstructure Component Means

The microstructure components are identical across models because they come from the same market data.

| Component | Mean |
|---|---:|
| microprice_shift_ticks | +0.011207 |
| book_pressure_shift_ticks | +0.018340 |
| trade_pressure_shift_ticks | -0.002801 |
| cancel_pressure_shift_ticks | -0.000439 |

Interpretation:

```text
Microprice and book pressure mildly push fair value upward.
Trade pressure and cancel pressure mildly push fair value downward.
The net microstructure effect is moderate and interpretable.
```

---

## 9. Interpretation

The full-sample results show that V5 is stable and logically consistent.

Key observations:

```text
1. All three models generated the same number of rows.
2. Prediction merge and market data merge worked correctly.
3. Quoted rows have no NaN fair/edge values.
4. Quote rates are stable across models.
5. Bid quotes correspond to positive alpha.
6. Ask quotes correspond to negative alpha.
7. Microstructure components are active and interpretable.
```

The three models produce similar quote rates. This suggests that in V5, quote decisions are driven not only by model prediction but also strongly by shared microstructure state:

```text
microprice
book pressure
trade pressure
cancel pressure
spread
fees
adverse selection buffer
inventory adjustment
```

This is expected for a microstructure-aware fair value market maker.

---

## 10. Current Limitations

V5 is still a quote-decision model, not a complete trading simulator.

It does not yet calculate:

```text
fill probability
queue position
partial fill
latency
realized PnL
unrealized PnL
inventory path
market impact
toxic fill ratio
max drawdown
```

Therefore, we should not claim that V5 improves realized PnL yet.

The correct conclusion is:

```text
V5 improves the structure and interpretability of the quote decision process.
PnL improvement must be verified in BacktestingModel.
```

---

## 11. Next Step: BacktestingModel

The next stage is to feed the V5 quote-decision CSVs into BacktestingModel.

Backtesting should evaluate:

```text
1. Fill rate
2. Realized PnL
3. PnL by quote side
4. PnL by alpha bucket
5. PnL by risk_state
6. PnL by volatility regime
7. PnL by liquidity state
8. PnL by spread regime
9. Toxic fill ratio
10. Inventory path
11. Max drawdown
12. Turnover and transaction costs
```

Recommended backtest inputs:

```text
quote_decisions_attention_h60_202410_100_v5.csv
quote_decisions_mlp2_h60_202410_100_v5.csv
quote_decisions_mlp60_h60_202410_100_v5.csv
```

---

## 12. How to Run

### 12.1 Run Quote Generation

Attention:

```bash
python scripts/generate_quotes.py \
  --config config/market_making_attention_h60_v5.yaml
```

MLP2:

```bash
python scripts/generate_quotes.py \
  --config config/market_making_mlp2_h60_v5.yaml
```

MLP60:

```bash
python scripts/generate_quotes.py \
  --config config/market_making_mlp60_h60_v5.yaml
```

---

### 12.2 Run with nohup

```bash
cd /home/fwz/projects/HFT_010-dev_fwz

nohup python MarketMakingModel/scripts/generate_quotes.py \
  --config MarketMakingModel/config/market_making_attention_h60_v5.yaml \
  > MarketMakingModel/logs/generate_quotes_attention_v5.log 2>&1 &

nohup python MarketMakingModel/scripts/generate_quotes.py \
  --config MarketMakingModel/config/market_making_mlp2_h60_v5.yaml \
  > MarketMakingModel/logs/generate_quotes_mlp2_v5.log 2>&1 &

nohup python MarketMakingModel/scripts/generate_quotes.py \
  --config MarketMakingModel/config/market_making_mlp60_h60_v5.yaml \
  > MarketMakingModel/logs/generate_quotes_mlp60_v5.log 2>&1 &
```

Check running jobs:

```bash
ps -ef | grep generate_quotes.py | grep -v grep
```

Check logs:

```bash
tail -f MarketMakingModel/logs/generate_quotes_attention_v5.log
```

---

### 12.3 Analyze V5 Outputs

```bash
cd /home/fwz/projects/HFT_010-dev_fwz/MarketMakingModel

python scripts/analyze_v5_outputs.py > logs/analyze_v5_outputs.log 2>&1
```

Check summary:

```bash
python - <<'PY'
import pandas as pd
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

path = "outputs/quote_decisions/v5_model_summary.csv"
df = pd.read_csv(path)
print(df)
PY
```

---

## 13. Final Status

Current status:

```text
MarketMakingModel V5 is complete at the quote-decision level.
```

Validated outputs:

```text
attention v5 quote decisions
mlp2 v5 quote decisions
mlp60 v5 quote decisions
```

Main conclusion:

```text
V5 successfully converts calibrated model alpha and high-frequency microstructure signals into stable, explainable maker-style quote decisions.
The generated quote-decision CSVs are ready to be used by BacktestingModel.
```
