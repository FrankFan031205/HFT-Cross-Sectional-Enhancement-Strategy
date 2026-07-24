# CrossSectionalModel

## 1. Overview

`CrossSectionalModel` is the cross-sectional alpha module of the HFT research pipeline.

The goal of this module is to convert existing high-frequency time-series microstructure factors into cross-sectional factors, evaluate their ability to rank stocks by next-minute returns, and generate a long-only stock selection signal that can later be integrated into the market-making model.

Unlike the time-series pricing model, which predicts short-horizon return for each stock independently, the cross-sectional model answers a different question:

> At the same timestamp, which stocks are relatively stronger or weaker than others?

This module is designed for A-share long-only / maker-aware usage. Therefore, the bottom-ranked stocks are not treated as short candidates. Instead, they are used as weak-stock avoidance or inventory-reduction signals.

---

## 2. Project Structure

```text
CrossSectionalModel/
├── README.md
├── config/
│   ├── cs_dataset_h60.yaml
│   ├── cs_eval_h60.yaml
│   ├── cs_analysis_h60.yaml
│   ├── cs_stability_h60.yaml
│   ├── cs_score_h60.yaml
│   ├── cs_score_eval_h60.yaml
│   └── cs_selection_h60.yaml
│
├── data/
│   ├── raw/
│   └── processed/
│       └── cs_dataset_h60_202410_100.csv
│
├── outputs/
│   ├── cs_factor_eval_h60_202410_100.csv
│   ├── analysis_h60_202410_100/
│   ├── stability_h60_202410_100/
│   ├── cs_score_h60_202410_100.csv
│   ├── cs_score_weights_h60_202410_100.csv
│   ├── cs_score_eval_h60_202410_100/
│   ├── cs_selection_signal_h60_202410_100.csv
│   └── cross_sectional_model_report.md
│
├── scripts/
│   ├── build_cs_dataset.py
│   ├── eval_cs_factors.py
│   ├── analyze_cs_factors.py
│   ├── analyze_cs_stability.py
│   ├── build_cs_score.py
│   ├── evaluate_cs_score.py
│   ├── build_cs_selection_signal.py
│   ├── merge_cs_with_quotes.py
│   └── train_cs_model.py
│
└── src/
    ├── __init__.py
    ├── preprocess.py
    ├── evaluate.py
    ├── model.py
    └── utils.py
```

---

## 3. Core Idea

The input data has the following structure:

```text
datetime, securityid, factor columns, label_60
```

Each row represents one stock at one timestamp.

The original factors are time-series microstructure factors, such as:

```text
order book imbalance
microprice deviation
weighted order book imbalance
depth imbalance
trade imbalance
order imbalance
cancel pressure
spread
volatility
```

The cross-sectional model transforms them into same-timestamp cross-sectional factors:

```text
factor_cs_z = (factor - cross-sectional mean) / cross-sectional std
```

For example:

```text
fwz2_obi_1
    ↓
fwz2_obi_1_cs_z
```

The meaning changes from:

```text
This stock has strong order book imbalance relative to its own history.
```

to:

```text
At this timestamp, this stock has stronger order book imbalance than other stocks.
```

---

## 4. Label Definition

The main prediction target is the next-60s return:

```text
label_60
```

For cross-sectional evaluation, we also construct:

```text
label_cs_60_rank
label_cs_60_demean
```

### `label_cs_60_rank`

This is the rank-normalized future return within the same timestamp:

```text
label_cs_60_rank = (rank(label_60 within datetime) / count - 0.5) * 2
```

It is approximately in the range:

```text
[-1, 1]
```

Interpretation:

```text
close to 1   -> future 60s return is among the strongest stocks
close to 0   -> future 60s return is around the cross-sectional median
close to -1  -> future 60s return is among the weakest stocks
```

### `label_cs_60_demean`

This is the future return relative to the same-timestamp average:

```text
label_cs_60_demean = label_60 - mean(label_60 within datetime)
```

It measures future excess return within the cross-section.

---

## 5. Pipeline

### Step 1: Build Cross-sectional Dataset

Script:

```bash
python CrossSectionalModel/scripts/build_cs_dataset.py \
  --config CrossSectionalModel/config/cs_dataset_h60.yaml
```

Input:

```text
FactorModel/data/raw/factor_features_202410_100.csv
```

Output:

```text
CrossSectionalModel/data/processed/cs_dataset_h60_202410_100.csv
```

This step creates:

```text
xxx_cs_z
label_cs_60_rank
label_cs_60_demean
```

It also normalizes `securityid` to six digits to avoid merge issues.

---

### Step 2: Evaluate Cross-sectional Factors

Script:

```bash
python CrossSectionalModel/scripts/eval_cs_factors.py \
  --config CrossSectionalModel/config/cs_eval_h60.yaml
```

Output:

```text
CrossSectionalModel/outputs/cs_factor_eval_h60_202410_100.csv
```

Main metrics:

```text
mean_rankic
rankic_ir
positive_ratio
top20_excess_return
bottom20_excess_return
missing_ratio
zero_ratio
```

The evaluation uses cross-sectional RankIC:

```text
RankIC_t = Corr(rank(factor_i,t), rank(label_60_i,t))
```

For every timestamp, the factor ranks all stocks. If stocks with higher factor values also have higher future returns, RankIC is positive.

---

### Step 3: Analyze Factor Families

Script:

```bash
python CrossSectionalModel/scripts/analyze_cs_factors.py \
  --config CrossSectionalModel/config/cs_analysis_h60.yaml
```

Output directory:

```text
CrossSectionalModel/outputs/analysis_h60_202410_100/
```

Main outputs:

```text
category_summary.csv
top_by_rankic.csv
top_by_top20_excess.csv
top_by_rankic_ir.csv
suspicious_factors.csv
all_factors_sorted.csv
cs_factor_analysis_report.md
```

This step summarizes which factor families work best.

---

### Step 4: Analyze Stability

Script:

```bash
python CrossSectionalModel/scripts/analyze_cs_stability.py \
  --config CrossSectionalModel/config/cs_stability_h60.yaml
```

Output directory:

```text
CrossSectionalModel/outputs/stability_h60_202410_100/
```

Main outputs:

```text
daily_factor_stability.csv
intraday_bucket_stability.csv
factor_stability_summary.csv
cs_factor_stability_report.md
```

This step checks whether the strongest factors are stable across:

```text
different trading days
different intraday periods
```

---

### Step 5: Build Combined Cross-sectional Score

Script:

```bash
python CrossSectionalModel/scripts/build_cs_score.py \
  --config CrossSectionalModel/config/cs_score_h60.yaml
```

Outputs:

```text
CrossSectionalModel/outputs/cs_score_h60_202410_100.csv
CrossSectionalModel/outputs/cs_score_weights_h60_202410_100.csv
```

The combined score is built from stable factors:

```text
cs_score = weighted average of selected cross-sectional factors
```

The first version uses RankIC-based weights.

The output contains:

```text
datetime
securityid
label_60
label_cs_60_rank
label_cs_60_demean
cs_score_raw
cs_score_z
cs_rank
cs_group
```

---

### Step 6: Evaluate Combined Score

Script:

```bash
python CrossSectionalModel/scripts/evaluate_cs_score.py \
  --config CrossSectionalModel/config/cs_score_eval_h60.yaml
```

Output directory:

```text
CrossSectionalModel/outputs/cs_score_eval_h60_202410_100/
```

Main outputs:

```text
cs_score_overall_summary.csv
cs_score_group_summary.csv
cs_score_daily_summary.csv
cs_score_intraday_bucket_summary.csv
cs_score_eval_report.md
```

This step checks whether the combined score performs better and more stably than individual factors.

---

### Step 7: Build Long-only Selection Signal

Script:

```bash
python CrossSectionalModel/scripts/build_cs_selection_signal.py \
  --config CrossSectionalModel/config/cs_selection_h60.yaml
```

Output:

```text
CrossSectionalModel/outputs/cs_selection_signal_h60_202410_100.csv
```

This file converts `cs_rank` into maker-aware long-only trading signals:

```text
strong_long_candidate
long_candidate
weak_long_candidate
avoid_candidate
reduce_candidate
strong_reduce_candidate
target_inventory_scale
bid_priority
ask_reduce_priority
```

---

## 6. Interpretation of Selection Signal

The model does not generate a long-short portfolio.

Instead:

```text
High cs_rank:
    candidate for bid-side quoting
    higher target inventory
    higher bid priority

Low cs_rank:
    avoid buying
    disable aggressive bid quote
    reduce target inventory
    if sellable inventory exists, quote ask to reduce position
```

This is suitable for A-share long-only market making, where freely shorting the bottom-ranked stocks is not realistic.

---

## 7. Current Key Results

The combined cross-sectional score achieved:

```text
mean_rankic              = 0.142881
rankic_ir                = 1.325587
positive_ratio           = 0.905518
top20_excess_return      = 0.000265
bottom20_excess_return   = -0.000280
top_minus_bottom_excess  = 0.000545
```

The target inventory signal shows monotonic future excess returns:

```text
target_inventory_scale    avg_future_excess
0.0                       -0.000128
0.2                        0.000074
0.4                        0.000125
0.7                        0.000195
1.0                        0.000329
```

This means that higher target inventory levels correspond to stronger future cross-sectional performance.

---

## 8. Usage in Market Making

The next step is to merge this cross-sectional signal with the time-series pricing / market-making model.

Expected merge keys:

```text
datetime
securityid
```

The first safe integration should use the cross-sectional signal as a bid-side filter:

```text
quote_bid_cs = quote_bid_original and long_candidate
quote_ask_cs = quote_ask_original
```

Later versions can use:

```text
target_inventory_scale
bid_priority
ask_reduce_priority
```

to control inventory and quote aggressiveness.

---

## 9. Recommended Next Steps

1. Merge `cs_selection_signal_h60_202410_100.csv` with the existing quote decision file.
2. Build a CS-filtered quote decision file.
3. Run market-making backtest.
4. Compare original strategy vs CS-filtered strategy.
5. Add inventory-aware ask-side reduction logic.
6. Add simple risk constraints.
7. Eventually integrate optimizer / Barra-style risk control.

---

## 10. Notes

Important sanity checks:

```text
1. Verify that all factors only use current or past information.
2. Check suspicious factors with names containing ret, check, future, or label.
3. Confirm that label_60 is strictly future return.
4. Avoid interpreting bottom-ranked stocks as short candidates.
5. Use bottom-ranked stocks as avoidance / inventory-reduction signals.
```
