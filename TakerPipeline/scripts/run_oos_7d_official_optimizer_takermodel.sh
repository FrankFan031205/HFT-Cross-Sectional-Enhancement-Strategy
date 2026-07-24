#!/usr/bin/env bash
set -euo pipefail

cd /mnt/data1/fwz/HFT_010-dev_fwz

ROOT=/mnt/data1/fwz/HFT_010-dev_fwz_outputs/TakerPipeline/feature_transformer_h120_oos_20241125_20250114
TAG=20241125_20241203_7d_official

mkdir -p "$ROOT/config" "$ROOT/logs" "$ROOT/hidden_factor" "$ROOT/pricing" "$ROOT/target_positions"
mkdir -p "$ROOT/taker_config" "$ROOT/taker_backtest/positions" "$ROOT/taker_backtest/metrics"

FULL_HIDDEN="$ROOT/hidden_factor/hidden_factor_cs_dl_feature_transformer_h120_20241125_20250114_742_oos.csv"
FULL_PRICING="$ROOT/pricing/priced_dataset_cs_dl_feature_transformer_h120_20241125_20250114_742_oos.csv"

HIDDEN_7D="$ROOT/hidden_factor/hidden_factor_cs_dl_feature_transformer_h120_${TAG}.csv"
PRICING_7D="$ROOT/pricing/priced_dataset_cs_dl_feature_transformer_h120_${TAG}.csv"

OPT_CFG="$ROOT/config/taker_position_optimizer_v2_8_feature_transformer_h120_oos_${TAG}.yaml"
OPT_OUT="$ROOT/target_positions/taker_target_positions_v2_8_feature_transformer_h120_local_global_742_pricing_oos_${TAG}.csv"
OPT_LOG="$ROOT/logs/taker_position_optimizer_v2_8_feature_transformer_h120_oos_${TAG}.log"
OPT_NOHUP_LOG="$ROOT/logs/run_taker_position_optimizer_v2_8_feature_transformer_h120_oos_${TAG}.nohup.log"

TAKER_CFG="$ROOT/taker_config/taker_model_feature_transformer_h120_oos_${TAG}.yaml"
TAKER_LOG="$ROOT/logs/taker_model_oos_${TAG}.log"

SUMMARY="$ROOT/taker_backtest/metrics/taker_summary_feature_transformer_h120_oos_${TAG}.csv"
TRADE_REASON="$ROOT/taker_backtest/metrics/taker_trade_reason_feature_transformer_h120_oos_${TAG}.csv"

echo "===== 1. subset 7d hidden/pricing ====="

python - <<PY
import pandas as pd
from pathlib import Path

dates = {20241125, 20241126, 20241127, 20241128, 20241129, 20241202, 20241203}

pairs = [
    (Path("$FULL_HIDDEN"), Path("$HIDDEN_7D")),
    (Path("$FULL_PRICING"), Path("$PRICING_7D")),
]

for src, dst in pairs:
    if not src.exists():
        raise FileNotFoundError(src)

    if dst.exists() and dst.stat().st_size > 0:
        print("exists, skip subset:", dst)
        continue

    print("subsetting:", src)
    first = True
    total = 0
    dst.parent.mkdir(parents=True, exist_ok=True)

    for i, chunk in enumerate(pd.read_csv(src, chunksize=1_000_000, low_memory=False)):
        chunk["date"] = chunk["date"].astype(int)
        chunk = chunk[chunk["date"].isin(dates)].copy()

        if len(chunk):
            chunk.to_csv(dst, mode="w" if first else "a", header=first, index=False)
            first = False
            total += len(chunk)

        print("chunk", i, "kept", len(chunk), "total", total, flush=True)

    print("saved:", dst)
    print("rows:", total)

    if total == 0:
        raise RuntimeError(f"subset has zero rows: {dst}")
PY

echo "===== 2. write official optimizer 7d config ====="

python - <<PY
from pathlib import Path
import yaml

base = Path("TakerOptimizer/config/taker_position_optimizer_v2_8_feature_transformer_h120_local_global_742_pricing.yaml")
if not base.exists():
    base = Path("TakerOptimizer/config/taker_position_optimizer_v2_8_local_global_742_feature_transformer_h120_pricing.yaml")

cfg = yaml.safe_load(base.read_text())

cfg.setdefault("project", {})
cfg["project"]["version"] = "v2_8_feature_transformer_h120_oos_${TAG}"

cfg.setdefault("data", {})
cfg["data"]["global_alpha_path"] = "$HIDDEN_7D"
cfg["data"]["local_alpha_path"] = "$HIDDEN_7D"
cfg["data"]["pricing_path"] = "$PRICING_7D"
cfg["data"]["output_path"] = "$OPT_OUT"
cfg["data"]["log_path"] = "$OPT_LOG"

cfg.setdefault("columns", {})
cfg["columns"]["global_alpha_col"] = "hidden_factor_cs_dl_feature_transformer_h120"
cfg["columns"]["local_alpha_col"] = "hidden_factor_cs_dl_feature_transformer_h120"
cfg["columns"]["pricing_pred_col"] = "pred_ret"
cfg["columns"]["fair_price_col"] = "fair_price"

out = Path("$OPT_CFG")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))

print("base:", base)
print("saved:", out)
print("target:", "$OPT_OUT")
PY

echo "===== 3. run official TakerOptimizer 7d ====="

python -m py_compile TakerOptimizer/scripts/apply_taker_position_optimizer_v2_8_local_global.py

python TakerOptimizer/scripts/apply_taker_position_optimizer_v2_8_local_global.py \
  --config "$OPT_CFG" \
  > "$OPT_NOHUP_LOG" 2>&1

echo "optimizer done."
ls -lh "$OPT_OUT"

echo "===== 4. inspect optimizer output ====="

python - <<PY
import pandas as pd

path = "$OPT_OUT"
df = pd.read_csv(path, low_memory=False)

print("shape:", df.shape)
print("date:", df["date"].min(), "->", df["date"].max())
print("datetimes:", df["datetime"].nunique())
print("symbols:", df["securityid"].nunique())

for c in ["optimizer_status", "optimizer_state", "side"]:
    if c in df.columns:
        print("\\n" + c)
        print(df[c].value_counts().head(20))

print("\\neffective_target_weight:")
print(df["effective_target_weight"].describe())

active = df[df["effective_target_weight"].abs() > 1e-12].copy()
print("\\nactive rows:", len(active))

if len(active):
    gross = active.groupby("datetime")["effective_target_weight"].apply(lambda x: x.abs().sum())
    print("\\ngross:")
    print(gross.describe())

    print("\\nactive names per datetime:")
    print(active.groupby("datetime")["securityid"].nunique().describe())
PY

echo "===== 5. write TakerModel 7d config ====="

cat > "$TAKER_CFG" <<YAML
project:
  name: taker_model_feature_transformer_h120_oos_${TAG}
  version: v3b_oos_official_optimizer_7d
  seed: 42

data:
  market_data_dir: /mnt/data1/fwz/HFT_010-dev_fwz/PricingModel/data/market_return_20241022_20250114_742_by_date
  market_file_pattern: market_return_{date}_742.csv
  optimizer_output_path: $OPT_OUT

  start_date: 20241125
  end_date: 20241203

output:
  position_output_path: $ROOT/taker_backtest/positions/taker_positions_feature_transformer_h120_oos_${TAG}.csv
  minute_metrics_path: $ROOT/taker_backtest/metrics/taker_minute_metrics_feature_transformer_h120_oos_${TAG}.csv
  daily_metrics_path: $ROOT/taker_backtest/metrics/taker_daily_metrics_feature_transformer_h120_oos_${TAG}.csv
  trade_reason_path: $TRADE_REASON
  summary_path: $SUMMARY

columns:
  datetime_col: datetime
  date_col: date
  symbol_col: securityid

  target_weight_col: effective_target_weight
  label_col: label_120

  bid_col: bid1
  ask_col: ask1
  mid_col: mid_price

time_alignment:
  optimizer_datetime_role: decision_time
  execution_lag_minutes: 1

execution:
  capital: 200000000
  taker_fee_bps: 0.5
  slippage_bps: 0.0

  entry_rebalance_ratio: 0.5
  exit_rebalance_ratio: 1.0

filters:
  require_optimal_for_entry: false
  require_valid_market: true

  entry_min_abs_delta_notional: 50000.0
  entry_max_spread_bps: 10.0
  entry_min_abs_net_alpha_bps: null

  exit_max_spread_bps: 999.0
  hold_min_abs_net_alpha_bps: null

  exit_when_target_zero: true
  exit_when_direction_flip: true
  reduce_when_target_smaller: false

runtime:
  market_chunksize: 2000000
  max_missing_rate: 0.2
YAML

echo "saved taker config: $TAKER_CFG"

echo "===== 6. run TakerModel 7d ====="

python TakerModel/scripts/run_taker_model_v3_exit_control_multicsv.py \
  --config "$TAKER_CFG" \
  > "$TAKER_LOG" 2>&1

echo "taker model done."

echo "===== 7. final result ====="

echo
echo "===== summary ====="
cat "$SUMMARY"

echo
echo "===== trade reason ====="
cat "$TRADE_REASON"

echo
echo "files:"
echo "$OPT_OUT"
echo "$SUMMARY"
echo "$TRADE_REASON"
