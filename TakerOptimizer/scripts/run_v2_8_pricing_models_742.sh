#!/usr/bin/env bash
set -e

cd /home/fwz/projects/HFT_010-dev_fwz

BASE_CFG="TakerOptimizer/config/taker_position_optimizer_v2_8_local_global_742.yaml"
PRICING_ROOT="PricingModel/output/pricing_30d_ret120_selected_12_742factors_rank_guarded/pricing"
MARKET_PATH="PricingModel/data/market_return_20241022_20250114_742_by_date"

MODELS=(
  cs_dl_cnn_gru_h30
  cs_dl_feature_transformer_h30
  cs_dl_feature_transformer_h90
  cs_dl_feature_transformer_h120
  cs_dl_gru_h30
  cs_dl_gru_h60
  cs_dl_lstm_h30
  cs_dl_lstm_h60
  cs_dl_tcn_h30
  cs_dl_tcn_h60
  cs_dl_temporal_transformer_h30
  cs_dl_temporal_transformer_h60
)

mkdir -p TakerOptimizer/config
mkdir -p TakerOptimizer/outputs/logs
mkdir -p TakerOptimizer/outputs/target_positions

python -m py_compile TakerOptimizer/scripts/apply_taker_position_optimizer_v2_8_local_global.py

for MODEL in "${MODELS[@]}"; do
  ALPHA_PATH="FactorModel/outputs/hidden_factor_${MODEL}_20241022_20241122_742.csv"
  PRICING_PATH="${PRICING_ROOT}/${MODEL}/priced_dataset.csv"

  if [ ! -f "$ALPHA_PATH" ]; then
    echo "[SKIP] missing alpha: $ALPHA_PATH"
    continue
  fi

  if [ ! -f "$PRICING_PATH" ]; then
    echo "[SKIP] missing pricing: $PRICING_PATH"
    continue
  fi

  FACTOR_COL="hidden_factor_${MODEL}"
  PRED_COL="pred_ret_${MODEL}"
  FAIR_COL="fair_price_${MODEL}"

  SHORT_NAME="${MODEL#cs_dl_}"

  CFG_PATH="TakerOptimizer/config/taker_position_optimizer_v2_8_${SHORT_NAME}_local_global_742_pricing.yaml"
  OUT_PATH="TakerOptimizer/outputs/target_positions/taker_target_positions_v2_8_${SHORT_NAME}_local_global_742_pricing.csv"
  LOG_PATH="TakerOptimizer/outputs/logs/taker_position_optimizer_v2_8_${SHORT_NAME}_local_global_742_pricing.log"
  NOHUP_LOG="TakerOptimizer/outputs/logs/run_taker_position_optimizer_v2_8_${SHORT_NAME}_local_global_742_pricing.nohup.log"

  echo "============================================================"
  echo "[MODEL] $MODEL"
  echo "[ALPHA] $ALPHA_PATH"
  echo "[PRICING] $PRICING_PATH"
  echo "[OUTPUT] $OUT_PATH"

  MODEL="$MODEL" \
  ALPHA_PATH="$ALPHA_PATH" \
  PRICING_PATH="$PRICING_PATH" \
  FACTOR_COL="$FACTOR_COL" \
  PRED_COL="$PRED_COL" \
  FAIR_COL="$FAIR_COL" \
  CFG_PATH="$CFG_PATH" \
  OUT_PATH="$OUT_PATH" \
  LOG_PATH="$LOG_PATH" \
  MARKET_PATH="$MARKET_PATH" \
  python - <<'PY'
import os
import yaml
from pathlib import Path
import pandas as pd

base_cfg = Path("TakerOptimizer/config/taker_position_optimizer_v2_8_local_global_742.yaml")
cfg_path = Path(os.environ["CFG_PATH"])

with open(base_cfg, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

model = os.environ["MODEL"]

cfg["project"]["version"] = f"v2_8_local_global_742_{model}_pricing"

cfg["data"]["start_date"] = 20241022
cfg["data"]["end_date"] = 20241122
cfg["data"]["market_data_path"] = os.environ["MARKET_PATH"]

cfg["data"]["global_alpha_path"] = os.environ["ALPHA_PATH"]
cfg["data"]["local_alpha_path"] = os.environ["ALPHA_PATH"]
cfg["data"]["pricing_path"] = os.environ["PRICING_PATH"]

cfg["columns"]["global_alpha_col"] = os.environ["FACTOR_COL"]
cfg["columns"]["local_alpha_col"] = os.environ["FACTOR_COL"]
cfg["columns"]["pricing_pred_col"] = os.environ["PRED_COL"]
cfg["columns"]["fair_price_col"] = os.environ["FAIR_COL"]

cfg["data"]["output_path"] = os.environ["OUT_PATH"]
cfg["data"]["log_path"] = os.environ["LOG_PATH"]

ocfg = cfg["optimizer"]
ocfg["capital"] = 200000000

ocfg["target_gross_limit"] = 0.60
ocfg["max_gross_limit"] = 0.80
ocfg["single_name_limit"] = 0.008
ocfg["turnover_limit"] = 0.08

ocfg["global_top_n"] = 300
ocfg["global_exit_rank_n"] = 450
ocfg["global_ewma_halflife_min"] = 15

ocfg["portfolio_rebalance_interval_min"] = 5
ocfg["rebalance_smoothing"] = 0.80
ocfg["exit_smoothing"] = 0.80
ocfg["target_weight_band"] = 0.00002

ocfg["min_hold_minutes"] = 10
ocfg["trade_cooldown_minutes"] = 2

ocfg["use_local_buy_gate"] = True
ocfg["local_buy_top_n"] = 400
ocfg["local_buy_edge_min_bps"] = -5.0

ocfg["max_spread_bps"] = 100.0
ocfg["volume_cap_ratio"] = 0.35
ocfg["min_trade_notional"] = 10000

pricing_path = os.environ["PRICING_PATH"]
pred_col = os.environ["PRED_COL"]
fair_col = os.environ["FAIR_COL"]

cols = pd.read_csv(pricing_path, nrows=0).columns.tolist()
missing = [c for c in [pred_col, fair_col] if c not in cols]
if missing:
    raise ValueError(f"pricing file missing columns {missing}: {pricing_path}")

with open(cfg_path, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

print("created config:", cfg_path)
PY

  python TakerOptimizer/scripts/apply_taker_position_optimizer_v2_8_local_global.py \
    --config "$CFG_PATH" > "$NOHUP_LOG" 2>&1

  echo "[DONE] $MODEL"
  tail -n 25 "$NOHUP_LOG"
done

echo "all done"
