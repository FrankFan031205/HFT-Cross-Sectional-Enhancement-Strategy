#!/usr/bin/env bash
set -e

cd /mnt/data1/fwz/HFT_010-dev_fwz

MARKET_PATH="PricingModel/data/market_return_20241022_20250114_742_by_date"
PRICING_ROOT="PricingModel/output/pricing_30d_ret120_selected_12_742factors_rank_guarded/pricing"

HORIZONS=(h30 h60 h90 h120)

mkdir -p TakerOptimizer/config
mkdir -p TakerOptimizer/outputs/logs
mkdir -p TakerOptimizer/outputs/target_positions

python -m py_compile TakerOptimizer/scripts/apply_taker_position_optimizer_v2_9_local_global_risk_overlay.py

for H in "${HORIZONS[@]}"; do
  MODEL="cs_dl_feature_transformer_${H}"
  SHORT_NAME="feature_transformer_${H}"

  ALPHA_PATH="FactorModel/outputs/hidden_factor_${MODEL}_20241022_20241122_742.csv"
  PRICING_PATH="${PRICING_ROOT}/${MODEL}/priced_dataset.csv"

  FACTOR_COL="hidden_factor_${MODEL}"
  PRED_COL="pred_ret_${MODEL}"
  FAIR_COL="fair_price_${MODEL}"

  CFG_PATH="TakerOptimizer/config/taker_position_optimizer_v2_9_${SHORT_NAME}_742_risk_overlay.yaml"
  OUT_PATH="TakerOptimizer/outputs/target_positions/taker_target_positions_v2_9_${SHORT_NAME}_742_risk_overlay.csv"
  LOG_PATH="TakerOptimizer/outputs/logs/taker_position_optimizer_v2_9_${SHORT_NAME}_742_risk_overlay.log"
  NOHUP_LOG="TakerOptimizer/outputs/logs/run_taker_position_optimizer_v2_9_${SHORT_NAME}_742_risk_overlay.nohup.log"

  echo "============================================================"
  echo "[MODEL] ${MODEL}"
  echo "[ALPHA] ${ALPHA_PATH}"
  echo "[PRICING] ${PRICING_PATH}"

  if [ ! -f "$ALPHA_PATH" ]; then
    echo "[SKIP] missing alpha: $ALPHA_PATH"
    continue
  fi

  if [ ! -f "$PRICING_PATH" ]; then
    echo "[SKIP] missing pricing: $PRICING_PATH"
    continue
  fi

  MODEL="$MODEL" \
  H="$H" \
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
from pathlib import Path
import pandas as pd
import yaml

model = os.environ["MODEL"]
h = os.environ["H"]

alpha_path = os.environ["ALPHA_PATH"]
pricing_path = os.environ["PRICING_PATH"]
factor_col = os.environ["FACTOR_COL"]
pred_col = os.environ["PRED_COL"]
fair_col = os.environ["FAIR_COL"]
cfg_path = Path(os.environ["CFG_PATH"])

# check alpha column
alpha_cols = pd.read_csv(alpha_path, nrows=0).columns.tolist()
if factor_col not in alpha_cols:
    raise ValueError(f"missing alpha col {factor_col} in {alpha_path}")

# check pricing columns
pricing_cols = pd.read_csv(pricing_path, nrows=0).columns.tolist()
missing = [c for c in [pred_col, fair_col] if c not in pricing_cols]
if missing:
    raise ValueError(f"missing pricing cols {missing} in {pricing_path}")

cfg = {
    "project": {
        "name": "taker_position_optimizer",
        "version": f"v2_9_{model}_742_risk_overlay",
        "seed": 42,
    },
    "data": {
        "start_date": 20241022,
        "end_date": 20241122,
        "market_data_path": os.environ["MARKET_PATH"],
        "global_alpha_path": alpha_path,
        "local_alpha_path": alpha_path,
        "pricing_path": pricing_path,
        "daily_pnl_path": "TakerOptimizer/inputs/daily_strategy_pnl.csv",
        "output_path": os.environ["OUT_PATH"],
        "log_path": os.environ["LOG_PATH"],
    },
    "columns": {
        "date_col": "date",
        "datetime_col": "datetime",
        "symbol_col": "securityid",
        "global_alpha_col": factor_col,
        "local_alpha_col": factor_col,
        "pricing_pred_col": pred_col,
        "fair_price_col": fair_col,
        "price_col": "mid_price",
        "bid_col": "bid1",
        "ask_col": "ask1",
        "spread_col": "spread",
        "bid_volume_col": "bid1_volume",
        "ask_volume_col": "ask1_volume",
        "limit_up_col": "limit_up_price",
        "limit_down_col": "limit_down_price",
    },
    "optimizer": {
        "capital": 200000000,

        "target_gross_limit": 0.80,
        "max_gross_limit": 0.80,
        "single_name_limit": 0.008,
        "turnover_limit": 0.08,

        "global_top_n": 300,
        "global_exit_rank_n": 450,
        "global_ewma_halflife_min": 15,

        "portfolio_rebalance_interval_min": 5,
        "rebalance_smoothing": 0.50,
        "entry_rebalance_ratio": 0.50,
        "exit_rebalance_ratio": 1.00,
        "exit_smoothing": 1.00,
        "reduce_when_target_smaller": False,
        "target_weight_band": 0.00010,

        "min_hold_minutes": 10,
        "trade_cooldown_minutes": 2,

        "use_local_buy_gate": True,
        "local_buy_top_n": 400,
        "local_buy_edge_min_bps": -5.0,

        "max_spread_bps": 999.0,
        "entry_max_spread_bps": 10.0,
        "exit_max_spread_bps": 999.0,

        "volume_cap_ratio": 0.35,

        "min_trade_notional": 10000,
        "entry_min_abs_delta_notional": 50000.0,
        "exit_min_abs_delta_notional": 0.0,

        "block_before_time": "09:35:00",
        "block_after_time": "14:50:00",

        "lot_size": 100,
        "chunksize": 1000000,

        "regime_aware_target_gross": {
            "base_target_gross": 0.80,

            "daily_loss_derisk": {
                "enabled": True,
                "daily_loss_stop": -500000,
                "cooldown_days": 2,
                "risk_off_target_scale": 0.30,
                "block_new_buy": True,
            },

            "intraday_market_regime": {
                "enabled": True,
                "market_regime_min_minutes": 30,
                "market_ret_stop": -0.015,
                "market_up_ratio_stop": 0.20,
                "market_target_scale": 0.20,
                "block_new_buy": True,
            },
        },
    },
}

cfg_path.parent.mkdir(parents=True, exist_ok=True)

with open(cfg_path, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

print("created config:", cfg_path)
print("output:", os.environ["OUT_PATH"])
PY

  python TakerOptimizer/scripts/apply_taker_position_optimizer_v2_9_local_global_risk_overlay.py \
    --config "$CFG_PATH" > "$NOHUP_LOG" 2>&1

  echo "[DONE] ${MODEL}"
  tail -n 40 "$NOHUP_LOG"
done

echo "all done"
