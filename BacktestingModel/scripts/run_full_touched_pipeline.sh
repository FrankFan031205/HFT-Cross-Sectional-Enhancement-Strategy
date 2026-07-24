#!/usr/bin/env bash
set -e

echo "===== start full touched pipeline: $(date) ====="

echo "[0] standardize config"
python - <<'PY'
import yaml

path = "config/backtest.yaml"

with open(path, "r") as f:
    cfg = yaml.safe_load(f)

tag = "mlp2_h60_202410_100"

cfg["fill_model"]["mode"] = "touched_trade"

cfg["signal"] = {
    "col": "hidden_factor_mlp2_h60",
    "n_bins": 5
}

cfg["output"]["fill_path"] = f"outputs/fills/fills_touched_{tag}.csv"
cfg["output"]["enriched_fill_path"] = f"outputs/fills/fills_touched_enriched_{tag}.csv"
cfg["output"]["pnl_path"] = f"outputs/trades/trades_pnl_touched_{tag}.csv"
cfg["output"]["daily_pnl_path"] = f"outputs/metrics/daily_pnl_touched_{tag}.csv"
cfg["output"]["summary_path"] = f"outputs/metrics/summary_touched_{tag}.csv"
cfg["output"]["factor_pnl_path"] = f"outputs/metrics/factor_pnl_touched_{tag}.csv"
cfg["output"]["trade_cache_path"] = f"outputs/cache/raw_trades_{tag}.csv"
cfg["output"]["snapshot_cache_path"] = f"outputs/cache/snapshot_state_for_quotes_{tag}.csv"

cfg["fee"] = {
    "mode": "single_fill_side_specific",
    "commission_rate": 0.00005,
    "transfer_fee_rate": 0.00001,
    "handling_fee_rate": 0.0000341,
    "regulatory_fee_rate": 0.00002,
    "stamp_duty_rate": 0.0005
}

with open(path, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

print("config ready")
PY

echo "[1] remove old cache"
rm -f outputs/cache/raw_trades_mlp2_h60_202410_100.csv
rm -f outputs/cache/snapshot_state_for_quotes_mlp2_h60_202410_100.csv

echo "[2] run full touched fill simulation"
python scripts/run_fill_simulation.py --config config/backtest.yaml

echo "[3] enrich fills with factor"
python scripts/enrich_fills_with_factor.py \
  --config config/backtest.yaml \
  --fills outputs/fills/fills_touched_mlp2_h60_202410_100.csv \
  --output outputs/fills/fills_touched_enriched_mlp2_h60_202410_100.csv

echo "[4] run factor PnL backtest"
python scripts/run_attention_pnl_backtest.py \
  --config config/backtest.yaml \
  --fills outputs/fills/fills_touched_enriched_mlp2_h60_202410_100.csv \
  --output outputs/trades/trades_pnl_touched_mlp2_h60_202410_100.csv

echo "===== finished full touched pipeline: $(date) ====="
