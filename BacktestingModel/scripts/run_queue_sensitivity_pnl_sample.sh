#!/usr/bin/env bash
set -e

TAG="mlp2_h60_202410_100"
BASE_CONFIG="config/backtest.yaml"

mkdir -p outputs/config_runs outputs/fills outputs/trades outputs/metrics logs

echo "===== queue sensitivity pnl sample start: $(date) ====="

for mult in 0.0 0.05 0.1 0.25 0.5 1.0
do
  mtag=$(echo "$mult" | sed 's/\./p/g')

  fill_path="outputs/fills/fills_queue_mult_${mtag}_sample_${TAG}.csv"
  enriched_path="outputs/fills/fills_queue_mult_${mtag}_sample_enriched_${TAG}.csv"
  pnl_path="outputs/trades/trades_pnl_queue_mult_${mtag}_sample_${TAG}.csv"
  daily_path="outputs/metrics/daily_pnl_queue_mult_${mtag}_sample_${TAG}.csv"
  summary_path="outputs/metrics/summary_queue_mult_${mtag}_sample_${TAG}.csv"
  factor_path="outputs/metrics/factor_pnl_queue_mult_${mtag}_sample_${TAG}.csv"
  run_config="outputs/config_runs/backtest_queue_mult_${mtag}_pnl_sample_${TAG}.yaml"

  echo ""
  echo "===== queue multiplier ${mult} pnl ====="

  if [ ! -f "$fill_path" ]; then
    echo "missing fill file: $fill_path"
    continue
  fi

  python - <<PY
import yaml

base = "${BASE_CONFIG}"
out = "${run_config}"

with open(base, "r") as f:
    cfg = yaml.safe_load(f)

cfg["signal"] = {
    "col": "hidden_factor_mlp2_h60",
    "n_bins": 5
}

cfg["output"]["enriched_fill_path"] = "${enriched_path}"
cfg["output"]["pnl_path"] = "${pnl_path}"
cfg["output"]["daily_pnl_path"] = "${daily_path}"
cfg["output"]["summary_path"] = "${summary_path}"
cfg["output"]["factor_pnl_path"] = "${factor_path}"

with open(out, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

print("wrote", out)
PY

  rows=$(python - <<PY
import pandas as pd
path = "${fill_path}"
df = pd.read_csv(path)
print(len(df))
PY
)

  if [ "$rows" = "0" ]; then
    echo "fill file is empty, skip pnl"
    continue
  fi

  python scripts/enrich_fills_with_factor.py \
    --config "$run_config" \
    --fills "$fill_path" \
    --output "$enriched_path"

  python scripts/run_attention_pnl_backtest.py \
    --config "$run_config" \
    --fills "$enriched_path" \
    --output "$pnl_path"

done

echo "===== queue sensitivity pnl sample finished: $(date) ====="
