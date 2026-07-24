#!/usr/bin/env bash
set -e

TAG="mlp2_h60_202410_100"
BASE_CONFIG="config/backtest.yaml"

mkdir -p outputs/config_runs outputs/fills outputs/cache outputs/metrics logs

echo "===== queue sensitivity sample start: $(date) ====="

for mult in 0.0 0.05 0.1 0.25 0.5 1.0
do
  mtag=$(echo "$mult" | sed 's/\./p/g')
  run_config="outputs/config_runs/backtest_queue_mult_${mtag}_sample_${TAG}.yaml"

  echo ""
  echo "===== queue multiplier ${mult} ====="

  python - <<PY
import yaml

base = "${BASE_CONFIG}"
out = "${run_config}"
tag = "${TAG}"
mult = float("${mult}")
mtag = "${mtag}"

with open(base, "r") as f:
    cfg = yaml.safe_load(f)

cfg["fill_model"]["mode"] = "queue_aware_trade"
cfg["fill_model"]["queue_ahead_multiplier"] = mult

cfg["signal"] = {
    "col": "hidden_factor_mlp2_h60",
    "n_bins": 5
}

cfg["output"]["fill_path"] = f"outputs/fills/fills_queue_mult_{mtag}_sample_{tag}.csv"
cfg["output"]["trade_cache_path"] = f"outputs/cache/raw_trades_sample_{tag}.csv"
cfg["output"]["snapshot_cache_path"] = f"outputs/cache/snapshot_state_for_quotes_sample_{tag}.csv"

with open(out, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

print("wrote", out)
print("fill_path:", cfg["output"]["fill_path"])
PY

  python scripts/run_fill_simulation.py \
    --config "$run_config" \
    --max-quotes 5000

done

echo "===== queue sensitivity sample finished: $(date) ====="
