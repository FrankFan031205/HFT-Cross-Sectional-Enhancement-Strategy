#!/usr/bin/env bash
set -euo pipefail

cd /home/fwz/projects/HFT_010-dev_fwz/BacktestingModel

TAG="mlp2_h60_202410_100"
QUOTE_PATH="../MarketMakingModel/outputs/quote_decisions/quote_decisions_mlp2_h60_202410_100_v5.csv"

LOG_DIR="logs"
mkdir -p "$LOG_DIR" outputs/fills outputs/trades outputs/metrics outputs/cache outputs/config_runs

MAIN_LOG="$LOG_DIR/weekend_100stock_after_current.log"

exec > >(tee -a "$MAIN_LOG") 2>&1

echo "===== weekend 100-stock backtest watcher started: $(date) ====="

LOCK_FILE="outputs/cache/weekend_100stock_after_current.lock"
exec 9>"$LOCK_FILE"

if ! flock -n 9; then
  echo "Another weekend watcher is already running. Exit."
  exit 1
fi

echo "[0] waiting for existing fill simulation process to finish"

while true; do
  running=$(pgrep -af "python .*scripts/run_fill_simulation.py" || true)

  if [ -z "$running" ]; then
    echo "No existing run_fill_simulation.py process found."
    break
  fi

  echo "Existing fill simulation still running:"
  echo "$running"
  echo "sleep 300 seconds..."
  sleep 300
done

echo "[1] standardize backtest config for v5 input and standard output names"

python - <<PY
import yaml

path = "config/backtest.yaml"

with open(path, "r") as f:
    cfg = yaml.safe_load(f)

tag = "${TAG}"

cfg["input"]["quote_decision_path"] = "${QUOTE_PATH}"

cfg["fill_model"]["mode"] = "touched_trade"
cfg["fill_model"]["queue_ahead_multiplier"] = 0.0

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

print("config standardized")
print("quote_decision_path:", cfg["input"]["quote_decision_path"])
print("fill_path:", cfg["output"]["fill_path"])
PY

FILL_PATH="outputs/fills/fills_touched_${TAG}.csv"
ENRICHED_PATH="outputs/fills/fills_touched_enriched_${TAG}.csv"
PNL_PATH="outputs/trades/trades_pnl_touched_${TAG}.csv"

if [ ! -f "$FILL_PATH" ]; then
  echo "[2] touched fill file not found, run full touched fill simulation"
  python scripts/run_fill_simulation.py --config config/backtest.yaml
else
  echo "[2] touched fill file already exists: $FILL_PATH"
  ls -lh "$FILL_PATH"
fi

echo "[3] check touched fills"

python - <<PY
import pandas as pd
path = "${FILL_PATH}"
df = pd.read_csv(path, low_memory=False)
print("fills shape:", df.shape)
if len(df):
    print("decision_time range:", df["decision_time"].min(), "->", df["decision_time"].max())
    print("num securities:", df["securityid"].nunique())
    print("side counts:")
    print(df["side"].value_counts())
else:
    print("WARNING: fills file is empty")
PY

echo "[4] enrich fills with hidden_factor_mlp2_h60"

python scripts/enrich_fills_with_factor.py \
  --config config/backtest.yaml \
  --fills "$FILL_PATH" \
  --output "$ENRICHED_PATH"

echo "[5] run touched 60s markout PnL with A-share single-fill fees"

python scripts/run_attention_pnl_backtest.py \
  --config config/backtest.yaml \
  --fills "$ENRICHED_PATH" \
  --output "$PNL_PATH"

echo "[6] print touched summary"

python - <<PY
import pandas as pd

summary_path = "outputs/metrics/summary_touched_${TAG}.csv"
factor_path = "outputs/metrics/factor_pnl_touched_${TAG}.csv"
daily_path = "outputs/metrics/daily_pnl_touched_${TAG}.csv"

print("\\n===== touched summary =====")
summary = pd.read_csv(summary_path)
print(summary.T.to_string())

print("\\n===== touched daily pnl head =====")
daily = pd.read_csv(daily_path)
print(daily.head(20).to_string(index=False))

print("\\n===== touched factor pnl =====")
factor = pd.read_csv(factor_path)
pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 200)
print(factor.to_string(index=False))
PY

if [ "${RUN_QUEUE_005:-0}" = "1" ]; then
  echo "[7] RUN_QUEUE_005=1, run full queue_multiplier=0.05 sensitivity"

  QCFG="outputs/config_runs/backtest_queue_mult_0p05_${TAG}.yaml"

  python - <<PY
import yaml

base = "config/backtest.yaml"
out = "${QCFG}"
tag = "${TAG}"

with open(base, "r") as f:
    cfg = yaml.safe_load(f)

cfg["fill_model"]["mode"] = "queue_aware_trade"
cfg["fill_model"]["queue_ahead_multiplier"] = 0.05

cfg["output"]["fill_path"] = f"outputs/fills/fills_queue_mult_0p05_{tag}.csv"
cfg["output"]["enriched_fill_path"] = f"outputs/fills/fills_queue_mult_0p05_enriched_{tag}.csv"
cfg["output"]["pnl_path"] = f"outputs/trades/trades_pnl_queue_mult_0p05_{tag}.csv"
cfg["output"]["daily_pnl_path"] = f"outputs/metrics/daily_pnl_queue_mult_0p05_{tag}.csv"
cfg["output"]["summary_path"] = f"outputs/metrics/summary_queue_mult_0p05_{tag}.csv"
cfg["output"]["factor_pnl_path"] = f"outputs/metrics/factor_pnl_queue_mult_0p05_{tag}.csv"

with open(out, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

print("wrote", out)
print("queue fill path:", cfg["output"]["fill_path"])
PY

  python scripts/run_fill_simulation.py --config "$QCFG"

  python scripts/enrich_fills_with_factor.py \
    --config "$QCFG" \
    --fills "outputs/fills/fills_queue_mult_0p05_${TAG}.csv" \
    --output "outputs/fills/fills_queue_mult_0p05_enriched_${TAG}.csv"

  python scripts/run_attention_pnl_backtest.py \
    --config "$QCFG" \
    --fills "outputs/fills/fills_queue_mult_0p05_enriched_${TAG}.csv" \
    --output "outputs/trades/trades_pnl_queue_mult_0p05_${TAG}.csv"

  echo "[8] print queue 0.05 summary"

  python - <<PY
import pandas as pd

summary_path = "outputs/metrics/summary_queue_mult_0p05_${TAG}.csv"
factor_path = "outputs/metrics/factor_pnl_queue_mult_0p05_${TAG}.csv"

print("\\n===== queue 0.05 summary =====")
summary = pd.read_csv(summary_path)
print(summary.T.to_string())

print("\\n===== queue 0.05 factor pnl =====")
factor = pd.read_csv(factor_path)
pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 200)
print(factor.to_string(index=False))
PY
else
  echo "[7] skip queue_multiplier=0.05 full run. Set RUN_QUEUE_005=1 to enable."
fi

echo "===== weekend 100-stock backtest watcher finished: $(date) ====="
