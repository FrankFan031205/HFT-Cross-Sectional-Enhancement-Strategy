#!/usr/bin/env bash
set -e

cd /home/fwz/projects/HFT_010-dev_fwz/BacktestingModel

TAG="mlp2_h60_202410_100"

echo "===== watcher start: $(date) ====="

echo "[0] wait for current run_fill_simulation.py to finish"

while pgrep -af "python .*scripts/run_fill_simulation.py" >/dev/null
do
  echo "fill simulation still running..."
  pgrep -af "python .*scripts/run_fill_simulation.py"
  sleep 300
done

echo "current fill simulation finished: $(date)"

FILL_PATH="outputs/fills/fills_touched_${TAG}.csv"
ENRICHED_PATH="outputs/fills/fills_touched_enriched_${TAG}.csv"
PNL_PATH="outputs/trades/trades_pnl_touched_${TAG}.csv"

echo "[1] check touched fill file"
ls -lh "$FILL_PATH"

python - <<PY
import pandas as pd
path = "${FILL_PATH}"
df = pd.read_csv(path, low_memory=False)
print("fills shape:", df.shape)
if len(df):
    print("date range:", df["decision_time"].min(), "->", df["decision_time"].max())
    print("num securities:", df["securityid"].nunique())
    print(df["side"].value_counts())
PY

echo "[2] enrich touched fills"

python scripts/enrich_fills_with_factor.py \
  --config config/backtest.yaml \
  --fills "$FILL_PATH" \
  --output "$ENRICHED_PATH"

echo "[3] run touched PnL"

python scripts/run_attention_pnl_backtest.py \
  --config config/backtest.yaml \
  --fills "$ENRICHED_PATH" \
  --output "$PNL_PATH"

echo "[4] prepare queue=0.05 config"

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

echo "[5] run full queue=0.05 fill simulation"

python scripts/run_fill_simulation.py --config "$QCFG"

echo "[6] enrich queue=0.05 fills"

python scripts/enrich_fills_with_factor.py \
  --config "$QCFG" \
  --fills "outputs/fills/fills_queue_mult_0p05_${TAG}.csv" \
  --output "outputs/fills/fills_queue_mult_0p05_enriched_${TAG}.csv"

echo "[7] run queue=0.05 PnL"

python scripts/run_attention_pnl_backtest.py \
  --config "$QCFG" \
  --fills "outputs/fills/fills_queue_mult_0p05_enriched_${TAG}.csv" \
  --output "outputs/trades/trades_pnl_queue_mult_0p05_${TAG}.csv"

echo "[8] print summaries"

python - <<PY
import pandas as pd

files = [
    "outputs/metrics/summary_touched_${TAG}.csv",
    "outputs/metrics/summary_queue_mult_0p05_${TAG}.csv",
]

for f in files:
    print("\\n==========", f, "==========")
    df = pd.read_csv(f)
    print(df.T.to_string())
PY

echo "===== watcher finished: $(date) ====="
