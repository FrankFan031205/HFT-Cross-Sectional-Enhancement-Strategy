#!/usr/bin/env bash
set -euo pipefail

cd /mnt/data1/fwz/HFT_010-dev_fwz/PricingModel
mkdir -p logs

SLEEP_SEC=300

MARKET_742_DIR="data/market_return_20241022_20250114_742_by_date"
MARKET_CSI1000_DIR="data/market_return_20241022_20250114_csi1000_by_date"

FACTOR_742_H360="../FactorModel/outputs/hidden_factor_cs_dl_feature_transformer_h360_20241022_20250114_742_50f.csv"
FACTOR_CSI1000_H120="../FactorModel/outputs/hidden_factor_cs_dl_feature_transformer_h120_20241022_20250114_csi1000_50f.csv"
FACTOR_CSI1000_H360="../FactorModel/outputs/hidden_factor_cs_dl_feature_transformer_h360_20241022_20250114_csi1000_50f.csv"

PRICING_SCRIPT="scripts/run_one_signal_pricing_minute_slim.py"


wait_for_file_stable() {
  local f="$1"
  local name="$2"

  echo "[wait file] $name: $f"

  while true; do
    if [ -f "$f" ]; then
      size1=$(stat -c%s "$f")
      sleep 30
      size2=$(stat -c%s "$f")

      if [ "$size1" = "$size2" ] && [ "$size1" -gt 1000000 ]; then
        echo "[ready file] $name"
        ls -lh "$f"
        break
      fi

      echo "[not ready] $name still writing: $size1 -> $size2"
    else
      echo "[missing] $name"
    fi

    sleep "$SLEEP_SEC"
  done
}


wait_until_no_ret360_writer() {
  echo "[wait] ret360 add scripts finish"

  while true; do
    if ps -ef | grep -E "add_ret360_to.*market_by_date.py|build_.*ret360" | grep -v grep; then
      echo "[still running] ret360 writer exists, sleep ${SLEEP_SEC}s"
      sleep "$SLEEP_SEC"
    else
      echo "[ready] no ret360 writer process"
      break
    fi
  done
}


wait_market_has_col_all_files() {
  local market_dir="$1"
  local pattern="$2"
  local col="$3"
  local name="$4"

  echo "[wait market col] $name needs $col"

  while true; do
    if python - <<PY
from pathlib import Path
import pandas as pd
import sys

market_dir = Path("$market_dir")
files = sorted(market_dir.glob("$pattern"))

if not files:
    print("[missing] no files:", market_dir / "$pattern")
    sys.exit(1)

bad = []
unreadable = []

for p in files:
    try:
        cols = pd.read_csv(p, nrows=1).columns.tolist()
    except Exception as e:
        unreadable.append((p.name, str(e)))
        continue

    if "$col" not in cols:
        bad.append(p.name)

print("market_dir:", market_dir)
print("num_files:", len(files))
print("first:", files[0].name)
print("last :", files[-1].name)
print("missing_$col:", len(bad))
print("unreadable:", len(unreadable))

if bad:
    print("bad examples:", bad[:10])
if unreadable:
    print("unreadable examples:", unreadable[:3])

if bad or unreadable:
    sys.exit(1)

sys.exit(0)
PY
    then
      echo "[ready market col] $name has $col in all files"
      break
    else
      echo "[not ready] $name missing $col, sleep ${SLEEP_SEC}s"
      sleep "$SLEEP_SEC"
    fi
  done
}


run_pricing() {
  local run_name="$1"
  local market_dir="$2"
  local market_glob="$3"
  local factor_path="$4"
  local target="$5"
  local factor_col="$6"
  local signal="$7"

  local metrics="output/${run_name}/reports/pricing_metrics_by_signal.csv"
  local priced="output/${run_name}/pricing/${signal}/priced_dataset.csv"
  local log_file="logs/${run_name}.log"

  if [ -f "$metrics" ] && [ -f "$priced" ]; then
    echo "[skip] already done: $run_name"
    echo "[metrics] $metrics"
    cat "$metrics"
    return
  fi

  echo "========== run pricing: $run_name =========="
  echo "[log] $log_file"

  python "$PRICING_SCRIPT" \
    --market_dir "$market_dir" \
    --market_glob "$market_glob" \
    --factor "$factor_path" \
    --target "$target" \
    --factor_col "$factor_col" \
    --signal "$signal" \
    --run_name "$run_name" \
    --calibration rank_guarded \
    --chunksize 2000000 \
    > "$log_file" 2>&1

  echo "[done] $run_name"
  echo "[metrics]"
  cat "$metrics"
  echo
  echo "[priced]"
  ls -lh "$priced"
}


echo "========== check pricing script =========="

if [ ! -f "$PRICING_SCRIPT" ]; then
  echo "[error] missing $PRICING_SCRIPT"
  echo "先创建 minute-slim pricing 脚本，再跑这个总控。"
  exit 1
fi


echo "========== wait factor csv files =========="

wait_for_file_stable "$FACTOR_742_H360" "742 h360 factor"
wait_for_file_stable "$FACTOR_CSI1000_H120" "csi1000 h120 factor"
wait_for_file_stable "$FACTOR_CSI1000_H360" "csi1000 h360 factor"


echo "========== wait ret360 writer done =========="

wait_until_no_ret360_writer


echo "========== wait market columns ready =========="

wait_market_has_col_all_files "$MARKET_742_DIR" "market_return_*_742.csv" "ret_360" "742 h360 market"
wait_market_has_col_all_files "$MARKET_CSI1000_DIR" "market_return_*_csi1000.csv" "ret_120" "csi1000 h120 market"
wait_market_has_col_all_files "$MARKET_CSI1000_DIR" "market_return_*_csi1000.csv" "ret_360" "csi1000 h360 market"


echo "========== run three minute-slim pricing jobs =========="

run_pricing \
  "pricing_minute_insample_20241022_20250114_742_feature_transformer_h360_50f_ret360_rank_guarded" \
  "$MARKET_742_DIR" \
  "market_return_*_742.csv" \
  "$FACTOR_742_H360" \
  "ret_360" \
  "hidden_factor_cs_dl_feature_transformer_h360" \
  "cs_dl_feature_transformer_h360"

run_pricing \
  "pricing_minute_insample_20241022_20250114_csi1000_feature_transformer_h120_50f_ret120_rank_guarded" \
  "$MARKET_CSI1000_DIR" \
  "market_return_*_csi1000.csv" \
  "$FACTOR_CSI1000_H120" \
  "ret_120" \
  "hidden_factor_cs_dl_feature_transformer_h120" \
  "cs_dl_feature_transformer_h120"

run_pricing \
  "pricing_minute_insample_20241022_20250114_csi1000_feature_transformer_h360_50f_ret360_rank_guarded" \
  "$MARKET_CSI1000_DIR" \
  "market_return_*_csi1000.csv" \
  "$FACTOR_CSI1000_H360" \
  "ret_360" \
  "hidden_factor_cs_dl_feature_transformer_h360" \
  "cs_dl_feature_transformer_h360"


echo "========== all done =========="

echo "742 h360:"
echo "output/pricing_minute_insample_20241022_20250114_742_feature_transformer_h360_50f_ret360_rank_guarded/pricing/cs_dl_feature_transformer_h360/priced_dataset.csv"

echo "csi1000 h120:"
echo "output/pricing_minute_insample_20241022_20250114_csi1000_feature_transformer_h120_50f_ret120_rank_guarded/pricing/cs_dl_feature_transformer_h120/priced_dataset.csv"

echo "csi1000 h360:"
echo "output/pricing_minute_insample_20241022_20250114_csi1000_feature_transformer_h360_50f_ret360_rank_guarded/pricing/cs_dl_feature_transformer_h360/priced_dataset.csv"
