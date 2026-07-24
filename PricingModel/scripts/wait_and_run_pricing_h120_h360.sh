#!/usr/bin/env bash
set -euo pipefail

cd /mnt/data1/fwz/HFT_010-dev_fwz/PricingModel
mkdir -p logs

# ========== paths ==========

MARKET_742_DIR="data/market_return_20241022_20250114_742_by_date"
MARKET_CSI1000_DIR="data/market_return_20241022_20250114_csi1000_by_date"

FACTOR_DIR="../FactorModel/outputs"

FACTOR_742_H360="${FACTOR_DIR}/hidden_factor_cs_dl_feature_transformer_h360_20241022_20250114_742_50f.csv"
FACTOR_CSI1000_H120="${FACTOR_DIR}/hidden_factor_cs_dl_feature_transformer_h120_20241022_20250114_csi1000_50f.csv"
FACTOR_CSI1000_H360="${FACTOR_DIR}/hidden_factor_cs_dl_feature_transformer_h360_20241022_20250114_csi1000_50f.csv"

SLEEP_SEC=300

# ========== helper functions ==========

wait_for_file() {
  local f="$1"
  local name="$2"

  echo "[wait] $name: $f"

  while true; do
    if [ -f "$f" ]; then
      # 避免文件还在写入，要求 size 5分钟内稳定
      size1=$(stat -c%s "$f")
      sleep 30
      size2=$(stat -c%s "$f")

      if [ "$size1" = "$size2" ] && [ "$size1" -gt 1000000 ]; then
        echo "[ready] $name exists and size stable: $f"
        ls -lh "$f"
        break
      else
        echo "[not ready] $name exists but still changing: size1=$size1 size2=$size2"
      fi
    else
      echo "[missing] $name not found yet"
    fi

    sleep "$SLEEP_SEC"
  done
}

wait_for_market_col() {
  local market_dir="$1"
  local pattern="$2"
  local col="$3"
  local name="$4"

  echo "[wait] $name needs column: $col in $market_dir/$pattern"

  while true; do
    python - <<PY
from pathlib import Path
import pandas as pd
import sys

market_dir = Path("$market_dir")
files = sorted(market_dir.glob("$pattern"))

if len(files) == 0:
    print("[missing] no market files found:", market_dir / "$pattern")
    sys.exit(1)

bad = []
for p in files:
    try:
        cols = pd.read_csv(p, nrows=1).columns.tolist()
    except Exception as e:
        print("[bad] cannot read", p, e)
        sys.exit(1)

    if "$col" not in cols:
        bad.append(p.name)

print("num files:", len(files))
print("first:", files[0].name)
print("last :", files[-1].name)

if bad:
    print("[not ready] files missing $col:", len(bad))
    print("examples:", bad[:10])
    sys.exit(1)

print("[ready] all files have column $col")
sys.exit(0)
PY

    if [ $? -eq 0 ]; then
      echo "[ready] $name market column $col ready"
      break
    fi

    echo "[sleep] wait ${SLEEP_SEC}s for $name market column $col"
    sleep "$SLEEP_SEC"
  done
}

run_pricing_if_needed() {
  local run_name="$1"
  local market_dir="$2"
  local factor_glob="$3"
  local target="$4"
  local log_file="logs/${run_name}.log"
  local metrics_file="output/${run_name}/reports/pricing_metrics_by_signal.csv"

  if [ -f "$metrics_file" ]; then
    echo "[skip] pricing already done: $run_name"
    echo "       $metrics_file"
    return
  fi

  echo "[run] $run_name"
  echo "[log] $log_file"

  python scripts/run_batch_pricing_from_factors.py \
    --market "$market_dir" \
    --factor_dir "$FACTOR_DIR" \
    --factor_glob "$factor_glob" \
    --target "$target" \
    --run_name "$run_name" \
    --calibration rank_guarded \
    --chunksize 2000000 \
    > "$log_file" 2>&1

  echo "[done] $run_name"
  echo "[metrics]"
  cat "$metrics_file"
  echo
}

# ========== wait for inputs ==========

echo "========== wait for factor files =========="

wait_for_file "$FACTOR_742_H360" "742 h360 factor"
wait_for_file "$FACTOR_CSI1000_H120" "csi1000 h120 factor"
wait_for_file "$FACTOR_CSI1000_H360" "csi1000 h360 factor"

echo "========== wait for market target columns =========="

# h120 pricing needs ret_120
wait_for_market_col "$MARKET_CSI1000_DIR" "market_return_*_csi1000.csv" "ret_120" "csi1000 h120"

# h360 pricing needs ret_360
wait_for_market_col "$MARKET_742_DIR" "market_return_*_742.csv" "ret_360" "742 h360"
wait_for_market_col "$MARKET_CSI1000_DIR" "market_return_*_csi1000.csv" "ret_360" "csi1000 h360"

# ========== run pricing ==========

echo "========== run pricing =========="

run_pricing_if_needed \
  "pricing_insample_20241022_20250114_742_feature_transformer_h360_50f_ret360_rank_guarded" \
  "$MARKET_742_DIR" \
  "hidden_factor_cs_dl_feature_transformer_h360_20241022_20250114_742_50f.csv" \
  "ret_360"

run_pricing_if_needed \
  "pricing_insample_20241022_20250114_csi1000_feature_transformer_h120_50f_ret120_rank_guarded" \
  "$MARKET_CSI1000_DIR" \
  "hidden_factor_cs_dl_feature_transformer_h120_20241022_20250114_csi1000_50f.csv" \
  "ret_120"

run_pricing_if_needed \
  "pricing_insample_20241022_20250114_csi1000_feature_transformer_h360_50f_ret360_rank_guarded" \
  "$MARKET_CSI1000_DIR" \
  "hidden_factor_cs_dl_feature_transformer_h360_20241022_20250114_csi1000_50f.csv" \
  "ret_360"

echo "========== all pricing done =========="

echo "742 h360:"
echo "output/pricing_insample_20241022_20250114_742_feature_transformer_h360_50f_ret360_rank_guarded/pricing/cs_dl_feature_transformer_h360/priced_dataset.csv"

echo "csi1000 h120:"
echo "output/pricing_insample_20241022_20250114_csi1000_feature_transformer_h120_50f_ret120_rank_guarded/pricing/cs_dl_feature_transformer_h120/priced_dataset.csv"

echo "csi1000 h360:"
echo "output/pricing_insample_20241022_20250114_csi1000_feature_transformer_h360_50f_ret360_rank_guarded/pricing/cs_dl_feature_transformer_h360/priced_dataset.csv"
