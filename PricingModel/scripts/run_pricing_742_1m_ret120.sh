#!/bin/bash
set -e

cd "$(dirname "$0")/.."
mkdir -p logs

MARKET_DIR="data/market_return_20241022_20250114_742_by_date"
FACTOR_DIR="../FactorModel/outputs"
FACTOR_GLOB="hidden_factor_*_20241022_20241122_742.csv"
RUN_NAME="pricing_1m_ret120_all_742factors"
LOG_FILE="logs/${RUN_NAME}.log"

if [ ! -d "$MARKET_DIR" ]; then
  echo "[error] market dir not found: $MARKET_DIR"
  exit 1
fi

COUNT=$(find "$FACTOR_DIR" -name "$FACTOR_GLOB" | wc -l)
echo "[info] factor files matched: $COUNT"

if [ "$COUNT" -eq 0 ]; then
  echo "[error] no factor files matched: $FACTOR_DIR/$FACTOR_GLOB"
  exit 1
fi

nohup python scripts/run_batch_pricing_from_factors.py \
  --market "$MARKET_DIR" \
  --factor_dir "$FACTOR_DIR" \
  --factor_glob "$FACTOR_GLOB" \
  --target ret_120 \
  --run_name "$RUN_NAME" \
  --calibration ols \
  --chunksize 2000000 \
  > "$LOG_FILE" 2>&1 &

echo "[started] $RUN_NAME"
echo "[log] tail -f $LOG_FILE"
