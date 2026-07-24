#!/usr/bin/env bash
set -e

cd /home/fwz/projects/HFT_010-dev_fwz/BacktestingModel

TAG="mlp2_h60_202410_100"
BASE="outputs/trades/trades_pnl_queue_mult_0p05_${TAG}.csv"

mkdir -p logs outputs/trades outputs/portfolio

for inv in 0 1000 5000 10000
do
  if [ "$inv" = "0" ]; then
    cap=5000
  else
    cap=$((inv * 5))
  fi

  echo ""
  echo "===== inventory scenario inv=${inv}, cap=${cap}, T+1 ====="

  OUT="outputs/trades/trades_pnl_queue_mult_0p05_inventory${inv}_tplus1_${TAG}.csv"
  SKIP="outputs/trades/skipped_queue_mult_0p05_inventory${inv}_tplus1_${TAG}.csv"
  PREFIX="outputs/portfolio/portfolio_replay_queue_0p05_inventory${inv}_tplus1_${TAG}"

  python scripts/filter_inventory_constraints.py \
    --trades "$BASE" \
    --output "$OUT" \
    --skipped-output "$SKIP" \
    --initial-position-per-symbol "$inv" \
    --max-position-per-symbol "$cap" \
    --tplus1 1

  python scripts/run_portfolio_replay.py \
    --trades "$OUT" \
    --model "queue_0p05_inventory${inv}_tplus1" \
    --out-prefix "$PREFIX" \
    --capital 0 \
    --initial-position-per-symbol "$inv" \
    --allow-short 0 \
    --record-every 10000

done

echo "===== inventory scenarios finished ====="
