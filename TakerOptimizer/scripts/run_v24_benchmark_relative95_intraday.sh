#!/usr/bin/env bash
set -euo pipefail

H10="/mnt/data1/fwz/HFT_010-dev_fwz_outputs/TakerPipeline/zzy_pure_cs_h10_res_minute_input_csi2000_weight_v7"
H20="/mnt/data1/fwz/HFT_010-dev_fwz_outputs/TakerPipeline/zzy_pure_cs_h20_res_minute_input_csi2000_weight_v7"
MARKET="/mnt/data1/fwz/HFT_010-dev_fwz_outputs/TakerPipeline/zzy_minute_market_for_curve_tmid_csi2000_weight_v7/*.parquet"

ROOT="/mnt/data1/fwz/HFT_010-dev_fwz_outputs/final_report/v24_benchmark_relative95_intraday_csi2000_v7"
TAG="pure_cs_v24_br95soft_h20raw_h10rg_step10_a10_03_t03_l18_csi2000"

OPTDIR="$ROOT/$TAG"
EVALDIR="$ROOT/${TAG}_canonical_eval"
LOG="$ROOT/$TAG.log"

mkdir -p "$ROOT" "$EVALDIR"

echo "===== optimizer start =====" | tee "$LOG"
date | tee -a "$LOG"

python TakerOptimizer/scripts/run_v24_benchmark_relative95_optimizer.py \
  --h10-dir "$H10" \
  --h20-dir "$H20" \
  --output-dir "$OPTDIR" \
  --tag "$TAG" \
  --rebalance-step-minutes 10 \
  --turnover-limit 0.03 \
  --alpha20-scale 0.0005 \
  --alpha10-delta-scale 0.0003 \
  --lambda-turnover 0.0018 \
  --lambda-active 0.0001 \
  --lambda-ridge 0.00001 \
  --lambda-trade-ridge 0.00001 \
  --score-mode-h20 raw \
  --score-mode-h10 rank_gate \
  --gross-target 0.95 \
  --gross-min 0.93 \
  --gross-max 0.96 \
  --active-l1-limit 0.35 \
  --active-l1-relax-list "0.50,0.75,1.00,1.25" \
  --cash-buffer 0.0005 \
  --solvers "OSQP,SCS,CLARABEL" \
  >> "$LOG" 2>&1

echo "===== plot start =====" | tee -a "$LOG"

python TakerModel/scripts/plot_nav_benchmark_warmstart_noovernight.py \
  --market-glob "$MARKET" \
  --positions "$OPTDIR/target_positions.csv" \
  --rebalance-summary "$OPTDIR/summary_by_rebalance.csv" \
  --out-dir "$EVALDIR" \
  --tag "$TAG" \
  --capital 200000000 \
  --init-gross 0.95 \
  --lot-size 100 \
  >> "$LOG" 2>&1

echo "===== DONE =====" | tee -a "$LOG"
echo "OPTDIR=$OPTDIR" | tee -a "$LOG"
echo "EVALDIR=$EVALDIR" | tee -a "$LOG"
