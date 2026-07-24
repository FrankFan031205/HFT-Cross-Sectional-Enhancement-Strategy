#!/usr/bin/env bash
set -euo pipefail

H10="/mnt/data1/fwz/HFT_010-dev_fwz_outputs/TakerPipeline/zzy_pure_cs_h10_res_minute_input_csi2000_weight_v7"
H20="/mnt/data1/fwz/HFT_010-dev_fwz_outputs/TakerPipeline/zzy_pure_cs_h20_res_minute_input_csi2000_weight_v7"
MARKET="/mnt/data1/fwz/HFT_010-dev_fwz_outputs/TakerPipeline/zzy_minute_market_for_curve_tmid_csi2000_weight_v7/*.parquet"

ROOT="/mnt/data1/fwz/HFT_010-dev_fwz_outputs/final_report/v25b_core_active_sleeve_csi2000_v7"
TAG="pure_cs_v25b_core095_active025_grosspen20_target20_exec10_t035_csi2000"

OPTDIR="$ROOT/$TAG"
EVALDIR="$ROOT/${TAG}_canonical_eval"
LOG="$ROOT/$TAG.log"

mkdir -p "$ROOT" "$EVALDIR"

echo "===== v25b optimizer start =====" | tee "$LOG"
date | tee -a "$LOG"

python TakerOptimizer/scripts/run_v25_core_active_sleeve_optimizer.py \
  --h10-dir "$H10" \
  --h20-dir "$H20" \
  --output-dir "$OPTDIR" \
  --tag "$TAG" \
  --stock-core-gross 0.95 \
  --active-l1-budget 0.25 \
  --target-top-frac 0.10 \
  --target-bottom-frac 0.10 \
  --target-weighting benchmark \
  --target-update-step-minutes 20 \
  --rebalance-step-minutes 10 \
  --turnover-limit 0.035 \
  --alpha10-delta-scale 0.0003 \
  --lambda-track 1.0 \
  --lambda-gross 20.0 \
  --lambda-turnover 0.0018 \
  --lambda-trade-ridge 0.00001 \
  --score-mode-h20 raw \
  --score-mode-h10 rank_gate \
  --single-name-cap 0.008 \
  --cash-buffer 0.0005 \
  --solvers "OSQP,SCS,CLARABEL" \
  >> "$LOG" 2>&1

echo "===== canonical eval start =====" | tee -a "$LOG"
date | tee -a "$LOG"

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
date | tee -a "$LOG"
