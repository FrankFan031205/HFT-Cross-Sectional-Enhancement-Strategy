#!/usr/bin/env bash
set -euo pipefail

H10="/mnt/data1/fwz/HFT_010-dev_fwz_outputs/TakerPipeline/zzy_pure_cs_h10_res_minute_input_csi2000_weight_v7"
H20="/mnt/data1/fwz/HFT_010-dev_fwz_outputs/TakerPipeline/zzy_pure_cs_h20_res_minute_input_csi2000_weight_v7"
MARKET="/mnt/data1/fwz/HFT_010-dev_fwz_outputs/TakerPipeline/zzy_minute_market_for_curve_tmid_csi2000_weight_v7/*.parquet"

ROOT="/mnt/data1/fwz/HFT_010-dev_fwz_outputs/final_report/v20_v16_highgross_intraday_csi2000_v7"
mkdir -p "$ROOT"

run_one () {
  local name="$1"
  local gtarget="$2"
  local gmin="$3"
  local gmax="$4"

  local TAG="pure_cs_v20_${name}_h20raw_h10rg_step10_a10_03_t02_l18_csi2000"
  local OPTDIR="$ROOT/$TAG"
  local EVALDIR="$ROOT/${TAG}_canonical_eval"
  local LOG="$ROOT/${TAG}.log"

  echo
  echo "========== $TAG =========="
  echo "gross_target=$gtarget gross_min=$gmin gross_max=$gmax"

  python TakerOptimizer/scripts/run_v16_dual_alpha_optimizer.py \
    --h10-dir "$H10" \
    --h20-dir "$H20" \
    --output-dir "$OPTDIR" \
    --tag "$TAG" \
    --rebalance-step-minutes 10 \
    --turnover-limit 0.02 \
    --alpha20-scale 0.0005 \
    --alpha10-delta-scale 0.0003 \
    --lambda-turnover 0.0018 \
    --lambda-active 0.0001 \
    --lambda-ridge 0.00001 \
    --lambda-trade-ridge 0.00001 \
    --score-mode-h20 raw \
    --score-mode-h10 rank_gate \
    --gross-target "$gtarget" \
    --gross-min "$gmin" \
    --gross-max "$gmax" \
    --cash-buffer 0.0005 \
    --solvers "OSQP,SCS,CLARABEL" \
    > "$LOG" 2>&1

  mkdir -p "$EVALDIR"

  python TakerModel/scripts/plot_nav_benchmark_warmstart_noovernight.py \
    --market-glob "$MARKET" \
    --positions "$OPTDIR/target_positions.csv" \
    --rebalance-summary "$OPTDIR/summary_by_rebalance.csv" \
    --out-dir "$EVALDIR" \
    --tag "$TAG" \
    --capital 200000000 \
    --init-gross "$gtarget" \
    --lot-size 100 \
    >> "$LOG" 2>&1

  echo "[DONE] $TAG"
}

run_one "gross098" 0.98 0.95 1.00
run_one "gross100" 1.00 0.98 1.00
