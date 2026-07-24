#!/usr/bin/env bash
set -uo pipefail

H10="/mnt/data1/fwz/HFT_010-dev_fwz_outputs/TakerPipeline/zzy_lasso0707_h10_res_input_csi2000_weight_v7_compat"
H20="/mnt/data1/fwz/HFT_010-dev_fwz_outputs/TakerPipeline/zzy_lasso0707_h20_res_input_csi2000_weight_v7_compat"
MARKET="/mnt/data1/fwz/HFT_010-dev_fwz_outputs/TakerPipeline/zzy_minute_market_for_curve_tmid_csi2000_weight_v7/*.parquet"

ROOT="/mnt/data1/fwz/HFT_010-dev_fwz_outputs/final_report/v17_lasso0707_delta_strength_csi2000_v7"
mkdir -p "$ROOT"

run_one () {
  local name="$1"
  local mode10="$2"
  local a10="$3"
  local tlim="$4"
  local lturn="$5"

  local TAG="pure_cs_v17_${name}_csi2000"
  local OPTDIR="$ROOT/$TAG"
  local EVALDIR="$ROOT/${TAG}_canonical_eval"
  local LOG="$ROOT/${TAG}.log"

  echo
  echo "========== $TAG =========="
  echo "h20=raw h10=$mode10 a10=$a10 tlim=$tlim lturn=$lturn"

  python TakerOptimizer/scripts/run_v16_dual_alpha_optimizer.py \
    --h10-dir "$H10" \
    --h20-dir "$H20" \
    --output-dir "$OPTDIR" \
    --tag "$TAG" \
    --rebalance-step-minutes 10 \
    --turnover-limit "$tlim" \
    --alpha20-scale 0.0005 \
    --alpha10-delta-scale "$a10" \
    --lambda-turnover "$lturn" \
    --lambda-active 0.0001 \
    --lambda-ridge 0.00001 \
    --lambda-trade-ridge 0.00001 \
    --score-mode-h20 raw \
    --score-mode-h10 "$mode10" \
    --solvers "OSQP,SCS,CLARABEL" \
    > "$LOG" 2>&1

  if [ $? -ne 0 ]; then
    echo "[FAILED optimizer] $TAG, see $LOG"
    return 0
  fi

  mkdir -p "$EVALDIR"

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

  echo "[DONE] $TAG"
}

# baseline-like but stronger h10
run_one "h20raw_h10rg_a10_05_t02_l12"   rank_gate 0.0005 0.02 0.0012
run_one "h20raw_h10rg_a10_08_t03_l10"   rank_gate 0.0008 0.03 0.0010

# 放松 h10 gate
run_one "h20raw_h10rank_a10_05_t02_l12" rank      0.0005 0.02 0.0012
run_one "h20raw_h10rank_a10_08_t03_l10" rank      0.0008 0.03 0.0010

# 更激进一版，看有没有 alpha 表达
run_one "h20raw_h10raw_a10_05_t02_l12"  raw       0.0005 0.02 0.0012
