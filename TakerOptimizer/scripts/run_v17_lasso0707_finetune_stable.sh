#!/usr/bin/env bash
set -uo pipefail

H10="/mnt/data1/fwz/HFT_010-dev_fwz_outputs/TakerPipeline/zzy_lasso0707_h10_res_input_csi2000_weight_v7_compat"
H20="/mnt/data1/fwz/HFT_010-dev_fwz_outputs/TakerPipeline/zzy_lasso0707_h20_res_input_csi2000_weight_v7_compat"
MARKET="/mnt/data1/fwz/HFT_010-dev_fwz_outputs/TakerPipeline/zzy_minute_market_for_curve_tmid_csi2000_weight_v7/*.parquet"

ROOT="/mnt/data1/fwz/HFT_010-dev_fwz_outputs/final_report/v17_lasso0707_finetune_stable_csi2000_v7"
mkdir -p "$ROOT"

run_one () {
  local a20="$1"
  local a10="$2"
  local lturn="$3"

  local name="h20raw_h10rg_a20_${a20}_a10_${a10}_l${lturn}"
  name="${name//./p}"
  local TAG="pure_cs_v17_${name}_csi2000"
  local OPTDIR="$ROOT/$TAG"
  local EVALDIR="$ROOT/${TAG}_canonical_eval"
  local LOG="$ROOT/${TAG}.log"

  echo
  echo "========== $TAG =========="
  echo "a20=$a20 a10=$a10 lturn=$lturn"

  python TakerOptimizer/scripts/run_v16_dual_alpha_optimizer.py \
    --h10-dir "$H10" \
    --h20-dir "$H20" \
    --output-dir "$OPTDIR" \
    --tag "$TAG" \
    --rebalance-step-minutes 10 \
    --turnover-limit 0.02 \
    --alpha20-scale "$a20" \
    --alpha10-delta-scale "$a10" \
    --lambda-turnover "$lturn" \
    --lambda-active 0.0001 \
    --lambda-ridge 0.00001 \
    --lambda-trade-ridge 0.00001 \
    --score-mode-h20 raw \
    --score-mode-h10 rank_gate \
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

for a20 in 0.0005 0.0008; do
  for a10 in 0.0004 0.0005 0.0006 0.0007; do
    for lturn in 0.0010 0.0012 0.0015 0.0018; do
      run_one "$a20" "$a10" "$lturn"
    done
  done
done
