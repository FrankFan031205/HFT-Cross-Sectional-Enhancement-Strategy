#!/usr/bin/env bash
set -uo pipefail

H10="/mnt/data1/fwz/HFT_010-dev_fwz_outputs/TakerPipeline/zzy_lasso0707_h10_res_input_csi2000_weight_v7_compat"
H20="/mnt/data1/fwz/HFT_010-dev_fwz_outputs/TakerPipeline/zzy_lasso0707_h20_res_input_csi2000_weight_v7_compat"
MARKET="/mnt/data1/fwz/HFT_010-dev_fwz_outputs/TakerPipeline/zzy_minute_market_for_curve_tmid_csi2000_weight_v7/*.parquet"

ROOT="/mnt/data1/fwz/HFT_010-dev_fwz_outputs/final_report/v17_lasso0707_intensity_sweep_csi2000_v7"
mkdir -p "$ROOT"

run_one () {
  local name="$1"
  local mode20="$2"
  local mode10="$3"
  local a20="$4"
  local a10="$5"
  local tlim="$6"
  local lturn="$7"

  local TAG="pure_cs_v17_${name}_csi2000"
  local OPTDIR="$ROOT/$TAG"
  local EVALDIR="$ROOT/${TAG}_canonical_eval"
  local LOG="$ROOT/${TAG}.log"

  echo
  echo "========== $TAG =========="
  echo "mode20=$mode20 mode10=$mode10 a20=$a20 a10=$a10 tlim=$tlim lturn=$lturn"

  python TakerOptimizer/scripts/run_v16_dual_alpha_optimizer.py \
    --h10-dir "$H10" \
    --h20-dir "$H20" \
    --output-dir "$OPTDIR" \
    --tag "$TAG" \
    --rebalance-step-minutes 10 \
    --turnover-limit "$tlim" \
    --alpha20-scale "$a20" \
    --alpha10-delta-scale "$a10" \
    --lambda-turnover "$lturn" \
    --lambda-active 0.0001 \
    --lambda-ridge 0.00001 \
    --lambda-trade-ridge 0.00001 \
    --score-mode-h20 "$mode20" \
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

  if [ $? -ne 0 ]; then
    echo "[FAILED plot] $TAG, see $LOG"
    return 0
  fi

  echo "[DONE] $TAG"
}

# baseline repeat
run_one "baseline_h20raw_h10rg_a20_05_a10_03_t02_l18" raw rank_gate 0.0005 0.0003 0.02 0.0018

# 1) 提高 h10 delta alpha，但保持 rank_gate
run_one "h20raw_h10rg_a20_05_a10_05_t02_l18" raw rank_gate 0.0005 0.0005 0.02 0.0018
run_one "h20raw_h10rg_a20_05_a10_08_t02_l18" raw rank_gate 0.0005 0.0008 0.02 0.0018
run_one "h20raw_h10rg_a20_05_a10_10_t02_l18" raw rank_gate 0.0005 0.0010 0.02 0.0018

# 2) 放松 h10 gate：rank_gate -> rank / csz / raw
run_one "h20raw_h10rank_a20_05_a10_05_t02_l18" raw rank 0.0005 0.0005 0.02 0.0018
run_one "h20raw_h10csz_a20_05_a10_05_t02_l18"  raw csz  0.0005 0.0005 0.02 0.0018
run_one "h20raw_h10raw_a20_05_a10_05_t02_l18"  raw raw  0.0005 0.0005 0.02 0.0018

# 3) 新版 raw scale 可能偏弱：h20 也改成 csz / rank
run_one "h20csz_h10rg_a20_05_a10_05_t02_l18"  csz  rank_gate 0.0005 0.0005 0.02 0.0018
run_one "h20rank_h10rg_a20_05_a10_05_t02_l18" rank rank_gate 0.0005 0.0005 0.02 0.0018

# 4) 同时增加 alpha 和放松成本约束
run_one "h20raw_h10rank_a20_10_a10_08_t03_l10" raw rank 0.0010 0.0008 0.03 0.0010
run_one "h20csz_h10rank_a20_10_a10_08_t03_l10" csz rank 0.0010 0.0008 0.03 0.0010
run_one "h20raw_h10csz_a20_10_a10_08_t03_l10"  raw csz  0.0010 0.0008 0.03 0.0010
