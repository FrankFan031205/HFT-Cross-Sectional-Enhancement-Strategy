#!/usr/bin/env bash
set -euo pipefail

H10="/mnt/data1/fwz/HFT_010-dev_fwz_outputs/TakerPipeline/zzy_pure_cs_h10_res_minute_input_csi2000_weight_v7"
H20="/mnt/data1/fwz/HFT_010-dev_fwz_outputs/TakerPipeline/zzy_pure_cs_h20_res_minute_input_csi2000_weight_v7"
MARKET="/mnt/data1/fwz/HFT_010-dev_fwz_outputs/TakerPipeline/zzy_minute_market_for_curve_tmid_csi2000_weight_v7/*.parquet"

ROOT="/mnt/data1/fwz/HFT_010-dev_fwz_outputs/final_report/v16_dual_alpha_fix_top1_csi2000_v7"
mkdir -p "$ROOT"

run_one () {
  local name="$1"
  local step="$2"
  local tlim="$3"
  local a10="$4"
  local lturn="$5"
  local mode20="$6"
  local mode10="$7"
  local topk="$8"

  local tag="pure_cs_v16_${name}_csi2000"
  local optdir="$ROOT/${tag}"
  local evaldir="$ROOT/${tag}_canonical_eval"

  echo
  echo "========== $tag =========="
  echo "step=$step tlim=$tlim a10=$a10 lturn=$lturn mode20=$mode20 mode10=$mode10 topk=$topk"

  python TakerOptimizer/scripts/run_v16_dual_alpha_optimizer.py \
    --h10-dir "$H10" \
    --h20-dir "$H20" \
    --output-dir "$optdir" \
    --tag "$tag" \
    --rebalance-step-minutes "$step" \
    --turnover-limit "$tlim" \
    --alpha20-scale 0.0005 \
    --alpha10-delta-scale "$a10" \
    --lambda-turnover "$lturn" \
    --lambda-active 0.0001 \
    --lambda-ridge 0.00001 \
    --lambda-trade-ridge 0.00001 \
    --score-mode-h20 "$mode20" \
    --score-mode-h10 "$mode10" \
    --max-opt-names "$topk" \
    --solvers "CLARABEL,SCS,OSQP"

  mkdir -p "$evaldir"

  python TakerModel/scripts/plot_nav_benchmark_warmstart_noovernight.py \
    --market-glob "$MARKET" \
    --positions "$optdir/target_positions.csv" \
    --rebalance-summary "$optdir/summary_by_rebalance.csv" \
    --out-dir "$evaldir" \
    --tag "$tag" \
    --capital 200000000 \
    --init-gross 0.95 \
    --lot-size 100
}

# 1) top1 原逻辑，但限制求解股票数，目标：保住 alpha，降低 fallback
run_one "raw_step10_a10_02_t02_l18_top700"  10 0.02 0.0002 0.0018 raw raw 700
run_one "raw_step10_a10_02_t02_l18_top1000" 10 0.02 0.0002 0.0018 raw raw 1000
run_one "raw_step10_a10_02_t02_l18_top1300" 10 0.02 0.0002 0.0018 raw raw 1300

# 2) 放松单次换手限制，目标：减少不可行 / solver 压力
run_one "raw_step10_a10_02_t03_l18_top1000" 10 0.03 0.0002 0.0018 raw raw 1000
run_one "raw_step10_a10_02_t04_l18_top1000" 10 0.04 0.0002 0.0018 raw raw 1000

# 3) h20 保持 raw，h10 用 rank_gate 控制短周期噪声
run_one "hybrid_h20raw_h10rg_step10_a10_02_t02_l18_top1000" 10 0.02 0.0002 0.0018 raw rank_gate 1000
run_one "hybrid_h20raw_h10rg_step10_a10_03_t02_l18_top1000" 10 0.02 0.0003 0.0018 raw rank_gate 1000

# 4) 保留全量 rank_gate 稳定性，但稍微增强 h10 delta
run_one "rg_step20_a10_03_t04_l18_full" 20 0.04 0.0003 0.0018 rank_gate rank_gate 0
run_one "rg_step20_a10_05_t04_l18_full" 20 0.04 0.0005 0.0018 rank_gate rank_gate 0
