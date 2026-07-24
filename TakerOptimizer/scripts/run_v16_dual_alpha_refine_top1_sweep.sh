#!/usr/bin/env bash
set -uo pipefail

H10="/mnt/data1/fwz/HFT_010-dev_fwz_outputs/TakerPipeline/zzy_pure_cs_h10_res_minute_input_csi2000_weight_v7"
H20="/mnt/data1/fwz/HFT_010-dev_fwz_outputs/TakerPipeline/zzy_pure_cs_h20_res_minute_input_csi2000_weight_v7"
MARKET="/mnt/data1/fwz/HFT_010-dev_fwz_outputs/TakerPipeline/zzy_minute_market_for_curve_tmid_csi2000_weight_v7/*.parquet"

ROOT="/mnt/data1/fwz/HFT_010-dev_fwz_outputs/final_report/v16_dual_alpha_refine_top1_csi2000_v7"
mkdir -p "$ROOT"

run_one () {
  local name="$1"
  local step="$2"
  local tlim="$3"
  local a10="$4"
  local lturn="$5"
  local ltrade="$6"
  local mode20="$7"
  local mode10="$8"
  local solvers="$9"

  local tag="pure_cs_v16_${name}_csi2000"
  local optdir="$ROOT/${tag}"
  local evaldir="$ROOT/${tag}_canonical_eval"
  local logfile="$ROOT/${tag}.log"

  echo
  echo "========== $tag =========="
  echo "step=$step tlim=$tlim a10=$a10 lturn=$lturn ltrade=$ltrade mode20=$mode20 mode10=$mode10 solvers=$solvers"

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
    --lambda-trade-ridge "$ltrade" \
    --score-mode-h20 "$mode20" \
    --score-mode-h10 "$mode10" \
    --solvers "$solvers" \
    > "$logfile" 2>&1

  if [ $? -ne 0 ]; then
    echo "[FAILED optimizer] $tag, see $logfile"
    return 0
  fi

  mkdir -p "$evaldir"

  python TakerModel/scripts/plot_nav_benchmark_warmstart_noovernight.py \
    --market-glob "$MARKET" \
    --positions "$optdir/target_positions.csv" \
    --rebalance-summary "$optdir/summary_by_rebalance.csv" \
    --out-dir "$evaldir" \
    --tag "$tag" \
    --capital 200000000 \
    --init-gross 0.95 \
    --lot-size 100 \
    >> "$logfile" 2>&1

  if [ $? -ne 0 ]; then
    echo "[FAILED plot] $tag, see $logfile"
    return 0
  fi

  echo "[DONE] $tag"
}

# A. 只换 solver 顺序，验证 fallback 是不是 CLARABEL 首发问题
run_one "raw_step10_a10_02_t02_l18_osqpfirst" 10 0.02 0.0002 0.0018 0.00001 raw raw "OSQP,SCS,CLARABEL"
run_one "raw_step10_a10_02_t02_l18_scsfirst"  10 0.02 0.0002 0.0018 0.00001 raw raw "SCS,OSQP,CLARABEL"

# B. 放松单次换手，减少 tight turnover 造成的数值压力
run_one "raw_step10_a10_02_t03_l18_osqpfirst" 10 0.03 0.0002 0.0018 0.00001 raw raw "OSQP,SCS,CLARABEL"
run_one "raw_step10_a10_02_t04_l18_osqpfirst" 10 0.04 0.0002 0.0018 0.00001 raw raw "OSQP,SCS,CLARABEL"

# C. 增强 trade ridge，让 delta 更平滑
run_one "raw_step10_a10_02_t02_l18_tr5e5_osqpfirst" 10 0.02 0.0002 0.0018 0.00005 raw raw "OSQP,SCS,CLARABEL"
run_one "raw_step10_a10_02_t02_l18_tr1e4_osqpfirst" 10 0.02 0.0002 0.0018 0.00010 raw raw "OSQP,SCS,CLARABEL"

# D. hybrid: h20 继续 raw，h10 用 rank_gate 降噪
run_one "hybrid_h20raw_h10rg_step10_a10_02_t02_l18_osqpfirst" 10 0.02 0.0002 0.0018 0.00001 raw rank_gate "OSQP,SCS,CLARABEL"
run_one "hybrid_h20raw_h10rg_step10_a10_03_t02_l18_osqpfirst" 10 0.02 0.0003 0.0018 0.00001 raw rank_gate "OSQP,SCS,CLARABEL"

# E. 稳健版增强：以 fallback=0 的 rank_gate 为基础，稍微增加 h10 delta
run_one "rg_step20_a10_03_t04_l18" 20 0.04 0.0003 0.0018 0.00001 rank_gate rank_gate "CLARABEL,SCS,OSQP"
run_one "rg_step20_a10_05_t04_l18" 20 0.04 0.0005 0.0018 0.00001 rank_gate rank_gate "CLARABEL,SCS,OSQP"
