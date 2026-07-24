#!/usr/bin/env bash
set -euo pipefail

H10="/mnt/data1/fwz/HFT_010-dev_fwz_outputs/TakerPipeline/zzy_lasso0707_h10_res_input_csi2000_weight_v7_compat"
H20="/mnt/data1/fwz/HFT_010-dev_fwz_outputs/TakerPipeline/zzy_lasso0707_h20_res_input_csi2000_weight_v7_compat"
MARKET="/mnt/data1/fwz/HFT_010-dev_fwz_outputs/TakerPipeline/zzy_minute_market_for_curve_tmid_csi2000_weight_v7/*.parquet"

ROOT="/mnt/data1/fwz/HFT_010-dev_fwz_outputs/final_report/v17_lasso0707_dual_alpha_noovernight_csi2000_v7"
TAG="pure_cs_v17_lasso0707_hybrid_h20raw_h10rg_step10_a10_03_t02_l18_csi2000"

OPTDIR="$ROOT/$TAG"
EVALDIR="$ROOT/${TAG}_canonical_eval"

mkdir -p "$ROOT" "$EVALDIR"

echo "===== optimizer ====="
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
  --solvers "OSQP,SCS,CLARABEL"

echo "===== canonical no-overnight plot ====="
python TakerModel/scripts/plot_nav_benchmark_warmstart_noovernight.py \
  --market-glob "$MARKET" \
  --positions "$OPTDIR/target_positions.csv" \
  --rebalance-summary "$OPTDIR/summary_by_rebalance.csv" \
  --out-dir "$EVALDIR" \
  --tag "$TAG" \
  --capital 200000000 \
  --init-gross 0.95 \
  --lot-size 100

echo "===== mentor plot ====="
CURVE="$EVALDIR/${TAG}_nav_curve_benchmark_warmstart_noovernight.csv"
MENTOR="$EVALDIR/mentor_plot"
mkdir -p "$MENTOR"

python TakerModel/scripts/fix_purecs_curve_for_mentor.py \
  --curve-csv "$CURVE" \
  --out-dir "$MENTOR" \
  --drop-open-minutes 0 \
  --drop-close-minutes 0 \
  --top-n 50

echo "===== outputs ====="
find "$EVALDIR" -maxdepth 2 -type f | sort
