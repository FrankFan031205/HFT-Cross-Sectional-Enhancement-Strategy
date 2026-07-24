#!/usr/bin/env bash
set -euo pipefail

cd /mnt/data1/fwz/HFT_010-dev_fwz

TS=$(date +%Y%m%d_%H%M%S)
LOGDIR=/mnt/data1/fwz/HFT_010-dev_fwz/TakerOptimizer/logs
mkdir -p "$LOGDIR"

OUTDIR=/mnt/data1/fwz/HFT_010-dev_fwz/TakerOptimizer/outputs/pure_cs_cvxpy_h20_r10_v1_t1aware
CFG=TakerOptimizer/config/pure_cs_cvxpy_h20_r10_v1_t1aware.yaml

echo "[START] $(date)"

echo "[STEP 1] optimizer"
python TakerOptimizer/scripts/run_pure_cs_cvxpy_index_enhance_t1aware.py \
  --config "$CFG"

echo "[STEP 2] backtest / curve generation"
# TODO:
# 把下面这行替换成你当前实际的“根据 target_positions 生成 curve csv”的命令
BT_CMD='echo "PLEASE_REPLACE_BT_CMD_WITH_YOUR_REAL_BACKTEST_COMMAND" && exit 1'
bash -lc "$BT_CMD"

echo "[STEP 3] plotting"
# TODO:
# 把下面这个 CURVE_FILE 改成你回测输出的 curve csv/parquet
CURVE_FILE="$OUTDIR/curve.csv"
PNG_OUT="$OUTDIR/time_sliced_sleeve_pure_cs_nav.png"

python TakerOptimizer/scripts/make_nav_plot_from_curve_csv.py \
  --input "$CURVE_FILE" \
  --output "$PNG_OUT" \
  --title "Time-sliced Sleeve Pure-CS, Raw Gradual Build"

echo "[DONE] $(date)"
