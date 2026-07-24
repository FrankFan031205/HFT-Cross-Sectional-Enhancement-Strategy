#!/usr/bin/env bash
set -e

cd /home/fwz/projects/HFT_010-dev_fwz/BacktestingModel

TAG="mlp2_h60_202410_100"
BASE="outputs/trades/trades_pnl_queue_mult_0p05_${TAG}.csv"

mkdir -p logs outputs/trades outputs/portfolio outputs/metrics/analysis

echo "===== B5.2 policy + inventory scenarios start: $(date) ====="

echo "[1] create policy filtered trade files"

python - <<'PY'
import pandas as pd
import os

TAG = "mlp2_h60_202410_100"
BASE = f"outputs/trades/trades_pnl_queue_mult_0p05_{TAG}.csv"
SIGNAL = "hidden_factor_mlp2_h60"

df = pd.read_csv(BASE, low_memory=False)
s = df[SIGNAL]
abs_s = s.abs()

policies = {
    "all": pd.Series(True, index=df.index),
    "abs_top50": abs_s >= abs_s.quantile(0.50),
    "abs_top40": abs_s >= abs_s.quantile(0.60),
    "abs_top30": abs_s >= abs_s.quantile(0.70),
    "abs_top20": abs_s >= abs_s.quantile(0.80),
    "abs_top40_buy_only": (abs_s >= abs_s.quantile(0.60)) & (df["side"] == "BUY"),
    "abs_top30_buy_only": (abs_s >= abs_s.quantile(0.70)) & (df["side"] == "BUY"),
}

for name, mask in policies.items():
    out = f"outputs/trades/trades_pnl_queue_mult_0p05_{name}_{TAG}.csv"
    sub = df.loc[mask].copy()
    sub.to_csv(out, index=False)
    print(name, sub.shape, out)

print("done")
PY

echo "[2] run inventory constrained replay for each policy"

for policy in all abs_top50 abs_top40 abs_top30 abs_top20 abs_top40_buy_only abs_top30_buy_only
do
  for inv in 0 1000 5000 10000
  do
    if [ "$inv" = "0" ]; then
      cap=5000
    else
      cap=$((inv * 5))
    fi

    echo ""
    echo "===== policy=${policy}, inv=${inv}, cap=${cap}, T+1 ====="

    BASE_POLICY="outputs/trades/trades_pnl_queue_mult_0p05_${policy}_${TAG}.csv"
    OUT="outputs/trades/trades_pnl_queue_mult_0p05_${policy}_inventory${inv}_tplus1_${TAG}.csv"
    SKIP="outputs/trades/skipped_queue_mult_0p05_${policy}_inventory${inv}_tplus1_${TAG}.csv"
    PREFIX="outputs/portfolio/portfolio_replay_queue_0p05_${policy}_inventory${inv}_tplus1_${TAG}"

    python scripts/filter_inventory_constraints.py \
      --trades "$BASE_POLICY" \
      --output "$OUT" \
      --skipped-output "$SKIP" \
      --initial-position-per-symbol "$inv" \
      --max-position-per-symbol "$cap" \
      --tplus1 1

    python scripts/run_portfolio_replay.py \
      --trades "$OUT" \
      --model "queue_0p05_${policy}_inventory${inv}_tplus1" \
      --out-prefix "$PREFIX" \
      --capital 0 \
      --initial-position-per-symbol "$inv" \
      --allow-short 0 \
      --record-every 10000
  done
done

echo "===== B5.2 policy + inventory scenarios finished: $(date) ====="
