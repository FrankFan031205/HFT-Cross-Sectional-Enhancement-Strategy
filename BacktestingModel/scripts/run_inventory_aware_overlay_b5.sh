#!/usr/bin/env bash
set -e

cd /home/fwz/projects/HFT_010-dev_fwz/BacktestingModel

TAG="mlp2_h60_202410_100"
BASE="outputs/trades/trades_pnl_queue_mult_0p05_${TAG}.csv"

mkdir -p logs outputs/trades outputs/portfolio outputs/metrics/analysis

echo "===== B5.4 inventory-aware overlay start: $(date) ====="

echo "[1] create policy trade files"

python - <<'PY'
import pandas as pd

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
}

for name, mask in policies.items():
    out = f"outputs/trades/trades_pnl_queue_mult_0p05_{name}_{TAG}.csv"
    sub = df.loc[mask].copy()
    sub.to_csv(out, index=False)
    print(name, sub.shape, out)
PY

echo "[2] run inventory-aware overlays"

# band format:
# name:sell_floor:buy_block:max_cap
BANDS=(
  "hardonly:0:15000:15000"
  "loose:1000:12000:15000"
  "medium:2000:10000:15000"
  "tight:3000:9000:15000"
  "medium_cap20000:2000:12000:20000"
)

for policy in all abs_top50 abs_top40 abs_top30
do
  for band_spec in "${BANDS[@]}"
  do
    IFS=':' read -r band sell_floor buy_block cap <<< "$band_spec"

    inv=5000

    echo ""
    echo "===== policy=${policy}, band=${band}, inv=${inv}, sell_floor=${sell_floor}, buy_block=${buy_block}, cap=${cap} ====="

    BASE_POLICY="outputs/trades/trades_pnl_queue_mult_0p05_${policy}_${TAG}.csv"

    OUT="outputs/trades/trades_pnl_queue_mult_0p05_${policy}_inventory${inv}_cap${cap}_${band}_tplus1_${TAG}.csv"
    SKIP="outputs/trades/skipped_queue_mult_0p05_${policy}_inventory${inv}_cap${cap}_${band}_tplus1_${TAG}.csv"
    PREFIX="outputs/portfolio/portfolio_replay_queue_0p05_${policy}_inventory${inv}_cap${cap}_${band}_tplus1_${TAG}"

    python scripts/filter_inventory_aware_overlay.py \
      --trades "$BASE_POLICY" \
      --output "$OUT" \
      --skipped-output "$SKIP" \
      --initial-position-per-symbol "$inv" \
      --max-position-per-symbol "$cap" \
      --sell-floor-position "$sell_floor" \
      --buy-block-position "$buy_block" \
      --tplus1 1

    if [ -s "$OUT" ]; then
      python scripts/run_portfolio_replay.py \
        --trades "$OUT" \
        --model "queue_0p05_${policy}_inventory${inv}_cap${cap}_${band}_tplus1" \
        --out-prefix "$PREFIX" \
        --capital 0 \
        --initial-position-per-symbol "$inv" \
        --allow-short 0 \
        --record-every 10000
    else
      echo "accepted trade file empty, skip replay: $OUT"
    fi
  done
done

echo "===== B5.4 inventory-aware overlay finished: $(date) ====="
