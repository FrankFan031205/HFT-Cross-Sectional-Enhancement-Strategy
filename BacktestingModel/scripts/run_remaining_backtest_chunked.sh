#!/usr/bin/env bash
set -euo pipefail

cd /home/fwz/projects/HFT_010-dev_fwz/BacktestingModel

TAG="mlp2_h60_202410_100"
QUOTE_PATH="../MarketMakingModel/outputs/quote_decisions/quote_decisions_mlp2_h60_202410_100_v5.csv"

mkdir -p logs outputs/fills outputs/trades outputs/metrics outputs/cache outputs/config_runs

echo "===== remaining backtest pipeline start: $(date) ====="

echo "[0] standardize config"

python - <<PY
import yaml

path = "config/backtest.yaml"

with open(path, "r") as f:
    cfg = yaml.safe_load(f)

tag = "${TAG}"

cfg["input"]["quote_decision_path"] = "${QUOTE_PATH}"

cfg["fill_model"]["mode"] = "touched_trade"
cfg["fill_model"]["queue_ahead_multiplier"] = 0.0

cfg["signal"] = {
    "col": "hidden_factor_mlp2_h60",
    "n_bins": 5
}

cfg["output"]["fill_path"] = f"outputs/fills/fills_touched_{tag}.csv"
cfg["output"]["enriched_fill_path"] = f"outputs/fills/fills_touched_enriched_{tag}.csv"
cfg["output"]["pnl_path"] = f"outputs/trades/trades_pnl_touched_{tag}.csv"
cfg["output"]["daily_pnl_path"] = f"outputs/metrics/daily_pnl_touched_{tag}.csv"
cfg["output"]["summary_path"] = f"outputs/metrics/summary_touched_{tag}.csv"
cfg["output"]["factor_pnl_path"] = f"outputs/metrics/factor_pnl_touched_{tag}.csv"
cfg["output"]["trade_cache_path"] = f"outputs/cache/raw_trades_{tag}.csv"
cfg["output"]["snapshot_cache_path"] = f"outputs/cache/snapshot_state_for_quotes_{tag}.csv"

cfg["fee"] = {
    "mode": "single_fill_side_specific",
    "commission_rate": 0.00005,
    "transfer_fee_rate": 0.00001,
    "handling_fee_rate": 0.0000341,
    "regulatory_fee_rate": 0.00002,
    "stamp_duty_rate": 0.0005
}

with open(path, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

print("config ready")
print("quote_decision_path:", cfg["input"]["quote_decision_path"])
print("touched fill_path:", cfg["output"]["fill_path"])
PY

echo "[1] create chunked enrich script"

cat > scripts/enrich_fills_with_factor_chunked.py <<'PY'
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.io import load_yaml, parse_datetime_series, save_csv


def sid(s):
    return s.astype(str).str.replace(".0", "", regex=False).str.zfill(6)


def make_key(dt, sec):
    x = pd.to_datetime(dt)
    return x.dt.floor("ms").dt.strftime("%Y-%m-%d %H:%M:%S.%f").str[:-3] + "|" + sid(sec)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/backtest.yaml")
    parser.add_argument("--fills", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--chunksize", type=int, default=1000000)
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    signal_col = cfg.get("signal", {}).get("col", "hidden_factor_mlp2_h60")
    fill_path = args.fills
    quote_path = cfg["input"]["quote_decision_path"]
    out_path = args.output

    print("[enrich] loading fills:", fill_path)
    fills = pd.read_csv(fill_path, low_memory=False)
    fills["decision_time"] = pd.to_datetime(fills["decision_time"])
    fills["securityid"] = sid(fills["securityid"])

    print("fills shape:", fills.shape)
    if len(fills) == 0:
        save_csv(fills, out_path)
        return

    print("fills date range:", fills["decision_time"].min(), "->", fills["decision_time"].max())
    print("fills num securities:", fills["securityid"].nunique())

    needed_keys = set(make_key(fills["decision_time"], fills["securityid"]))
    print("needed keys:", len(needed_keys))

    header = pd.read_csv(quote_path, nrows=0)
    cols = header.columns.tolist()

    if signal_col not in cols:
        candidates = [
            c for c in cols
            if "hidden" in c.lower() or "factor" in c.lower() or "pred" in c.lower() or "alpha" in c.lower()
        ]
        raise RuntimeError(f"{signal_col} not found. Candidate signal cols: {candidates}")

    usecols = ["datetime", "securityid", signal_col]

    optional = [
        "raw_pred", "pred_used", "fair_price", "quote_fair_price",
        "bid_edge", "ask_edge", "risk_state",
        "quote_bid", "quote_ask", "bid_price", "ask_price",
        "bid_size", "ask_size"
    ]

    for c in optional:
        if c in cols and c not in usecols:
            usecols.append(c)

    print("quote path:", quote_path)
    print("usecols:", usecols)
    print("chunksize:", args.chunksize)

    parts = []
    total = 0
    matched = 0

    for i, chunk in enumerate(pd.read_csv(quote_path, usecols=usecols, chunksize=args.chunksize, low_memory=False)):
        total += len(chunk)

        chunk["decision_time"] = parse_datetime_series(chunk["datetime"], "quote datetime")
        chunk["securityid"] = sid(chunk["securityid"])

        k = make_key(chunk["decision_time"], chunk["securityid"])
        m = k.isin(needed_keys)

        out = chunk.loc[m].copy()
        if len(out):
            parts.append(out)
            matched += len(out)

        if i % 10 == 0:
            print(f"[quote] chunk={i}, scanned={total}, matched={matched}")

    if not parts:
        raise RuntimeError("No quote rows matched fills. Check datetime/securityid alignment.")

    quotes = pd.concat(parts, ignore_index=True)
    quotes = quotes.drop_duplicates(["securityid", "decision_time"])
    quotes = quotes.drop(columns=["datetime"], errors="ignore")

    print("matched quote rows:", len(quotes))
    print("signal missing ratio in quotes:", quotes[signal_col].isna().mean())

    enriched = fills.merge(
        quotes,
        on=["securityid", "decision_time"],
        how="left",
    )

    print("enriched shape:", enriched.shape)
    print("signal missing ratio after merge:", enriched[signal_col].isna().mean())
    print("signal stats:")
    print(enriched[signal_col].describe())

    save_csv(enriched, out_path)


if __name__ == "__main__":
    main()
PY

FILL_PATH="outputs/fills/fills_touched_${TAG}.csv"
ENRICHED_PATH="outputs/fills/fills_touched_enriched_${TAG}.csv"
PNL_PATH="outputs/trades/trades_pnl_touched_${TAG}.csv"

echo "[2] check touched fills"

if [ ! -f "$FILL_PATH" ]; then
  echo "ERROR: touched fill file not found: $FILL_PATH"
  echo "Run fill simulation first."
  exit 1
fi

ls -lh "$FILL_PATH"

python - <<PY
import pandas as pd
path = "${FILL_PATH}"
df = pd.read_csv(path, low_memory=False)
print("fills shape:", df.shape)
if len(df):
    print("date range:", df["decision_time"].min(), "->", df["decision_time"].max())
    print("num securities:", df["securityid"].nunique())
    print("side counts:")
    print(df["side"].value_counts())
PY

echo "[3] chunked enrich touched fills"

python scripts/enrich_fills_with_factor_chunked.py \
  --config config/backtest.yaml \
  --fills "$FILL_PATH" \
  --output "$ENRICHED_PATH" \
  --chunksize 1000000

echo "[4] touched PnL"

python scripts/run_attention_pnl_backtest.py \
  --config config/backtest.yaml \
  --fills "$ENRICHED_PATH" \
  --output "$PNL_PATH"

echo "[5] prepare queue=0.05 config"

QCFG="outputs/config_runs/backtest_queue_mult_0p05_${TAG}.yaml"

python - <<PY
import yaml

base = "config/backtest.yaml"
out = "${QCFG}"
tag = "${TAG}"

with open(base, "r") as f:
    cfg = yaml.safe_load(f)

cfg["fill_model"]["mode"] = "queue_aware_trade"
cfg["fill_model"]["queue_ahead_multiplier"] = 0.05

cfg["output"]["fill_path"] = f"outputs/fills/fills_queue_mult_0p05_{tag}.csv"
cfg["output"]["enriched_fill_path"] = f"outputs/fills/fills_queue_mult_0p05_enriched_{tag}.csv"
cfg["output"]["pnl_path"] = f"outputs/trades/trades_pnl_queue_mult_0p05_{tag}.csv"
cfg["output"]["daily_pnl_path"] = f"outputs/metrics/daily_pnl_queue_mult_0p05_{tag}.csv"
cfg["output"]["summary_path"] = f"outputs/metrics/summary_queue_mult_0p05_{tag}.csv"
cfg["output"]["factor_pnl_path"] = f"outputs/metrics/factor_pnl_queue_mult_0p05_{tag}.csv"

# Reuse touched cache because quote universe/time range is the same.
cfg["output"]["trade_cache_path"] = f"outputs/cache/raw_trades_{tag}.csv"
cfg["output"]["snapshot_cache_path"] = f"outputs/cache/snapshot_state_for_quotes_{tag}.csv"

with open(out, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

print("wrote", out)
print("queue fill path:", cfg["output"]["fill_path"])
PY

echo "[6] run full queue=0.05 fill simulation"

python scripts/run_fill_simulation.py --config "$QCFG"

QFILL="outputs/fills/fills_queue_mult_0p05_${TAG}.csv"
QENRICHED="outputs/fills/fills_queue_mult_0p05_enriched_${TAG}.csv"
QPNL="outputs/trades/trades_pnl_queue_mult_0p05_${TAG}.csv"

echo "[7] check queue=0.05 fills"

python - <<PY
import pandas as pd
path = "${QFILL}"
df = pd.read_csv(path, low_memory=False)
print("queue fills shape:", df.shape)
if len(df):
    print("date range:", df["decision_time"].min(), "->", df["decision_time"].max())
    print("num securities:", df["securityid"].nunique())
    print("side counts:")
    print(df["side"].value_counts())
PY

if [ -s "$QFILL" ]; then
  echo "[8] chunked enrich queue=0.05 fills"

  python scripts/enrich_fills_with_factor_chunked.py \
    --config "$QCFG" \
    --fills "$QFILL" \
    --output "$QENRICHED" \
    --chunksize 1000000

  echo "[9] queue=0.05 PnL"

  python scripts/run_attention_pnl_backtest.py \
    --config "$QCFG" \
    --fills "$QENRICHED" \
    --output "$QPNL"
else
  echo "queue=0.05 fill file empty; skip PnL."
fi

echo "[10] print summaries"

python - <<PY
import os
import pandas as pd

files = [
    "outputs/metrics/summary_touched_${TAG}.csv",
    "outputs/metrics/summary_queue_mult_0p05_${TAG}.csv",
]

for f in files:
    print("\\n==========", f, "==========")
    if os.path.exists(f):
        df = pd.read_csv(f)
        print(df.T.to_string())
    else:
        print("missing")
PY

echo "===== remaining backtest pipeline finished: $(date) ====="
