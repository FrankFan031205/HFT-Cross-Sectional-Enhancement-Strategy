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
