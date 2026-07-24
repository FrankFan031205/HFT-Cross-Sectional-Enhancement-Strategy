import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.io import load_yaml, parse_datetime_series, save_csv


def sid(s):
    return s.astype(str).str.replace(".0", "", regex=False).str.zfill(6)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/backtest.yaml")
    parser.add_argument("--fills", default="")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    signal_col = cfg.get("signal", {}).get("col", "hidden_factor_mlp2_h60")
    fill_path = args.fills or cfg["output"]["fill_path"]
    quote_path = cfg["input"]["quote_decision_path"]
    out_path = args.output or cfg["output"].get("enriched_fill_path", "outputs/fills/fills_touched_enriched.csv")

    print("[1] loading fills:", fill_path)
    fills = pd.read_csv(fill_path, low_memory=False)
    fills["decision_time"] = pd.to_datetime(fills["decision_time"])
    fills["securityid"] = sid(fills["securityid"])

    print("fills shape:", fills.shape)

    print("[2] loading quote decisions:", quote_path)
    quotes = pd.read_csv(quote_path, low_memory=False)

    if signal_col not in quotes.columns:
        candidates = [
            c for c in quotes.columns
            if "hidden" in c.lower() or "factor" in c.lower() or "pred" in c.lower() or "alpha" in c.lower()
        ]
        raise RuntimeError(
            f"{signal_col} not found in quote decision file. Candidate signal columns: {candidates}"
        )

    quotes["decision_time"] = parse_datetime_series(quotes["datetime"], "quote datetime")
    quotes["securityid"] = sid(quotes["securityid"])

    keep = ["securityid", "decision_time", signal_col]

    for c in [
        "raw_pred",
        "pred_used",
        "fair_price",
        "quote_fair_price",
        "bid_edge",
        "ask_edge",
        "risk_state",
        "quote_bid",
        "quote_ask",
        "bid_price",
        "ask_price",
        "bid_size",
        "ask_size",
    ]:
        if c in quotes.columns and c not in keep:
            keep.append(c)

    q = quotes[keep].copy()
    q = q.drop_duplicates(["securityid", "decision_time"])
    q = q.sort_values(["securityid", "decision_time"])

    outs = []

    for sec, fg in fills.groupby("securityid", sort=False):
        qg = q[q["securityid"] == sec]
        if len(qg) == 0:
            continue

        fg = fg.sort_values("decision_time").copy()
        qg = qg.sort_values("decision_time").copy()

        mg = pd.merge_asof(
            fg,
            qg,
            on="decision_time",
            by="securityid",
            direction="nearest",
            tolerance=pd.Timedelta(milliseconds=1000),
        )
        outs.append(mg)

    if not outs:
        raise RuntimeError("No fills matched quote decisions.")

    out = pd.concat(outs, ignore_index=True)

    print("matched fills shape:", out.shape)
    print("signal col:", signal_col)
    print("signal missing ratio:", out[signal_col].isna().mean())
    print("signal stats:")
    print(out[signal_col].describe())

    save_csv(out, out_path)


if __name__ == "__main__":
    main()
