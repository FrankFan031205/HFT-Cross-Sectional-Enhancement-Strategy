# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    inp = Path(args.input)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(inp, low_memory=False)

    # TakerModel from_positions scripts require these names.
    df["execution_date"] = df["date"].astype(int)

    # Our realtime adapter already made datetime like "2024-12-17 09:30:00".
    if "datetime" in df.columns:
        df["execution_datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    else:
        # fallback only
        df["execution_datetime"] = pd.to_datetime(df["date"].astype(str), errors="coerce")

    df["securityid"] = df["securityid"].astype(str).str.zfill(6)

    # Long-only: use effective target weight as final target.
    if "effective_target_weight" in df.columns:
        df["effective_target_weight"] = pd.to_numeric(df["effective_target_weight"], errors="coerce").fillna(0.0).clip(lower=0.0)
    elif "target_weight" in df.columns:
        df["effective_target_weight"] = pd.to_numeric(df["target_weight"], errors="coerce").fillna(0.0).clip(lower=0.0)
    else:
        raise ValueError("cannot find effective_target_weight or target_weight")

    if "target_weight" not in df.columns:
        df["target_weight"] = df["effective_target_weight"]
    else:
        df["target_weight"] = pd.to_numeric(df["target_weight"], errors="coerce").fillna(0.0).clip(lower=0.0)

    if "desired_target_weight" not in df.columns:
        df["desired_target_weight"] = df["target_weight"]

    # Make sure price columns exist and are numeric.
    for c in ["mid_price", "bid_price", "ask_price"]:
        if c not in df.columns:
            raise ValueError(f"missing required price column: {c}")
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # TakerModel v6/v7/v8 can read spread_bps_realized or spread_bps.
    if "spread_bps" not in df.columns:
        if "ask_price" in df.columns and "bid_price" in df.columns and "mid_price" in df.columns:
            df["spread_bps"] = (df["ask_price"] - df["bid_price"]) / df["mid_price"] * 10000.0
        else:
            df["spread_bps"] = np.nan

    df["spread_bps_realized"] = pd.to_numeric(df["spread_bps"], errors="coerce")

    # Keep useful audit columns if present.
    keep = [
        "execution_date", "execution_datetime", "date", "datetime", "minute",
        "securityid",
        "mid_price", "bid_price", "ask_price",
        "spread_bps", "spread_bps_realized",
        "effective_target_weight", "target_weight", "desired_target_weight",
        "state", "side", "blocked_reason", "selected",
        "target_qty", "effective_target_qty", "current_qty",
        "gross_weight", "buy_edge_bps", "sell_edge_bps",
        "global_alpha_raw", "global_alpha_z", "global_score",
        "local_alpha_raw", "local_alpha_z",
    ]
    keep = [c for c in keep if c in df.columns]

    df = df[keep].dropna(subset=["execution_datetime", "securityid", "mid_price"])
    df = df.sort_values(["execution_datetime", "securityid"])

    df.to_csv(out, index=False)

    print("[saved]", out)
    print("[shape]", df.shape)
    print("[time]", df["execution_datetime"].min(), "->", df["execution_datetime"].max())
    print("[dates]", df["execution_date"].nunique())
    print("[date-time]", df[["execution_date", "execution_datetime"]].drop_duplicates().shape[0])
    print("[target weight]")
    print(df["effective_target_weight"].describe())
    print("[negative target rows]", int((df["effective_target_weight"] < 0).sum()))
    print("[spread]")
    print(df["spread_bps_realized"].describe())


if __name__ == "__main__":
    main()
