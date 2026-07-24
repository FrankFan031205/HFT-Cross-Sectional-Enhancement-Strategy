#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def parse_time(df):
    time_col = find_col(df, [
        "fill_time",
        "decision_time",
        "markout_time",
        "datetime",
        "timestamp",
        "time",
    ])

    if time_col is None:
        raise ValueError("cannot find time column")

    s = df[time_col].astype(str)

    # handle 20241022_093000000
    dt = pd.to_datetime(s.str.replace("_", " ", regex=False), errors="coerce")
    if dt.notna().sum() > 0:
        return dt

    dt = pd.to_datetime(s, errors="coerce")
    if dt.notna().sum() > 0:
        return dt

    raise ValueError("cannot parse time column: " + time_col)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="trades_pnl csv")
    parser.add_argument("--output-dir", default="outputs/metrics/hourly_trade_curves")
    parser.add_argument("--pnl-col", default="net_pnl")
    parser.add_argument("--title", default="")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path, low_memory=False)

    if args.pnl_col not in df.columns:
        raise ValueError(f"pnl column not found: {args.pnl_col}")

    df["_dt"] = parse_time(df)
    df["_pnl"] = pd.to_numeric(df[args.pnl_col], errors="coerce")
    df = df[df["_dt"].notna() & df["_pnl"].notna()].copy()

    df = df.sort_values("_dt")
    df["_hour"] = df["_dt"].dt.floor("1H")

    hourly = (
        df.groupby("_hour")
        .agg(
            num_trades=("_pnl", "size"),
            hourly_pnl=("_pnl", "sum"),
        )
        .reset_index()
        .rename(columns={"_hour": "datetime"})
    )

    hourly["cum_pnl"] = hourly["hourly_pnl"].cumsum()

    stem = input_path.stem
    out_csv = output_dir / f"{stem}_hourly_trade_pnl.csv"
    out_png = output_dir / f"{stem}_hourly_trade_pnl_curve.png"

    hourly.to_csv(out_csv, index=False)

    plt.figure(figsize=(16, 6))
    plt.plot(hourly["datetime"], hourly["cum_pnl"], marker="o", linewidth=1.5)
    plt.title(args.title or f"Hourly Trade-Level Cumulative PnL: {stem}")
    plt.xlabel("Datetime, hourly")
    plt.ylabel("Cumulative net PnL")
    plt.xticks(rotation=45, ha="right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

    print("input:", input_path)
    print("rows:", len(df))
    print("hourly rows:", len(hourly))
    print("total net pnl:", hourly["hourly_pnl"].sum())
    print("saved csv:", out_csv)
    print("saved png:", out_png)


if __name__ == "__main__":
    main()
