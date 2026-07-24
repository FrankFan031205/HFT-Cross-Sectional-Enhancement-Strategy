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


def parse_datetime(df):
    dt_col = find_col(df, ["datetime", "time", "timestamp", "decision_time", "fill_time", "markout_time"])

    if dt_col is not None:
        s = df[dt_col].astype(str)

        # Handles formats like 20241022_093000000
        if s.str.contains("_").any():
            dt = pd.to_datetime(s.str.replace("_", " ", regex=False), errors="coerce")
            if dt.notna().sum() > 0:
                return dt

        dt = pd.to_datetime(s, errors="coerce")
        if dt.notna().sum() > 0:
            return dt

    date_col = find_col(df, ["date", "trading_date"])
    time_col = find_col(df, ["clock_time", "bar_time", "hhmmss"])

    if date_col is not None and time_col is not None:
        s = df[date_col].astype(str) + " " + df[time_col].astype(str)
        dt = pd.to_datetime(s, errors="coerce")
        if dt.notna().sum() > 0:
            return dt

    if date_col is not None:
        dt = pd.to_datetime(df[date_col].astype(str), errors="coerce")
        if dt.notna().sum() > 0:
            return dt

    raise ValueError("Cannot infer datetime column")


def pick_value_col(df):
    candidates = [
        "cum_pnl",
        "cumulative_pnl",
        "equity",
        "final_equity",
        "daily_cum_pnl",
        "portfolio_value",
        "pnl",
        "daily_pnl",
    ]
    c = find_col(df, candidates)
    if c is not None:
        return c

    numeric_cols = []
    for col in df.columns:
        if col.startswith("_"):
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        if s.notna().sum() > 0:
            numeric_cols.append(col)

    if not numeric_cols:
        raise ValueError("Cannot infer value column")

    return numeric_cols[-1]


def make_hourly_curve(input_path, output_dir, value_col=None, title=None):
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path, low_memory=False)
    if df.empty:
        raise ValueError("Input file is empty: " + str(input_path))

    df["_dt"] = parse_datetime(df)
    df = df[df["_dt"].notna()].copy()
    df = df.sort_values("_dt")

    if value_col is None:
        value_col = pick_value_col(df)

    df["_value"] = pd.to_numeric(df[value_col], errors="coerce")
    df = df[df["_value"].notna()].copy()

    # One-hour curve: use last available value in each hour.
    hourly = (
        df.set_index("_dt")["_value"]
        .resample("1H")
        .last()
        .dropna()
        .reset_index()
        .rename(columns={"_dt": "datetime", "_value": value_col})
    )

    # Also compute hourly change.
    hourly["hourly_change"] = hourly[value_col].diff()
    if len(hourly) > 0:
        hourly.loc[hourly.index[0], "hourly_change"] = hourly.loc[hourly.index[0], value_col]

    stem = input_path.stem
    hourly_csv = output_dir / f"{stem}_hourly.csv"
    png_path = output_dir / f"{stem}_hourly_curve.png"

    hourly.to_csv(hourly_csv, index=False)

    plt.figure(figsize=(14, 6))
    plt.plot(hourly["datetime"], hourly[value_col], marker="o", linewidth=1.5)
    plt.title(title or f"Hourly Performance Curve: {stem}")
    plt.xlabel("Datetime, hourly")
    plt.ylabel(value_col)
    plt.xticks(rotation=45, ha="right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(png_path, dpi=160)
    plt.close()

    print("input:", input_path)
    print("value_col:", value_col)
    print("hourly rows:", len(hourly))
    print("saved csv:", hourly_csv)
    print("saved png:", png_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input curve csv")
    parser.add_argument("--output-dir", default="outputs/metrics/performance_pack/hourly_curves")
    parser.add_argument("--value-col", default=None)
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    make_hourly_curve(
        input_path=args.input,
        output_dir=args.output_dir,
        value_col=args.value_col,
        title=args.title,
    )


if __name__ == "__main__":
    main()
