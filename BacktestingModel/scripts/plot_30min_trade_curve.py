#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_dt(s):
    s = s.astype(str).str.strip()
    dt = pd.to_datetime(s, errors="coerce")

    bad = dt.isna()
    if bad.any():
        x = s[bad]
        m = x.str.extract(r"(?P<date>\d{8})[_\sT]?(?P<time>\d{6,9})")
        ok = m["date"].notna() & m["time"].notna()

        if ok.any():
            date_part = m.loc[ok, "date"]
            time_part = m.loc[ok, "time"].astype(str)
            hhmmss = time_part.str.slice(0, 6)
            frac = time_part.str.slice(6)

            base = pd.to_datetime(
                date_part + hhmmss,
                format="%Y%m%d%H%M%S",
                errors="coerce",
            )

            ns = frac.apply(lambda z: int(str(z).ljust(9, "0")[:9]) if str(z) else 0)
            parsed = base + pd.to_timedelta(ns.values, unit="ns")

            dt.loc[x.index[ok]] = parsed.values

    return dt


def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def make_trading_bins(dates, freq="30min"):
    all_bins = []

    for d in sorted(dates):
        d = str(d)
        day = pd.to_datetime(d, format="%Y%m%d")

        morning = pd.date_range(
            day + pd.Timedelta(hours=9, minutes=30),
            day + pd.Timedelta(hours=11, minutes=30),
            freq=freq,
        )

        afternoon = pd.date_range(
            day + pd.Timedelta(hours=13, minutes=0),
            day + pd.Timedelta(hours=15, minutes=0),
            freq=freq,
        )

        bins = list(morning) + list(afternoon)
        all_bins.extend(bins)

    return pd.DataFrame({"bin_time": all_bins})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trades", required=True)
    parser.add_argument("--output-dir", default="outputs/metrics/half_hour_curves")
    parser.add_argument("--freq", default="30min")
    parser.add_argument("--pnl-col", default="net_pnl")
    parser.add_argument("--title", default="")
    parser.add_argument("--max-xticks", type=int, default=40)
    args = parser.parse_args()

    trades_path = Path(args.trades)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(trades_path, low_memory=False)

    time_col = find_col(df, ["fill_time", "decision_time", "datetime", "timestamp", "time"])
    if time_col is None:
        raise ValueError("Cannot find time column. Need fill_time / decision_time / datetime.")

    if args.pnl_col not in df.columns:
        raise ValueError(f"Cannot find pnl column: {args.pnl_col}")

    df["_dt"] = parse_dt(df[time_col])
    df["_pnl"] = pd.to_numeric(df[args.pnl_col], errors="coerce")

    df = df[df["_dt"].notna() & df["_pnl"].notna()].copy()
    df = df.sort_values("_dt")

    df["_date"] = df["_dt"].dt.strftime("%Y%m%d")
    df["_bin_time"] = df["_dt"].dt.floor(args.freq)

    # A-share lunch break adjustment:
    # trades between 11:30 and 13:00 are mapped to 11:30 or 13:00 depending on floor.
    # Normally there should be no trades during lunch.
    hourly = (
        df.groupby("_bin_time")
        .agg(
            num_trades=("_pnl", "size"),
            bin_pnl=("_pnl", "sum"),
        )
        .reset_index()
        .rename(columns={"_bin_time": "bin_time"})
    )

    dates = sorted(df["_date"].unique())
    bins = make_trading_bins(dates, freq=args.freq)

    out = bins.merge(hourly, on="bin_time", how="left")
    out["num_trades"] = out["num_trades"].fillna(0).astype(int)
    out["bin_pnl"] = out["bin_pnl"].fillna(0.0)
    out["cum_pnl"] = out["bin_pnl"].cumsum()

    out["trading_index"] = range(len(out))
    out["label"] = out["bin_time"].dt.strftime("%Y-%m-%d %H:%M")

    stem = trades_path.stem
    out_csv = output_dir / f"{stem}_30min_trade_pnl.csv"
    out_png = output_dir / f"{stem}_30min_trade_cum_pnl.png"
    out_bar_png = output_dir / f"{stem}_30min_trade_pnl_bar.png"

    out.to_csv(out_csv, index=False)

    tick_step = max(1, int(np.ceil(len(out) / max(1, args.max_xticks))))
    tick_idx = list(range(0, len(out), tick_step))
    if len(out) - 1 not in tick_idx:
        tick_idx.append(len(out) - 1)

    plt.figure(figsize=(18, 6))
    plt.plot(out["trading_index"], out["cum_pnl"], marker="o", linewidth=1.4)
    plt.title(args.title or f"30min Trade-Level Cumulative PnL: {stem}")
    plt.xlabel("Trading time")
    plt.ylabel("Cumulative net PnL")
    plt.xticks(
        out.loc[tick_idx, "trading_index"],
        out.loc[tick_idx, "label"],
        rotation=45,
        ha="right",
    )
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

    plt.figure(figsize=(18, 6))
    plt.bar(out["trading_index"], out["bin_pnl"])
    plt.title(args.title or f"30min Trade-Level PnL Bar: {stem}")
    plt.xlabel("Trading time")
    plt.ylabel("30min net PnL")
    plt.xticks(
        out.loc[tick_idx, "trading_index"],
        out.loc[tick_idx, "label"],
        rotation=45,
        ha="right",
    )
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_bar_png, dpi=160)
    plt.close()

    print("input:", trades_path)
    print("rows:", len(df))
    print("30min points:", len(out))
    print("total trade-level net_pnl:", out["bin_pnl"].sum())
    print("saved csv:", out_csv)
    print("saved curve png:", out_png)
    print("saved bar png:", out_bar_png)


if __name__ == "__main__":
    main()
