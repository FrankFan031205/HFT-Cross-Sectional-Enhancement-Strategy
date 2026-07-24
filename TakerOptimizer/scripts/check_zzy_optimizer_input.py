#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Quick sanity check for prepared ZZY optimizer input parquet files."""
from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--sample", type=int, default=5)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    files = sorted(Path(args.input_dir).glob("optimizer_input_*.parquet"))
    if not files:
        raise SystemExit(f"No optimizer_input_*.parquet under {args.input_dir}")

    lf = pl.scan_parquet([str(p) for p in files])
    print("[files]")
    for p in files[:10]:
        print(" ", p)
    if len(files) > 10:
        print(f"  ... {len(files) - 10} more")

    print("\n[overall]")
    print(lf.select(
        pl.len().alias("rows"),
        pl.col("date").n_unique().alias("dates"),
        pl.col("sid").n_unique().alias("sids"),
        pl.col("ts").n_unique().alias("ts_count"),
        pl.col("exec_spread_bps").mean().alias("avg_spread_bps"),
        pl.col("volume_60s").mean().alias("avg_volume_60s"),
    ).collect())

    print("\n[null check]")
    key_cols = ["pred_ret", "signal_z", "exec_buy_price", "exec_sell_price", "exec_mid_price", "volume_60s"]
    print(lf.select([pl.col(c).null_count().alias(c) for c in key_cols]).collect())

    print("\n[daily]")
    print(lf.group_by("date").agg(
        pl.len().alias("rows"),
        pl.col("sid").n_unique().alias("sids"),
        pl.col("ts").n_unique().alias("ts_count"),
        pl.col("signal_z").std().alias("signal_z_std"),
        pl.col("exec_spread_bps").mean().alias("avg_spread_bps"),
    ).sort("date").collect())

    print("\n[head]")
    print(lf.head(args.sample).collect())


if __name__ == "__main__":
    main()
