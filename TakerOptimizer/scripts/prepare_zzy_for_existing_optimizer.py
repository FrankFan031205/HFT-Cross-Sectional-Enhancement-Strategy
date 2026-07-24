# -*- coding: utf-8 -*-
"""
Convert zzy ZZ2000 optimizer_input_*.parquet to the old CSV format
used by existing TakerOptimizer v2_7 / v2_8 / v2_9.

This is only an input adapter:
  - does not change optimizer logic
  - does not use fwd_ret_label for signal
  - uses signal_z as ranking alpha
  - creates a proxy pred_ret in return units for cost-aware alpha logic

Default: rebalance every 60 seconds because old optimizer's cooldown is minute-style.
"""
import argparse
from pathlib import Path

import polars as pl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--model-name", default="zzy_zz2000_h5_res")
    ap.add_argument("--price-multiplier", type=float, default=0.01)
    ap.add_argument("--rebalance-ts-step", type=int, default=60)
    ap.add_argument("--max-spread-bps", type=float, default=50.0)
    ap.add_argument("--alpha-scale-bps", type=float, default=10.0,
                    help="1 z-score maps to this many bps proxy return.")
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    out_root = Path(args.out_root)
    alpha_dir = out_root / "alpha"
    pricing_dir = out_root / "pricing"
    cfg_dir = out_root / "config"
    alpha_dir.mkdir(parents=True, exist_ok=True)
    pricing_dir.mkdir(parents=True, exist_ok=True)
    cfg_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob("optimizer_input_*.parquet"))
    if not files:
        raise SystemExit(f"no optimizer_input_*.parquet under {in_dir}")

    lf = pl.scan_parquet([str(x) for x in files])

    model = args.model_name
    alpha_col = f"alpha_{model}"
    pred_col = f"pred_ret_{model}"
    fair_col = f"fair_price_{model}"

    df = (
        lf
        .filter(
            (pl.col("ts") % int(args.rebalance_ts_step) == 0)
            & (pl.col("exec_spread_bps") <= float(args.max_spread_bps))
            & (pl.col("exec_mid_price") > 0)
            & (pl.col("exec_buy_price") > 0)
            & (pl.col("exec_sell_price") > 0)
            & (pl.col("volume_60s") > 0)
            & pl.col("signal_z").is_not_null()
        )
        .with_columns([
            (pl.col("ts") // 60).cast(pl.Int64).alias("minute"),
            pl.col("SecurityID").cast(pl.Utf8).alias("securityid"),

            (pl.col("exec_mid_price") * float(args.price_multiplier)).alias("mid_price"),
            (pl.col("exec_buy_price") * float(args.price_multiplier)).alias("ask_price"),
            (pl.col("exec_sell_price") * float(args.price_multiplier)).alias("bid_price"),

            pl.col("signal_z").alias(alpha_col),
            pl.col("signal_z").alias("signal_z"),

            (pl.col("signal_z") * float(args.alpha_scale_bps) / 10000.0).alias(pred_col),
        ])
        .with_columns([
            pl.col("mid_price").alias("price"),
            pl.col("mid_price").alias("last_price"),
            (pl.col("mid_price") * (1.0 + pl.col(pred_col))).alias(fair_col),
            (pl.col("date") * 100000 + pl.col("minute")).alias("datetime"),
            pl.col("exec_spread_bps").alias("spread_bps"),
            pl.col("max_participation_notional").alias("capacity_notional_raw"),
            (pl.col("max_participation_notional") * float(args.price_multiplier)).alias("capacity_notional"),
        ])
        .collect(streaming=True)
        .sort(["date", "minute", "securityid"])
    )

    alpha_path = alpha_dir / f"alpha_{model}.csv"
    pricing_path = pricing_dir / f"pricing_{model}.csv"
    cfg_path = cfg_dir / f"taker_position_optimizer_v2_7_{model}.yaml"

    alpha_cols = [
        "date", "datetime", "minute", "securityid",
        alpha_col, "signal_z",
        "mid_price", "price", "last_price",
        "spread_bps", "volume_60s", "capacity_notional",
    ]
    pricing_cols = [
        "date", "datetime", "minute", "securityid",
        pred_col, fair_col,
        "mid_price", "price", "last_price",
        "ask_price", "bid_price", "spread_bps",
        "volume_60s", "capacity_notional",
    ]

    df.select(alpha_cols).write_csv(alpha_path)
    df.select(pricing_cols).write_csv(pricing_path)

    cfg = f"""project:
  version: v2_7_{model}
data:
  alpha_path: {alpha_path}
  pricing_path: {pricing_path}
columns:
  date_col: date
  datetime_col: datetime
  minute_col: minute
  securityid_col: securityid
  alpha_col: {alpha_col}
  pricing_pred_col: {pred_col}
  fair_price_col: {fair_col}
  mid_price_col: mid_price
optimizer:
  capital: 200000000
  target_gross_limit: 0.60
  max_gross_limit: 0.60
  max_name_weight: 0.005
  single_name_limit: 0.005
  n_top: 100
  n_bottom: 100
  top_n: 100
  bottom_n: 100
  target_weight_band: 0.00010
  trade_cooldown_minutes: 2
  max_spread_bps: {float(args.max_spread_bps)}
  fee_bps: 0.5
  allow_short: true
"""
    cfg_path.write_text(cfg, encoding="utf-8")

    print("[saved alpha]", alpha_path)
    print("[saved pricing]", pricing_path)
    print("[saved config]", cfg_path)
    print("[shape]", df.shape)
    print(df.select(
        pl.len().alias("rows"),
        pl.col("date").n_unique().alias("dates"),
        pl.col("securityid").n_unique().alias("sids"),
        pl.col("minute").n_unique().alias("minutes"),
        pl.col(alpha_col).std().alias("alpha_std"),
        pl.col("spread_bps").mean().alias("avg_spread_bps"),
        pl.col("spread_bps").max().alias("max_spread_bps"),
    ))


if __name__ == "__main__":
    main()
