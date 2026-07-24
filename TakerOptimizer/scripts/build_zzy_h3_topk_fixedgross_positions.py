# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import polars as pl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--top-n", type=int, default=100)
    ap.add_argument("--target-gross", type=float, default=0.10)
    ap.add_argument("--single-name-limit", type=float, default=0.003)
    ap.add_argument("--capital", type=float, default=200_000_000.0)
    ap.add_argument("--rebalance-ts-step", type=int, default=180)
    ap.add_argument("--max-spread-bps", type=float, default=50.0)
    ap.add_argument("--price-multiplier", type=float, default=0.01)
    ap.add_argument("--capacity-cap-scale", type=float, default=1.0)
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    files = [str(x) for x in sorted(in_dir.glob("optimizer_input_*.parquet"))]
    if not files:
        raise SystemExit(f"no optimizer_input_*.parquet under {in_dir}")

    lf0 = pl.scan_parquet(files)
    max_ts_real = lf0.select(pl.col("ts_real").cast(pl.Int64).max()).collect().item()
    div = 1000 if max_ts_real > 235959 else 1

    date_s = pl.col("date").cast(pl.Utf8)
    hhmmss = (pl.col("ts_real").cast(pl.Int64) // div).cast(pl.Int64)
    time_s = hhmmss.cast(pl.Utf8).str.zfill(6)

    datetime_str = pl.concat_str([
        date_s.str.slice(0, 4), pl.lit("-"),
        date_s.str.slice(4, 2), pl.lit("-"),
        date_s.str.slice(6, 2), pl.lit(" "),
        time_s.str.slice(0, 2), pl.lit(":"),
        time_s.str.slice(2, 2), pl.lit(":"),
        time_s.str.slice(4, 2),
    ])

    base_w = float(args.target_gross) / float(args.top_n)

    df = (
        lf0
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
            pl.col("date").cast(pl.Int64).alias("execution_date"),
            datetime_str.alias("execution_datetime"),
            hhmmss.alias("minute"),
            pl.col("SecurityID").cast(pl.Utf8).str.zfill(6).alias("securityid"),

            (pl.col("exec_mid_price") * float(args.price_multiplier)).alias("mid_price"),
            (pl.col("exec_sell_price") * float(args.price_multiplier)).alias("bid_price"),
            (pl.col("exec_buy_price") * float(args.price_multiplier)).alias("ask_price"),

            pl.col("exec_spread_bps").alias("spread_bps"),
            pl.col("exec_spread_bps").alias("spread_bps_realized"),
            pl.col("signal_z").alias("score"),
            pl.col("signal_z").rank(method="ordinal", descending=True).over(["date", "ts"]).alias("rank"),
            (
                pl.col("max_participation_notional")
                * float(args.price_multiplier)
                * float(args.capacity_cap_scale)
                / float(args.capital)
            ).alias("capacity_weight"),
        ])
        .with_columns([
            (
                (pl.col("rank") <= int(args.top_n))
                & (pl.col("capacity_weight") > 0)
            ).alias("selected")
        ])
        .with_columns([
            pl.when(pl.col("selected"))
            .then(
                pl.min_horizontal([
                    pl.lit(base_w),
                    pl.lit(float(args.single_name_limit)),
                    pl.col("capacity_weight"),
                ])
            )
            .otherwise(0.0)
            .alias("effective_target_weight")
        ])
        .with_columns([
            pl.col("effective_target_weight").alias("target_weight"),
            pl.col("effective_target_weight").alias("desired_target_weight"),
            pl.col("effective_target_weight").alias("gross_weight"),
            pl.lit("TOPK_FIXED_GROSS").alias("state"),
            pl.when(pl.col("effective_target_weight") > 0).then(pl.lit("BUY")).otherwise(pl.lit("NONE")).alias("side"),
            pl.lit("none").alias("blocked_reason"),
        ])
        .select([
            "execution_date", "execution_datetime",
            "date", "minute", "securityid",
            "mid_price", "bid_price", "ask_price",
            "spread_bps", "spread_bps_realized",
            "effective_target_weight", "target_weight", "desired_target_weight",
            "gross_weight", "selected", "state", "side", "blocked_reason",
            "score", "rank", "capacity_weight",
        ])
        .sort(["execution_datetime", "securityid"])
        .collect()
    )

    df.write_csv(out)

    g = df.group_by(["execution_date", "minute"]).agg([
        pl.col("gross_weight").sum().alias("gross"),
        pl.col("selected").cast(pl.Int64).sum().alias("n_hold"),
    ])

    print("[saved]", out)
    print("[shape]", df.shape)
    print("[date-time]", df.select(["execution_date", "minute"]).unique().height)
    print("[gross]")
    print(g.select([
        pl.col("gross").mean().alias("avg_gross"),
        pl.col("gross").min().alias("min_gross"),
        pl.col("gross").max().alias("max_gross"),
        pl.col("n_hold").mean().alias("avg_n_hold"),
    ]))


if __name__ == "__main__":
    main()
