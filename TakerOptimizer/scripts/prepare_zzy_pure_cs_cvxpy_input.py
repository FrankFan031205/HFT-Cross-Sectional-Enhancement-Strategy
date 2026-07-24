# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import polars as pl


def valid_parquet_files(root: Path):
    files = []
    bad = []
    for p in sorted(root.glob("*/*/quotes.parquet")):
        try:
            if p.stat().st_size < 8:
                bad.append(str(p))
                continue
            with open(p, "rb") as f:
                head = f.read(4)
                f.seek(-4, 2)
                tail = f.read(4)
            if head == b"PAR1" and tail == b"PAR1":
                files.append(str(p))
            else:
                bad.append(str(p))
        except Exception:
            bad.append(str(p))
    return files, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/mnt/data1/zzy/optimizer_data")
    ap.add_argument("--model", choices=["res", "ts"], default="res")
    ap.add_argument("--horizon-min", type=int, default=20)
    ap.add_argument("--score-col", default="pred_z")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--price-multiplier", type=float, default=0.01)
    ap.add_argument("--minute-grid", type=int, default=1)
    ap.add_argument("--max-spread-bps", type=float, default=200.0)
    args = ap.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.model == "res":
        pred_path = data_root / "pred" / f"{args.horizon_min}min" / "test_predictions.parquet"
    else:
        pred_path = data_root / "pred_ts" / f"{args.horizon_min}min" / "test_predictions.parquet"

    quote_root = data_root / "quotes" / "TimeSeries"
    quote_files, bad_files = valid_parquet_files(quote_root)

    if not pred_path.exists():
        raise FileNotFoundError(pred_path)
    if not quote_files:
        raise RuntimeError(f"no valid quote parquet files under {quote_root}")

    print("[pred]", pred_path)
    print("[valid quote files]", len(quote_files))
    print("[bad quote files]", len(bad_files))
    for x in bad_files[:20]:
        print("  bad:", x)

    pred_schema = pl.scan_parquet(str(pred_path)).collect_schema()
    if args.score_col in pred_schema:
        score_col = args.score_col
    elif "pred_z" in pred_schema:
        score_col = "pred_z"
    else:
        score_col = "pred"

    # 用分钟网格，后面 optimizer 用 clock 方式每 20 分钟 rebalance
    grid_sec = int(args.minute_grid * 60)

    pred_lf = (
        pl.scan_parquet(str(pred_path))
        .with_columns([
            pl.col("date").cast(pl.Int64),
            pl.col("sid").cast(pl.Int64),
            pl.col("ts").cast(pl.Int64),
        ])
        .filter(pl.col("ts") % grid_sec == 0)
        .select([
            "date",
            "sid",
            "ts",
            pl.col(score_col).cast(pl.Float64).alias("pred_ret_h20"),
            pl.col("y_raw").cast(pl.Float64).alias("label_y_raw"),
        ])
    )

    quote_lf = (
        pl.scan_parquet(quote_files)
        .with_columns([
            pl.col("date").cast(pl.Int64),
            pl.col("sid").cast(pl.Int64),
            pl.col("ts").cast(pl.Int64),
        ])
        .filter(pl.col("ts") % grid_sec == 0)
        .select([
            "date", "sid", "ts", "ts_real",
            "task", "tbid", "tmid", "tavol", "tbvol", "vol",
        ])
    )

    ts_real_i = pl.col("ts_real").cast(pl.Int64)
    # 如果 ts_real 是 93000000 这种毫秒格式，除以 1000
    hhmmss = (
        pl.when(ts_real_i.max().over("date") > 235959)
        .then(ts_real_i // 1000)
        .otherwise(ts_real_i)
        .cast(pl.Int64)
    )
    time_s = hhmmss.cast(pl.Utf8).str.zfill(6)
    date_s = pl.col("date").cast(pl.Utf8)

    datetime_str = pl.concat_str([
        date_s.str.slice(0, 4), pl.lit("-"),
        date_s.str.slice(4, 2), pl.lit("-"),
        date_s.str.slice(6, 2), pl.lit(" "),
        time_s.str.slice(0, 2), pl.lit(":"),
        time_s.str.slice(2, 2), pl.lit(":"),
        time_s.str.slice(4, 2),
    ])

    joined = (
        pred_lf
        .join(quote_lf, on=["date", "sid", "ts"], how="inner")
        .with_columns([
            pl.when(pl.col("task") >= pl.col("tbid"))
              .then(pl.col("task"))
              .otherwise(pl.col("tbid"))
              .alias("ask_raw"),

            pl.when(pl.col("task") <= pl.col("tbid"))
              .then(pl.col("task"))
              .otherwise(pl.col("tbid"))
              .alias("bid_raw"),
        ])
        .with_columns([
            ((pl.col("ask_raw") + pl.col("bid_raw")) / 2.0).alias("mid_raw"),
        ])
        .with_columns([
            ((pl.col("ask_raw") - pl.col("bid_raw")) / pl.col("mid_raw") * 10000.0).alias("spread_bps"),
            (pl.col("mid_raw") * float(args.price_multiplier)).alias("mid_price"),
            (pl.col("bid_raw") * float(args.price_multiplier)).alias("bid1"),
            (pl.col("ask_raw") * float(args.price_multiplier)).alias("ask1"),
            pl.col("tbvol").fill_null(0.0).alias("bid_volume1"),
            pl.col("tavol").fill_null(0.0).alias("ask_volume1"),
            (pl.col("vol").fill_null(0.0) * pl.col("mid_raw") * float(args.price_multiplier)).alias("turnover_amount"),
            pl.col("sid").cast(pl.Utf8).str.zfill(6).alias("securityid"),
            datetime_str.alias("datetime"),
            hhmmss.alias("minute"),
        ])
        .filter(
            pl.col("pred_ret_h20").is_not_null()
            & pl.col("pred_ret_h20").is_finite()
            & (pl.col("mid_price") > 0)
            & (pl.col("bid1") > 0)
            & (pl.col("ask1") > 0)
            & (pl.col("ask1") >= pl.col("bid1"))
            & (pl.col("spread_bps") >= 0)
            & (pl.col("spread_bps") <= float(args.max_spread_bps))
        )
        .with_columns([
            (1.0 / pl.len().over(["date", "ts"])).alias("benchmark_weight"),
            pl.lit(None).cast(pl.Float64).alias("limit_up"),
            pl.lit(None).cast(pl.Float64).alias("limit_down"),
            pl.lit("unknown").alias("industry"),
        ])
        .select([
            "date", "datetime", "minute", "securityid", "sid", "ts",
            "pred_ret_h20", "label_y_raw",
            "mid_price", "bid1", "ask1", "bid_volume1", "ask_volume1",
            "spread_bps", "turnover_amount",
            "benchmark_weight",
            "limit_up", "limit_down", "industry",
        ])
        .sort(["date", "datetime", "securityid"])
    )

    print("[collect]")
    df = joined.collect(streaming=True)

    print("[rows]", df.height)
    print("[dates]", df.select([
        pl.col("date").min().alias("min_date"),
        pl.col("date").max().alias("max_date"),
        pl.col("date").n_unique().alias("n_dates"),
    ]))
    print("[snapshots]", df.select(["date", "datetime"]).unique().height)
    print("[symbols]", df.select(pl.col("securityid").n_unique().alias("n_symbols")))

    for d in df.select("date").unique().sort("date").to_series().to_list():
        sub = df.filter(pl.col("date") == d)
        out = out_dir / f"optimizer_input_{d}.parquet"
        sub.write_parquet(out)
        print("[saved]", out, "rows=", sub.height)

    print("[done]", out_dir)


if __name__ == "__main__":
    main()
