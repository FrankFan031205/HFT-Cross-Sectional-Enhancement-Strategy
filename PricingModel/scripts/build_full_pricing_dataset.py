import os
import csv
import argparse
import polars as pl


def get_csv_columns(path):
    with open(path, "r") as f:
        reader = csv.reader(f)
        return next(reader)


def scan_csv(path):
    return pl.scan_csv(
        path,
        schema_overrides={
            "securityid": pl.Utf8,
            "datetime": pl.Utf8,
        },
        ignore_errors=True,
    )


def ensure_dir(path):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def add_missing_cols(lf, required_cols, zero_cols=None):
    zero_cols = set(zero_cols or [])
    existing = set(lf.collect_schema().names())

    exprs = []
    for c in required_cols:
        if c not in existing:
            if c in zero_cols:
                exprs.append(pl.lit(0.0).alias(c))
            else:
                exprs.append(pl.lit(None).alias(c))

    if exprs:
        lf = lf.with_columns(exprs)

    return lf


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--market",
        default="PricingModel/data/market_return_202410_100.csv",
    )
    parser.add_argument(
        "--hidden_attention",
        default="FactorModel/outputs/hidden_factor_attention_h60_202410_100.csv",
    )
    parser.add_argument(
        "--raw_factor",
        default="FactorModel/data/raw/factor_features_202410_100.csv",
    )
    parser.add_argument(
        "--output",
        default="PricingModel/data/pricing_dataset_h60_202410_100_full.csv",
    )
    parser.add_argument("--tick_size", type=float, default=0.01)

    args = parser.parse_args()

    key_cols = ["date", "datetime", "securityid"]
    tick_size = args.tick_size
    eps = 1e-12

    print("loading market:", args.market)
    market_cols_all = get_csv_columns(args.market)

    market = scan_csv(args.market)

    print("loading hidden attention:", args.hidden_attention)
    hidden = scan_csv(args.hidden_attention).select(
        key_cols + ["hidden_factor_attention_h60"]
    )

    df = market.join(hidden, on=key_cols, how="left")

    if os.path.exists(args.raw_factor):
        print("loading raw factor proxy:", args.raw_factor)

        raw_cols_all = get_csv_columns(args.raw_factor)

        proxy_map = {
            "fwz2_trade_imbalance_10": "trade_imbalance",
            "fwz2_amount_imbalance_10": "amount_imbalance",
            "fwz2_trade_count_imbalance_10": "trade_count_imbalance",
            "fwz2_trade_intensity_10": "trade_intensity",
            "fwz2_large_trade_imbalance_10": "large_trade_imbalance",
            "fwz2_cancel_pressure_10": "cancel_pressure_imbalance",
            "fwz2_near_cancel_pressure_10": "near_cancel_pressure_imbalance",
            "fwz2_buy_cancel_ratio_10": "buy_cancel_ratio",
            "fwz2_sell_cancel_ratio_10": "sell_cancel_ratio",
            "fwz2_cancel_ratio_imbalance_10": "cancel_ratio_imbalance",
        }

        available_raw = [c for c in proxy_map if c in raw_cols_all]

        if available_raw:
            raw = scan_csv(args.raw_factor).select(key_cols + available_raw)

            rename_map = {c: proxy_map[c] for c in available_raw}
            raw = raw.rename(rename_map)

            df = df.join(raw, on=key_cols, how="left")
        else:
            print("no proxy columns found in raw factor file")
    else:
        print("raw factor file not found, skip proxy merge")

    price_cols = []
    volume_cols = []

    for level in range(1, 11):
        price_cols.extend([f"bid{level}", f"ask{level}"])
        volume_cols.extend([f"bid{level}_volume", f"ask{level}_volume"])

    must_have_zero = [
        "active_buy_volume",
        "active_sell_volume",
        "active_buy_amount",
        "active_sell_amount",
        "trade_count_buy",
        "trade_count_sell",
        "trade_imbalance",
        "amount_imbalance",
        "trade_count_imbalance",
        "trade_intensity",
        "large_trade_imbalance",
        "bid_cancel_volume",
        "ask_cancel_volume",
        "bid_cancel_amount",
        "ask_cancel_amount",
        "cancel_pressure_bid",
        "cancel_pressure_ask",
        "cancel_pressure_imbalance",
        "near_cancel_pressure_bid",
        "near_cancel_pressure_ask",
        "near_cancel_pressure_imbalance",
        "buy_cancel_ratio",
        "sell_cancel_ratio",
        "cancel_ratio_imbalance",
    ]

    base_required = (
        key_cols
        + price_cols
        + volume_cols
        + [
            "mid_price",
            "spread",
            "limit_up_price",
            "limit_down_price",
            "hidden_factor_attention_h60",
        ]
    )

    df = add_missing_cols(
        df,
        base_required + must_have_zero,
        zero_cols=volume_cols + must_have_zero,
    )

    # cast numeric
    numeric_cols = [
        c for c in (
            price_cols
            + volume_cols
            + [
                "mid_price",
                "spread",
                "limit_up_price",
                "limit_down_price",
                "hidden_factor_attention_h60",
            ]
            + must_have_zero
        )
        if c in df.collect_schema().names()
    ]

    df = df.with_columns([
        pl.col(c).cast(pl.Float64, strict=False).alias(c)
        for c in numeric_cols
    ])

    # core book derived fields
    df = df.with_columns([
        (pl.col("ask1") - pl.col("bid1")).alias("spread"),
        ((pl.col("ask1") - pl.col("bid1")) / tick_size).alias("spread_ticks"),
        (
            (pl.col("ask1") * pl.col("bid1_volume") + pl.col("bid1") * pl.col("ask1_volume"))
            / (pl.col("bid1_volume") + pl.col("ask1_volume") + eps)
        ).alias("microprice"),
    ])

    df = df.with_columns([
        ((pl.col("microprice") - pl.col("mid_price")) / tick_size).alias("microprice_shift_ticks"),
    ])

    # depth and imbalance
    for n in [1, 5, 10]:
        bid_depth_expr = sum(pl.col(f"bid{i}_volume") for i in range(1, n + 1))
        ask_depth_expr = sum(pl.col(f"ask{i}_volume") for i in range(1, n + 1))

        df = df.with_columns([
            bid_depth_expr.alias(f"bid_depth_{n}"),
            ask_depth_expr.alias(f"ask_depth_{n}"),
        ])

        df = df.with_columns([
            (pl.col(f"bid_depth_{n}") + pl.col(f"ask_depth_{n}")).alias(f"total_depth_{n}"),
            (
                (pl.col(f"bid_depth_{n}") - pl.col(f"ask_depth_{n}"))
                / (pl.col(f"bid_depth_{n}") + pl.col(f"ask_depth_{n}") + eps)
            ).alias(f"book_imbalance_{n}"),
        ])

    # sort before rolling / shifts
    df = df.sort(["date", "securityid", "datetime"])

    # historical returns for state features
    for h in [1, 5, 10, 20]:
        df = df.with_columns([
            (
                pl.col("mid_price") / pl.col("mid_price").shift(h).over(["date", "securityid"]) - 1.0
            ).alias(f"ret_{h}")
        ])

    # volatility
    df = df.with_columns([
        pl.col("ret_1").rolling_std(window_size=20).over(["date", "securityid"]).alias("volatility_20"),
        pl.col("ret_1").rolling_std(window_size=60).over(["date", "securityid"]).alias("volatility_60"),
        (
            (pl.col("mid_price") - pl.col("mid_price").shift(1).over(["date", "securityid"])) / tick_size
        ).alias("_mid_move_ticks"),
    ])

    df = df.with_columns([
        pl.col("_mid_move_ticks").rolling_std(window_size=20).over(["date", "securityid"]).alias("volatility_ticks")
    ])

    # simple liquidity state
    df = df.with_columns([
        pl.when(pl.col("spread_ticks") >= 3)
        .then(pl.lit("wide_spread"))
        .when(pl.col("total_depth_5") <= 0)
        .then(pl.lit("thin_book"))
        .otherwise(pl.lit("normal"))
        .alias("liquidity_state")
    ])

    required_cols = [
        "date",
        "datetime",
        "securityid",

        "bid1", "bid2", "bid3", "bid4", "bid5",
        "bid6", "bid7", "bid8", "bid9", "bid10",

        "ask1", "ask2", "ask3", "ask4", "ask5",
        "ask6", "ask7", "ask8", "ask9", "ask10",

        "bid1_volume", "bid2_volume", "bid3_volume", "bid4_volume", "bid5_volume",
        "bid6_volume", "bid7_volume", "bid8_volume", "bid9_volume", "bid10_volume",

        "ask1_volume", "ask2_volume", "ask3_volume", "ask4_volume", "ask5_volume",
        "ask6_volume", "ask7_volume", "ask8_volume", "ask9_volume", "ask10_volume",

        "mid_price",
        "spread",
        "spread_ticks",

        "limit_up_price",
        "limit_down_price",

        "hidden_factor_attention_h60",

        "microprice",
        "microprice_shift_ticks",

        "bid_depth_1",
        "ask_depth_1",
        "total_depth_1",
        "book_imbalance_1",

        "bid_depth_5",
        "ask_depth_5",
        "total_depth_5",
        "book_imbalance_5",

        "bid_depth_10",
        "ask_depth_10",
        "total_depth_10",
        "book_imbalance_10",

        "active_buy_volume",
        "active_sell_volume",
        "active_buy_amount",
        "active_sell_amount",
        "trade_count_buy",
        "trade_count_sell",
        "trade_imbalance",
        "amount_imbalance",
        "trade_count_imbalance",
        "trade_intensity",
        "large_trade_imbalance",

        "bid_cancel_volume",
        "ask_cancel_volume",
        "bid_cancel_amount",
        "ask_cancel_amount",
        "cancel_pressure_bid",
        "cancel_pressure_ask",
        "cancel_pressure_imbalance",
        "near_cancel_pressure_bid",
        "near_cancel_pressure_ask",
        "near_cancel_pressure_imbalance",
        "buy_cancel_ratio",
        "sell_cancel_ratio",
        "cancel_ratio_imbalance",

        "ret_1",
        "ret_5",
        "ret_10",
        "ret_20",
        "volatility_20",
        "volatility_60",
        "volatility_ticks",

        "liquidity_state",
    ]

    # preserve useful extra columns if they exist
    extra_cols = [
        "time",
        "ret_30",
        "ret_60",
        "ret_90",
        "ret_120",
        "label_30",
        "label_60",
        "label_90",
        "label_120",
    ]

    existing = set(df.collect_schema().names())
    output_cols = required_cols + [c for c in extra_cols if c in existing and c not in required_cols]

    df = add_missing_cols(df, required_cols, zero_cols=must_have_zero)

    ensure_dir(args.output)

    print("writing:", args.output)
    df.select(output_cols).sink_csv(args.output)

    print("saved:", args.output)
    print("required columns:", len(required_cols))
    print("output columns:", len(output_cols))


if __name__ == "__main__":
    main()