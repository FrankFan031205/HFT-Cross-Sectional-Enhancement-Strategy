import argparse
from pathlib import Path

import polars as pl
import yaml


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return repo_root() / p


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def bool_expr(col: str):
    return (
        pl.col(col)
        .cast(pl.Utf8)
        .str.to_lowercase()
        .is_in(["true", "1", "yes", "y"])
        .fill_null(False)
    )


def normalize_keys(lf: pl.LazyFrame, datetime_col: str, symbol_col: str) -> pl.LazyFrame:
    return lf.with_columns([
        pl.col(datetime_col).cast(pl.Utf8).alias(datetime_col),
        (
            pl.col(symbol_col)
            .cast(pl.Utf8)
            .str.replace(r"\.0$", "")
            .str.zfill(6)
            .alias(symbol_col)
        ),
    ])


def build_joined_lf(cfg: dict) -> pl.LazyFrame:
    quote_path = resolve_path(cfg["data"]["quote_path"])
    cs_signal_path = resolve_path(cfg["data"]["cs_signal_path"])

    col_cfg = cfg["columns"]
    datetime_col = col_cfg["datetime_col"]
    symbol_col = col_cfg["symbol_col"]
    quote_bid_col = col_cfg["quote_bid_col"]
    quote_ask_col = col_cfg["quote_ask_col"]

    cs_cols = cfg["cs_columns"]
    bid_filter_mode = cfg["logic"].get("bid_filter_mode", "long_candidate")
    keep_unmatched_original = bool(cfg["logic"].get("keep_unmatched_original", True))

    q_lf = pl.scan_csv(
        quote_path,
        infer_schema_length=10000,
        ignore_errors=True,
        null_values=["", "nan", "NaN", "NULL", "None"],
    )

    cs_lf = pl.scan_csv(
        cs_signal_path,
        infer_schema_length=10000,
        ignore_errors=True,
        null_values=["", "nan", "NaN", "NULL", "None"],
    )

    quote_schema = q_lf.schema
    cs_schema = cs_lf.schema

    required_quote_cols = [datetime_col, symbol_col, quote_bid_col, quote_ask_col]
    missing_quote = [c for c in required_quote_cols if c not in quote_schema]
    if missing_quote:
        raise ValueError(f"missing quote columns: {missing_quote}. Available: {list(quote_schema.keys())}")

    required_cs_cols = [datetime_col, symbol_col] + cs_cols
    missing_cs = [c for c in required_cs_cols if c not in cs_schema]
    if missing_cs:
        raise ValueError(f"missing cs columns: {missing_cs}. Available: {list(cs_schema.keys())}")

    q_lf = normalize_keys(q_lf, datetime_col, symbol_col)
    cs_lf = normalize_keys(cs_lf, datetime_col, symbol_col).select([datetime_col, symbol_col] + cs_cols)

    joined = q_lf.join(cs_lf, on=[datetime_col, symbol_col], how="left")

    joined = joined.with_columns([
        bool_expr(quote_bid_col).alias("quote_bid_original"),
        bool_expr(quote_ask_col).alias("quote_ask_original"),

        bool_expr("strong_long_candidate").alias("strong_long_candidate_bool"),
        bool_expr("long_candidate").alias("long_candidate_bool"),
        bool_expr("weak_long_candidate").alias("weak_long_candidate_bool"),
        bool_expr("avoid_candidate").alias("avoid_candidate_bool"),
        bool_expr("reduce_candidate").alias("reduce_candidate_bool"),
        bool_expr("strong_reduce_candidate").alias("strong_reduce_candidate_bool"),

        pl.col("cs_score_z").cast(pl.Float64, strict=False).alias("cs_score_z"),
        pl.col("cs_rank").cast(pl.Float64, strict=False).alias("cs_rank"),
        pl.col("target_inventory_scale").cast(pl.Float64, strict=False).fill_null(0.0).alias("target_inventory_scale"),
        pl.col("bid_priority").cast(pl.Float64, strict=False).fill_null(0.0).alias("bid_priority"),
        pl.col("ask_reduce_priority").cast(pl.Float64, strict=False).fill_null(0.0).alias("ask_reduce_priority"),
    ])

    if bid_filter_mode == "long_candidate":
        bid_filter = pl.col("long_candidate_bool")
    elif bid_filter_mode == "weak_long_candidate":
        bid_filter = pl.col("weak_long_candidate_bool")
    elif bid_filter_mode == "strong_long_candidate":
        bid_filter = pl.col("strong_long_candidate_bool")
    else:
        raise ValueError(f"unknown bid_filter_mode: {bid_filter_mode}")

    has_cs = pl.col("cs_rank").is_not_null()

    if keep_unmatched_original:
        quote_bid_cs_expr = pl.col("quote_bid_original") & ((~has_cs) | bid_filter)
    else:
        quote_bid_cs_expr = pl.col("quote_bid_original") & has_cs & bid_filter

    joined = joined.with_columns([
        has_cs.alias("has_cs_signal"),

        quote_bid_cs_expr.alias("quote_bid"),
        pl.col("quote_ask_original").alias("quote_ask"),

        (
            pl.col("quote_bid_original")
            & has_cs
            & (~bid_filter)
        ).alias("quote_bid_filtered_by_cs"),

        (
            pl.col("quote_bid_original")
            & has_cs
            & bid_filter
        ).alias("quote_bid_kept_by_cs"),

        (
            pl.col("quote_bid_original")
            & has_cs
            & pl.col("strong_long_candidate_bool")
        ).alias("quote_bid_cs_strong"),

        (
            pl.col("quote_bid_original")
            & has_cs
            & pl.col("weak_long_candidate_bool")
        ).alias("quote_bid_cs_weak"),

        (
            pl.col("reduce_candidate_bool")
            | pl.col("strong_reduce_candidate_bool")
        ).alias("cs_reduce_signal"),

        (
            pl.when(~has_cs)
            .then(pl.lit("no_cs_match"))
            .when(pl.col("quote_bid_original") & (~bid_filter))
            .then(pl.lit("bid_filtered_by_cs"))
            .when(pl.col("quote_bid_original") & bid_filter)
            .then(pl.lit("bid_kept_by_cs"))
            .otherwise(pl.lit("no_original_bid"))
        ).alias("cs_filter_status"),
    ])

    return joined


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="CrossSectionalModel/config/cs_merge_quotes_h60.yaml",
    )
    args = parser.parse_args()

    cfg = load_config(resolve_path(args.config))

    quote_path = resolve_path(cfg["data"]["quote_path"])
    cs_signal_path = resolve_path(cfg["data"]["cs_signal_path"])
    output_path = resolve_path(cfg["data"]["output_path"])
    summary_output_path = resolve_path(cfg["data"]["summary_output_path"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_output_path.parent.mkdir(parents=True, exist_ok=True)

    if not quote_path.exists():
        raise FileNotFoundError(f"quote file not found: {quote_path}")
    if not cs_signal_path.exists():
        raise FileNotFoundError(f"cs signal file not found: {cs_signal_path}")

    print(f"quote input: {quote_path}")
    print(f"cs input:    {cs_signal_path}")
    print(f"output:      {output_path}")

    joined = build_joined_lf(cfg)

    summary = joined.select([
        pl.len().alias("rows"),
        pl.col("has_cs_signal").mean().alias("cs_match_rate"),

        pl.col("quote_bid_original").mean().alias("quote_bid_original_rate"),
        pl.col("quote_bid").mean().alias("quote_bid_cs_rate"),
        pl.col("quote_bid_filtered_by_cs").mean().alias("quote_bid_filtered_rate"),

        pl.col("quote_ask_original").mean().alias("quote_ask_original_rate"),
        pl.col("quote_ask").mean().alias("quote_ask_cs_rate"),

        pl.when(pl.col("quote_bid_original") & pl.col("has_cs_signal"))
          .then(pl.col("cs_rank"))
          .otherwise(None)
          .mean()
          .alias("avg_cs_rank_original_bid_matched"),

        pl.when(pl.col("quote_bid") & pl.col("has_cs_signal"))
          .then(pl.col("cs_rank"))
          .otherwise(None)
          .mean()
          .alias("avg_cs_rank_cs_bid_matched"),

        pl.when(pl.col("quote_bid_original") & pl.col("has_cs_signal"))
          .then(pl.col("target_inventory_scale"))
          .otherwise(None)
          .mean()
          .alias("avg_target_inventory_original_bid_matched"),

        pl.when(pl.col("quote_bid") & pl.col("has_cs_signal"))
          .then(pl.col("target_inventory_scale"))
          .otherwise(None)
          .mean()
          .alias("avg_target_inventory_cs_bid_matched"),
    ]).collect(streaming=True)

    summary.write_csv(summary_output_path)

    print("\nOverall CS filter summary:")
    print(summary)

    group_summary = (
        joined
        .group_by("cs_group")
        .agg([
            pl.len().alias("rows"),
            pl.col("quote_bid_original").mean().alias("quote_bid_original_rate"),
            pl.col("quote_bid").mean().alias("quote_bid_cs_rate"),
            pl.col("quote_bid_filtered_by_cs").mean().alias("quote_bid_filtered_rate"),
            pl.col("quote_ask_original").mean().alias("quote_ask_original_rate"),
            pl.col("target_inventory_scale").mean().alias("avg_target_inventory_scale"),
            pl.col("cs_rank").mean().alias("avg_cs_rank"),
        ])
        .sort("avg_cs_rank", descending=True)
        .collect(streaming=True)
    )

    print("\nGroup summary:")
    print(group_summary)

    status_summary = (
        joined
        .group_by("cs_filter_status")
        .agg(pl.len().alias("rows"))
        .sort("rows", descending=True)
        .collect(streaming=True)
    )

    print("\nFilter status summary:")
    print(status_summary)

    print("\nWriting merged file in streaming mode...")
    joined.sink_csv(output_path)

    print(f"\nsaved merged quote file: {output_path}")
    print(f"saved summary: {summary_output_path}")


if __name__ == "__main__":
    main()
