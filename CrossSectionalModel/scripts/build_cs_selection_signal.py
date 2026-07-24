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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="CrossSectionalModel/config/cs_selection_h60.yaml",
    )
    args = parser.parse_args()

    cfg = load_config(resolve_path(args.config))

    score_path = resolve_path(cfg["data"]["score_path"])
    output_path = resolve_path(cfg["data"]["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not score_path.exists():
        raise FileNotFoundError(f"score file not found: {score_path}")

    col_cfg = cfg["columns"]
    sel_cfg = cfg["selection"]
    inv_cfg = cfg["target_inventory"]
    out_cfg = cfg["output"]

    datetime_col = col_cfg["datetime_col"]
    symbol_col = col_cfg["symbol_col"]
    label_col = col_cfg["label_col"]
    label_rank_col = col_cfg["label_rank_col"]
    label_excess_col = col_cfg["label_excess_col"]
    score_col = col_cfg["score_col"]
    rank_col = col_cfg["rank_col"]
    group_col = col_cfg["group_col"]

    keep_labels = bool(out_cfg.get("keep_labels", True))

    strong_long_th = float(sel_cfg["strong_long_threshold"])
    long_th = float(sel_cfg["long_threshold"])
    weak_long_th = float(sel_cfg["weak_long_threshold"])
    avoid_th = float(sel_cfg["avoid_threshold"])
    reduce_th = float(sel_cfg["reduce_threshold"])
    strong_reduce_th = float(sel_cfg["strong_reduce_threshold"])

    rank_90_inv = float(inv_cfg["rank_90"])
    rank_80_inv = float(inv_cfg["rank_80"])
    rank_70_inv = float(inv_cfg["rank_70"])
    rank_60_inv = float(inv_cfg["rank_60"])
    default_inv = float(inv_cfg["default"])

    needed_cols = [
        datetime_col,
        symbol_col,
        label_col,
        label_rank_col,
        label_excess_col,
        score_col,
        rank_col,
        group_col,
    ]

    lf = (
        pl.scan_csv(
            score_path,
            infer_schema_length=10000,
            ignore_errors=True,
            null_values=["", "nan", "NaN", "NULL", "None"],
        )
        .select(needed_cols)
        .with_columns([
            pl.col(symbol_col)
              .cast(pl.Utf8)
              .str.replace(r"\.0$", "")
              .str.zfill(6)
              .alias(symbol_col),

            pl.col(score_col).cast(pl.Float64, strict=False),
            pl.col(rank_col).cast(pl.Float64, strict=False),
            pl.col(label_col).cast(pl.Float64, strict=False),
            pl.col(label_rank_col).cast(pl.Float64, strict=False),
            pl.col(label_excess_col).cast(pl.Float64, strict=False),
        ])
    )

    lf = lf.with_columns([
        (pl.col(rank_col) >= strong_long_th).alias("strong_long_candidate"),
        (pl.col(rank_col) >= long_th).alias("long_candidate"),
        (pl.col(rank_col) >= weak_long_th).alias("weak_long_candidate"),

        (pl.col(rank_col) <= avoid_th).alias("avoid_candidate"),
        (pl.col(rank_col) <= reduce_th).alias("reduce_candidate"),
        (pl.col(rank_col) <= strong_reduce_th).alias("strong_reduce_candidate"),
    ])

    lf = lf.with_columns([
        (
            pl.when(pl.col(rank_col) >= 0.90)
            .then(pl.lit(rank_90_inv))
            .when(pl.col(rank_col) >= 0.80)
            .then(pl.lit(rank_80_inv))
            .when(pl.col(rank_col) >= 0.70)
            .then(pl.lit(rank_70_inv))
            .when(pl.col(rank_col) >= 0.60)
            .then(pl.lit(rank_60_inv))
            .otherwise(pl.lit(default_inv))
        ).alias("target_inventory_scale"),

        (
            pl.when(pl.col(rank_col) >= long_th)
            .then(pl.col(rank_col))
            .otherwise(pl.lit(0.0))
        ).alias("bid_priority"),

        (
            pl.when(pl.col(rank_col) <= reduce_th)
            .then(1.0 - pl.col(rank_col))
            .otherwise(pl.lit(0.0))
        ).alias("ask_reduce_priority"),
    ])

    output_cols = [
        datetime_col,
        symbol_col,
        score_col,
        rank_col,
        group_col,
        "strong_long_candidate",
        "long_candidate",
        "weak_long_candidate",
        "avoid_candidate",
        "reduce_candidate",
        "strong_reduce_candidate",
        "target_inventory_scale",
        "bid_priority",
        "ask_reduce_priority",
    ]

    if keep_labels:
        output_cols.extend([
            label_col,
            label_rank_col,
            label_excess_col,
        ])

    out = lf.select(output_cols).collect()
    out.write_csv(output_path)

    print(f"input:  {score_path}")
    print(f"output: {output_path}")
    print(f"shape:  {out.shape}")

    summary = (
        out.group_by(group_col)
        .agg([
            pl.len().alias("count"),
            pl.col(rank_col).mean().alias("avg_cs_rank"),
            pl.col("target_inventory_scale").mean().alias("avg_target_inventory_scale"),
            pl.col("bid_priority").mean().alias("avg_bid_priority"),
            pl.col("ask_reduce_priority").mean().alias("avg_ask_reduce_priority"),
            pl.col("long_candidate").mean().alias("long_candidate_rate"),
            pl.col("avoid_candidate").mean().alias("avoid_candidate_rate"),
            pl.col("reduce_candidate").mean().alias("reduce_candidate_rate"),
            pl.col(label_excess_col).mean().alias("avg_future_excess_return"),
        ])
        .sort("avg_cs_rank", descending=True)
    )

    print("\nSelection summary by group:")
    print(summary)

    print("\nDone.")


if __name__ == "__main__":
    main()
