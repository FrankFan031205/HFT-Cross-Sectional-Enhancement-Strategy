import argparse
from pathlib import Path

import pandas as pd
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


def select_factors_and_weights(cfg: dict) -> pd.DataFrame:
    stability_path = resolve_path(cfg["data"]["stability_path"])

    if not stability_path.exists():
        raise FileNotFoundError(f"stability file not found: {stability_path}")

    df = pd.read_csv(stability_path)

    factor_cfg = cfg["factors"]
    score_cfg = cfg["score"]

    numeric_cols = [
        "daily_mean_rankic_avg",
        "daily_rankic_ir",
        "positive_day_ratio",
        "daily_top20_excess_avg",
        "daily_bottom20_excess_avg",
        "worst_daily_rankic",
        "best_daily_rankic",
    ]

    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df[
        (df["positive_day_ratio"].fillna(0.0) >= float(factor_cfg.get("min_positive_day_ratio", 0.8)))
        & (df["daily_mean_rankic_avg"].fillna(-999.0) >= float(factor_cfg.get("min_daily_mean_rankic", 0.0)))
        & (df["daily_top20_excess_avg"].fillna(-999.0) >= float(factor_cfg.get("min_daily_top20_excess", 0.0)))
    ].copy()

    if df.empty:
        raise ValueError("no factor passed stability filters")

    sort_by = factor_cfg.get("sort_by", ["daily_mean_rankic_avg"])
    df = df.sort_values(sort_by, ascending=[False] * len(sort_by))

    mode = factor_cfg.get("mode", "top_n")

    if mode == "top_n":
        top_n = int(factor_cfg.get("top_n", 10))
        df = df.head(top_n).copy()
    else:
        factor_list = factor_cfg.get("factor_list", [])
        df = df[df["factor"].isin(factor_list)].copy()

    if df.empty:
        raise ValueError("no factor selected")

    weighting_method = score_cfg.get("weighting_method", "rankic")

    if weighting_method == "equal":
        df["raw_weight"] = 1.0
    elif weighting_method == "rankic":
        df["raw_weight"] = df["daily_mean_rankic_avg"].clip(lower=0.0)
    elif weighting_method == "ir":
        df["raw_weight"] = df["daily_rankic_ir"].clip(lower=0.0)
    elif weighting_method == "top_excess":
        df["raw_weight"] = df["daily_top20_excess_avg"].clip(lower=0.0)
    else:
        raise ValueError(f"unknown weighting_method: {weighting_method}")

    if df["raw_weight"].abs().sum() <= 1e-12:
        df["raw_weight"] = 1.0

    if bool(score_cfg.get("normalize_weights", True)):
        df["weight"] = df["raw_weight"] / df["raw_weight"].abs().sum()
    else:
        df["weight"] = df["raw_weight"]

    return df.reset_index(drop=True)


def build_score(cfg: dict, weight_df: pd.DataFrame):
    input_path = resolve_path(cfg["data"]["cs_dataset_path"])
    output_path = resolve_path(cfg["data"]["output_path"])
    weight_output_path = resolve_path(cfg["data"]["weight_output_path"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    weight_output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"cs dataset not found: {input_path}")

    col_cfg = cfg["columns"]
    score_cfg = cfg["score"]
    output_cfg = cfg["output"]

    datetime_col = col_cfg["datetime_col"]
    symbol_col = col_cfg["symbol_col"]
    label_col = col_cfg["label_col"]
    label_rank_col = col_cfg["label_rank_col"]
    label_excess_col = col_cfg["label_excess_col"]

    factor_cols = weight_df["factor"].tolist()
    weights = dict(zip(weight_df["factor"], weight_df["weight"]))

    keep_factor_values = bool(output_cfg.get("keep_factor_values", False))

    clip_factor_z = score_cfg.get("clip_factor_z", None)
    clip_score_z = score_cfg.get("clip_score_z", None)

    clip_factor_z = None if clip_factor_z is None else float(clip_factor_z)
    clip_score_z = None if clip_score_z is None else float(clip_score_z)

    needed_cols = [
        datetime_col,
        symbol_col,
        label_col,
        label_rank_col,
        label_excess_col,
    ] + factor_cols

    print(f"input:  {input_path}")
    print(f"output: {output_path}")
    print("\nSelected factors and weights:")
    print(weight_df[[
        "factor",
        "daily_mean_rankic_avg",
        "daily_rankic_ir",
        "positive_day_ratio",
        "daily_top20_excess_avg",
        "weight",
    ]].to_string(index=False))

    lf = (
        pl.scan_csv(
            input_path,
            infer_schema_length=10000,
            ignore_errors=True,
            null_values=["", "nan", "NaN", "NULL", "None"],
        )
        .select(needed_cols)
    )

    cast_exprs = [
        pl.col(label_col).cast(pl.Float64, strict=False).alias(label_col),
        pl.col(label_rank_col).cast(pl.Float64, strict=False).alias(label_rank_col),
        pl.col(label_excess_col).cast(pl.Float64, strict=False).alias(label_excess_col),
    ]

    for f in factor_cols:
        expr = pl.col(f).cast(pl.Float64, strict=False)
        if clip_factor_z is not None:
            expr = expr.clip(-clip_factor_z, clip_factor_z)
        cast_exprs.append(expr.alias(f))

    lf = lf.with_columns(cast_exprs)

    lf = lf.with_columns(
        pl.col(symbol_col)
        .cast(pl.Utf8)
        .str.replace(r"\.0$", "")
        .str.zfill(6)
        .alias(symbol_col)
    )

    score_expr = None

    for f in factor_cols:
        term = pl.col(f).fill_null(0.0) * float(weights[f])
        if score_expr is None:
            score_expr = term
        else:
            score_expr = score_expr + term

    lf = lf.with_columns(score_expr.alias("cs_score_raw"))

    score_mean = pl.col("cs_score_raw").mean().over(datetime_col)
    score_std = pl.col("cs_score_raw").std().over(datetime_col)

    score_z_expr = ((pl.col("cs_score_raw") - score_mean) / (score_std + 1e-8))

    if clip_score_z is not None:
        score_z_expr = score_z_expr.clip(-clip_score_z, clip_score_z)

    lf = lf.with_columns(score_z_expr.alias("cs_score_z"))

    lf = lf.with_columns(
        (
            pl.col("cs_score_z").rank(method="average").over(datetime_col)
            / pl.col("cs_score_z").count().over(datetime_col)
        ).alias("cs_rank")
    )

    lf = lf.with_columns(
        (
            pl.when(pl.col("cs_rank") >= 0.8)
            .then(pl.lit("top20"))
            .when(pl.col("cs_rank") <= 0.2)
            .then(pl.lit("bottom20"))
            .otherwise(pl.lit("middle"))
        ).alias("cs_group")
    )

    output_cols = [
        datetime_col,
        symbol_col,
        label_col,
        label_rank_col,
        label_excess_col,
        "cs_score_raw",
        "cs_score_z",
        "cs_rank",
        "cs_group",
    ]

    if keep_factor_values:
        output_cols.extend(factor_cols)

    out = lf.select(output_cols).collect()

    out.write_csv(output_path)
    weight_df.to_csv(weight_output_path, index=False)

    print(f"\noutput shape: {out.shape}")
    print(f"saved score:   {output_path}")
    print(f"saved weights: {weight_output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="CrossSectionalModel/config/cs_score_h60.yaml",
    )
    args = parser.parse_args()

    cfg = load_config(resolve_path(args.config))

    weight_df = select_factors_and_weights(cfg)
    build_score(cfg, weight_df)


if __name__ == "__main__":
    main()
