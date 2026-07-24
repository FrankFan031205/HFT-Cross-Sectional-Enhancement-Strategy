import argparse
import sys
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


def is_numeric_dtype(dtype) -> bool:
    return dtype in {
        pl.Int8, pl.Int16, pl.Int32, pl.Int64,
        pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
        pl.Float32, pl.Float64
    }


def infer_feature_columns(schema: dict, cfg: dict) -> list[str]:
    col_cfg = cfg["columns"]
    feat_cfg = cfg["features"]

    datetime_col = col_cfg["datetime_col"]
    symbol_col = col_cfg["symbol_col"]
    label_col = col_cfg["label_col"]

    exclude = set(feat_cfg.get("exclude_columns", []))
    exclude.update({datetime_col, symbol_col, label_col})

    prefixes = feat_cfg.get("include_prefixes", [])

    feature_cols = []
    for col, dtype in schema.items():
        if col in exclude:
            continue
        if not is_numeric_dtype(dtype):
            continue
        if prefixes:
            if not any(col.startswith(p) for p in prefixes):
                continue
        feature_cols.append(col)

    return feature_cols


def zscore_expr(col: str, group_col: str, eps: float, clip_value: float):
    mean_expr = pl.col(col).mean().over(group_col)
    std_expr = pl.col(col).std().over(group_col)

    expr = (pl.col(col) - mean_expr) / (std_expr + eps)

    if clip_value is not None:
        expr = expr.clip(-clip_value, clip_value)

    return expr.fill_nan(None).alias(f"{col}_cs_z")


def rank_expr(col: str, group_col: str):
    return (
        pl.col(col).rank(method="average").over(group_col)
        / pl.col(col).count().over(group_col)
    ).fill_nan(None).alias(f"{col}_cs_rank")


def build_cs_dataset(cfg: dict):
    input_path = resolve_path(cfg["data"]["input_path"])
    output_path = resolve_path(cfg["data"]["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    datetime_col = cfg["columns"]["datetime_col"]
    symbol_col = cfg["columns"]["symbol_col"]
    label_col = cfg["columns"]["label_col"]

    cs_cfg = cfg["cross_section"]
    min_stocks = int(cs_cfg.get("min_stocks_per_datetime", 30))
    eps = float(cs_cfg.get("eps", 1e-8))
    clip_zscore = cs_cfg.get("clip_zscore", None)
    clip_zscore = None if clip_zscore is None else float(clip_zscore)

    create_feature_cs_z = bool(cs_cfg.get("create_feature_cs_z", True))
    create_feature_cs_rank = bool(cs_cfg.get("create_feature_cs_rank", False))
    create_label_cs_rank = bool(cs_cfg.get("create_label_cs_rank", True))
    create_label_cs_demean = bool(cs_cfg.get("create_label_cs_demean", True))

    keep_original_features = bool(cfg["output"].get("keep_original_features", False))

    if not input_path.exists():
        raise FileNotFoundError(f"input file not found: {input_path}")

    print(f"input:  {input_path}")
    print(f"output: {output_path}")

    lf = pl.scan_csv(
        input_path,
        infer_schema_length=10000,
        ignore_errors=True,
        null_values=["", "nan", "NaN", "NULL", "None"]
    )

    schema = lf.schema

    required_cols = [datetime_col, symbol_col, label_col]
    missing = [c for c in required_cols if c not in schema]
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    feature_cols = infer_feature_columns(schema, cfg)

    if not feature_cols:
        raise ValueError("no feature columns detected. Check include_prefixes or input schema.")

    print(f"detected feature columns: {len(feature_cols)}")
    print("first 20 features:")
    for c in feature_cols[:20]:
        print(f"  {c}")

    cast_exprs = []
    for c in feature_cols + [label_col]:
        cast_exprs.append(pl.col(c).cast(pl.Float64, strict=False).alias(c))

    lf = lf.with_columns(cast_exprs)

    base_cols = [datetime_col, symbol_col, label_col]
    selected_cols = base_cols + feature_cols

    lf = lf.select(selected_cols)

    lf = lf.with_columns(
        pl.col(symbol_col)
        .cast(pl.Utf8)
        .str.replace(r"\.0$", "")
        .str.zfill(6)
        .alias(symbol_col)
    )

    lf = lf.filter(pl.col(label_col).is_not_null()) 

    lf = lf.with_columns(
        pl.col(symbol_col).count().over(datetime_col).alias("__cs_stock_count")
    )

    lf = lf.filter(pl.col("__cs_stock_count") >= min_stocks)

    output_exprs = [
        pl.col(datetime_col),
        pl.col(symbol_col),
        pl.col(label_col),
        pl.col("__cs_stock_count")
    ]

    if create_label_cs_rank:
        output_exprs.append(
            (
                (
                    pl.col(label_col).rank(method="average").over(datetime_col)
                    / pl.col(label_col).count().over(datetime_col)
                )
                - 0.5
            ).mul(2.0).fill_nan(None).alias("label_cs_60_rank")
        )

    if create_label_cs_demean:
        output_exprs.append(
            (
                pl.col(label_col)
                - pl.col(label_col).mean().over(datetime_col)
            ).fill_nan(None).alias("label_cs_60_demean")
        )

    if keep_original_features:
        output_exprs.extend([pl.col(c) for c in feature_cols])

    if create_feature_cs_z:
        output_exprs.extend([
            zscore_expr(c, datetime_col, eps, clip_zscore)
            for c in feature_cols
        ])

    if create_feature_cs_rank:
        output_exprs.extend([
            rank_expr(c, datetime_col)
            for c in feature_cols
        ])

    out_lf = lf.select(output_exprs)

    df = out_lf.collect()

    print(f"output shape: {df.shape}")
    print(f"saving to: {output_path}")

    df.write_csv(output_path)

    print("done")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="CrossSectionalModel/config/cs_dataset_h60.yaml"
    )
    args = parser.parse_args()

    cfg_path = resolve_path(args.config)
    cfg = load_config(cfg_path)
    build_cs_dataset(cfg)


if __name__ == "__main__":
    main()
