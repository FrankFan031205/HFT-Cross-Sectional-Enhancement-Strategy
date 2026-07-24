import argparse
from pathlib import Path

import polars as pl
import pandas as pd
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


def infer_factor_columns(schema: dict, cfg: dict) -> list[str]:
    feat_cfg = cfg["features"]
    suffix = feat_cfg.get("include_suffix", "_cs_z")
    exclude = set(feat_cfg.get("exclude_columns", []))

    factor_cols = []
    for col in schema:
        if col in exclude:
            continue
        if col.endswith(suffix):
            factor_cols.append(col)

    return factor_cols


def infer_category(factor: str) -> str:
    name = factor.lower()

    if "cancel" in name:
        return "cancel_pressure"
    if "trade" in name or "vwap" in name or "amount" in name or "active" in name:
        return "trade_flow"
    if "order" in name or "aggressive" in name or "fill" in name:
        return "order_flow"
    if "spread" in name or "microprice" in name:
        return "microprice_spread"
    if "depth" in name or "book" in name or "obi" in name:
        return "order_book"
    if "volatility" in name or "ret" in name:
        return "return_volatility"
    return "other"


def safe_ir(mean_value, std_value):
    if std_value is None or pd.isna(std_value) or abs(std_value) < 1e-12:
        return None
    return mean_value / std_value


def evaluate_batch(
    input_path: Path,
    batch_factors: list[str],
    cfg: dict,
) -> list[dict]:
    col_cfg = cfg["columns"]
    eval_cfg = cfg["evaluation"]

    datetime_col = col_cfg["datetime_col"]
    label_col = col_cfg["label_col"]
    label_rank_col = col_cfg["label_rank_col"]
    label_excess_col = col_cfg["label_excess_col"]

    top_q = float(eval_cfg.get("top_quantile", 0.2))
    bottom_q = float(eval_cfg.get("bottom_quantile", 0.2))
    min_obs = int(eval_cfg.get("min_obs_per_datetime", 30))

    top_cut = 1.0 - top_q
    bottom_cut = bottom_q

    needed_cols = [datetime_col, label_col, label_rank_col, label_excess_col] + batch_factors

    lf = pl.scan_csv(
        input_path,
        infer_schema_length=10000,
        ignore_errors=True,
        null_values=["", "nan", "NaN", "NULL", "None"]
    ).select(needed_cols)

    cast_exprs = [
        pl.col(label_col).cast(pl.Float64, strict=False),
        pl.col(label_rank_col).cast(pl.Float64, strict=False),
        pl.col(label_excess_col).cast(pl.Float64, strict=False),
    ]

    for f in batch_factors:
        cast_exprs.append(pl.col(f).cast(pl.Float64, strict=False))

    lf = lf.with_columns(cast_exprs)

    lf = lf.filter(
        pl.col(label_col).is_not_null()
        & pl.col(label_rank_col).is_not_null()
        & pl.col(label_excess_col).is_not_null()
    )

    rank_exprs = []
    for f in batch_factors:
        rank_col = f"{f}__rank"
        rank_exprs.append(
            (
                pl.col(f).rank(method="average").over(datetime_col)
                / pl.col(f).count().over(datetime_col)
            ).alias(rank_col)
        )

    lf_ranked = lf.with_columns(rank_exprs)

    agg_exprs = []

    for f in batch_factors:
        rank_col = f"{f}__rank"

        valid = pl.col(f).is_not_null() & pl.col(rank_col).is_not_null()

        agg_exprs.extend([
            pl.when(valid).then(pl.col(f)).count().alias(f"{f}__nobs"),

            pl.corr(rank_col, label_rank_col).alias(f"{f}__rankic"),

            pl.when(pl.col(rank_col) >= top_cut)
              .then(pl.col(label_col))
              .otherwise(None)
              .mean()
              .alias(f"{f}__top_ret"),

            pl.when(pl.col(rank_col) >= top_cut)
              .then(pl.col(label_excess_col))
              .otherwise(None)
              .mean()
              .alias(f"{f}__top_excess"),

            pl.when(pl.col(rank_col) <= bottom_cut)
              .then(pl.col(label_col))
              .otherwise(None)
              .mean()
              .alias(f"{f}__bottom_ret"),

            pl.when(pl.col(rank_col) <= bottom_cut)
              .then(pl.col(label_excess_col))
              .otherwise(None)
              .mean()
              .alias(f"{f}__bottom_excess"),
        ])

    by_time = (
        lf_ranked
        .group_by(datetime_col)
        .agg(agg_exprs)
        .filter(
            pl.max_horizontal([pl.col(f"{f}__nobs") for f in batch_factors]) >= min_obs
        )
        .collect()
    )

    quality_exprs = []

    for f in batch_factors:
        quality_exprs.extend([
            pl.col(f).is_null().mean().alias(f"{f}__missing_ratio"),
            pl.when(pl.col(f).is_not_null())
              .then(pl.col(f) == 0)
              .otherwise(None)
              .mean()
              .alias(f"{f}__zero_ratio"),
            pl.col(f).std().alias(f"{f}__std_all"),
        ])

    quality = lf.select(quality_exprs).collect().to_dicts()[0]

    pdf = by_time.to_pandas()

    results = []

    for f in batch_factors:
        ic_col = f"{f}__rankic"
        nobs_col = f"{f}__nobs"

        ic = pdf[ic_col].dropna()

        mean_rankic = ic.mean() if len(ic) else None
        std_rankic = ic.std(ddof=1) if len(ic) > 1 else None
        rankic_ir = safe_ir(mean_rankic, std_rankic) if mean_rankic is not None else None
        positive_ratio = (ic > 0).mean() if len(ic) else None

        top_ret = pdf[f"{f}__top_ret"].mean()
        top_excess = pdf[f"{f}__top_excess"].mean()
        bottom_ret = pdf[f"{f}__bottom_ret"].mean()
        bottom_excess = pdf[f"{f}__bottom_excess"].mean()

        result = {
            "factor": f,
            "category": infer_category(f),
            "mean_rankic": mean_rankic,
            "std_rankic": std_rankic,
            "rankic_ir": rankic_ir,
            "positive_ratio": positive_ratio,
            "valid_ic_count": int(len(ic)),
            "avg_nobs_per_datetime": pdf[nobs_col].mean(),
            "top20_return": top_ret,
            "top20_excess_return": top_excess,
            "bottom20_return": bottom_ret,
            "bottom20_excess_return": bottom_excess,
            "top_minus_bottom_return": top_ret - bottom_ret if pd.notna(top_ret) and pd.notna(bottom_ret) else None,
            "top_minus_bottom_excess": top_excess - bottom_excess if pd.notna(top_excess) and pd.notna(bottom_excess) else None,
            "missing_ratio": quality.get(f"{f}__missing_ratio"),
            "zero_ratio": quality.get(f"{f}__zero_ratio"),
            "std_all": quality.get(f"{f}__std_all"),
        }

        results.append(result)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="CrossSectionalModel/config/cs_eval_h60.yaml"
    )
    args = parser.parse_args()

    cfg_path = resolve_path(args.config)
    cfg = load_config(cfg_path)

    input_path = resolve_path(cfg["data"]["input_path"])
    output_path = resolve_path(cfg["data"]["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"input file not found: {input_path}")

    lf = pl.scan_csv(
        input_path,
        infer_schema_length=10000,
        ignore_errors=True,
        null_values=["", "nan", "NaN", "NULL", "None"]
    )

    schema = lf.schema
    factor_cols = infer_factor_columns(schema, cfg)

    if not factor_cols:
        raise ValueError("no factor columns detected")

    batch_size = int(cfg["evaluation"].get("batch_size", 6))

    print(f"input: {input_path}")
    print(f"output: {output_path}")
    print(f"detected factors: {len(factor_cols)}")
    print(f"batch size: {batch_size}")

    all_results = []

    for i in range(0, len(factor_cols), batch_size):
        batch = factor_cols[i:i + batch_size]
        print(f"\nEvaluating batch {i // batch_size + 1}: {batch}")
        batch_results = evaluate_batch(input_path, batch, cfg)
        all_results.extend(batch_results)

    out = pd.DataFrame(all_results)

    out = out.sort_values(
        ["mean_rankic", "top20_excess_return"],
        ascending=[False, False]
    )

    out.to_csv(output_path, index=False)

    print("\nTop 20 factors by mean_rankic:")
    print(out.head(20)[[
        "factor",
        "category",
        "mean_rankic",
        "rankic_ir",
        "positive_ratio",
        "top20_excess_return",
        "bottom20_excess_return",
        "missing_ratio",
        "zero_ratio"
    ]])

    print(f"\nsaved: {output_path}")


if __name__ == "__main__":
    main()
