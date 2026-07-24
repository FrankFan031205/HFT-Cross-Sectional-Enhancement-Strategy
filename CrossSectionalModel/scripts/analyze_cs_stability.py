import argparse
from pathlib import Path

import numpy as np
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


def safe_ir(mean_value, std_value):
    if std_value is None or pd.isna(std_value) or abs(std_value) < 1e-12:
        return np.nan
    return mean_value / std_value


def infer_date_time_columns(df: pd.DataFrame, datetime_col: str) -> pd.DataFrame:
    out = df.copy()
    dt = out[datetime_col].astype(str)

    out["date"] = dt.str.slice(0, 10)

    if " " in dt.iloc[0]:
        out["time"] = dt.str.split(" ").str[-1].str.slice(0, 8)
    else:
        out["time"] = dt.str.slice(11, 19)

    return out


def assign_intraday_bucket(time_str: str, buckets: list[dict]) -> str:
    if pd.isna(time_str):
        return "unknown"

    t = str(time_str)[:8]

    for b in buckets:
        if b["start"] <= t < b["end"]:
            return b["name"]

    return "other"


def select_factors(cfg: dict) -> list[str]:
    eval_path = resolve_path(cfg["data"]["eval_path"])
    factor_cfg = cfg["factors"]

    if not eval_path.exists():
        raise FileNotFoundError(f"eval file not found: {eval_path}")

    df = pd.read_csv(eval_path)

    suffix = factor_cfg.get("include_suffix", "_cs_z")
    df = df[df["factor"].astype(str).str.endswith(suffix)].copy()

    sort_by = factor_cfg.get("sort_by", ["mean_rankic"])
    for c in sort_by:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.sort_values(sort_by, ascending=[False] * len(sort_by))

    mode = factor_cfg.get("mode", "top_n")
    if mode == "top_n":
        top_n = int(factor_cfg.get("top_n", 20))
        factors = df["factor"].head(top_n).tolist()
    else:
        factors = factor_cfg.get("factor_list", [])

    if not factors:
        raise ValueError("no factors selected")

    return factors


def evaluate_batch(input_path: Path, batch_factors: list[str], cfg: dict) -> pd.DataFrame:
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

    lf = lf.with_columns(rank_exprs)

    agg_exprs = []

    for f in batch_factors:
        rank_col = f"{f}__rank"

        valid = pl.col(f).is_not_null() & pl.col(rank_col).is_not_null()

        agg_exprs.extend([
            pl.when(valid).then(pl.col(f)).count().alias(f"{f}__nobs"),

            pl.corr(rank_col, label_rank_col).alias(f"{f}__rankic"),

            pl.when(pl.col(rank_col) >= top_cut)
              .then(pl.col(label_excess_col))
              .otherwise(None)
              .mean()
              .alias(f"{f}__top20_excess_return"),

            pl.when(pl.col(rank_col) <= bottom_cut)
              .then(pl.col(label_excess_col))
              .otherwise(None)
              .mean()
              .alias(f"{f}__bottom20_excess_return"),
        ])

    by_time = (
        lf.group_by(datetime_col)
        .agg(agg_exprs)
        .filter(
            pl.max_horizontal([pl.col(f"{f}__nobs") for f in batch_factors]) >= min_obs
        )
        .collect()
    )

    return by_time.to_pandas()


def wide_to_long(by_time: pd.DataFrame, batch_factors: list[str], datetime_col: str) -> pd.DataFrame:
    rows = []

    for f in batch_factors:
        tmp = by_time[[
            datetime_col,
            f"{f}__nobs",
            f"{f}__rankic",
            f"{f}__top20_excess_return",
            f"{f}__bottom20_excess_return",
        ]].copy()

        tmp.columns = [
            datetime_col,
            "nobs",
            "rankic",
            "top20_excess_return",
            "bottom20_excess_return",
        ]

        tmp["factor"] = f
        rows.append(tmp)

    return pd.concat(rows, axis=0, ignore_index=True)


def summarize_stability(long_df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    datetime_col = cfg["columns"]["datetime_col"]
    buckets = cfg.get("intraday_buckets", [])

    df = infer_date_time_columns(long_df, datetime_col)
    df["intraday_bucket"] = df["time"].apply(lambda x: assign_intraday_bucket(x, buckets))

    daily = (
        df.groupby(["factor", "date"])
        .agg(
            daily_mean_rankic=("rankic", "mean"),
            daily_rankic_std=("rankic", "std"),
            daily_positive_ratio=("rankic", lambda x: (x > 0).mean()),
            daily_top20_excess=("top20_excess_return", "mean"),
            daily_bottom20_excess=("bottom20_excess_return", "mean"),
            datetime_count=("rankic", "count"),
            avg_nobs=("nobs", "mean"),
        )
        .reset_index()
    )

    bucket = (
        df.groupby(["factor", "intraday_bucket"])
        .agg(
            bucket_mean_rankic=("rankic", "mean"),
            bucket_rankic_std=("rankic", "std"),
            bucket_positive_ratio=("rankic", lambda x: (x > 0).mean()),
            bucket_top20_excess=("top20_excess_return", "mean"),
            bucket_bottom20_excess=("bottom20_excess_return", "mean"),
            datetime_count=("rankic", "count"),
            avg_nobs=("nobs", "mean"),
        )
        .reset_index()
    )

    summary = (
        daily.groupby("factor")
        .agg(
            day_count=("date", "count"),
            daily_mean_rankic_avg=("daily_mean_rankic", "mean"),
            daily_mean_rankic_std=("daily_mean_rankic", "std"),
            positive_day_ratio=("daily_mean_rankic", lambda x: (x > 0).mean()),
            daily_top20_excess_avg=("daily_top20_excess", "mean"),
            daily_bottom20_excess_avg=("daily_bottom20_excess", "mean"),
            worst_daily_rankic=("daily_mean_rankic", "min"),
            best_daily_rankic=("daily_mean_rankic", "max"),
            avg_datetime_count_per_day=("datetime_count", "mean"),
        )
        .reset_index()
    )

    summary["daily_rankic_ir"] = summary.apply(
        lambda r: safe_ir(r["daily_mean_rankic_avg"], r["daily_mean_rankic_std"]),
        axis=1,
    )

    summary = summary.sort_values(
        ["daily_mean_rankic_avg", "positive_day_ratio", "daily_top20_excess_avg"],
        ascending=[False, False, False],
    )

    return daily, bucket, summary


def write_report(output_dir: Path, selected_factors: list[str], daily: pd.DataFrame, bucket: pd.DataFrame, summary: pd.DataFrame):
    report_path = output_dir / "cs_factor_stability_report.md"

    with open(report_path, "w") as f:
        f.write("# Cross-sectional Factor Stability Report\n\n")

        f.write("## Overview\n\n")
        f.write(f"- Selected factors: {len(selected_factors)}\n")
        f.write(f"- Trading days: {daily['date'].nunique()}\n")
        f.write(f"- Factor-day records: {len(daily)}\n\n")

        f.write("## Stability Summary\n\n")
        show_cols = [
            "factor",
            "day_count",
            "daily_mean_rankic_avg",
            "daily_mean_rankic_std",
            "daily_rankic_ir",
            "positive_day_ratio",
            "daily_top20_excess_avg",
            "daily_bottom20_excess_avg",
            "worst_daily_rankic",
            "best_daily_rankic",
        ]
        f.write(summary[show_cols].to_markdown(index=False))
        f.write("\n\n")

        f.write("## Intraday Bucket Summary\n\n")
        show_bucket_cols = [
            "factor",
            "intraday_bucket",
            "bucket_mean_rankic",
            "bucket_positive_ratio",
            "bucket_top20_excess",
            "bucket_bottom20_excess",
            "datetime_count",
        ]
        f.write(bucket[show_bucket_cols].to_markdown(index=False))
        f.write("\n\n")

        f.write("## Interpretation\n\n")
        f.write("- A stable factor should have positive `daily_mean_rankic_avg` and high `positive_day_ratio`.\n")
        f.write("- `worst_daily_rankic` helps identify whether the factor breaks on specific days.\n")
        f.write("- Intraday bucket results show whether the factor is mainly open-driven, close-driven, or stable throughout the day.\n")
        f.write("- For long-only A-share usage, `daily_top20_excess_avg` is more directly relevant than long-short return.\n")

    print(f"saved report: {report_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="CrossSectionalModel/config/cs_stability_h60.yaml",
    )
    args = parser.parse_args()

    cfg = load_config(resolve_path(args.config))

    input_path = resolve_path(cfg["data"]["cs_dataset_path"])
    output_dir = resolve_path(cfg["data"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    datetime_col = cfg["columns"]["datetime_col"]

    selected_factors = select_factors(cfg)
    batch_size = int(cfg["evaluation"].get("batch_size", 5))

    print(f"input: {input_path}")
    print(f"output dir: {output_dir}")
    print(f"selected factors: {len(selected_factors)}")
    for f in selected_factors:
        print(f"  {f}")

    all_long = []

    for i in range(0, len(selected_factors), batch_size):
        batch = selected_factors[i:i + batch_size]
        print(f"\nEvaluating stability batch {i // batch_size + 1}: {batch}")

        by_time = evaluate_batch(input_path, batch, cfg)
        long_batch = wide_to_long(by_time, batch, datetime_col)
        all_long.append(long_batch)

    long_df = pd.concat(all_long, axis=0, ignore_index=True)

    daily, bucket, summary = summarize_stability(long_df, cfg)

    long_df.to_csv(output_dir / "datetime_factor_rankic_sample.csv", index=False)
    daily.to_csv(output_dir / "daily_factor_stability.csv", index=False)
    bucket.to_csv(output_dir / "intraday_bucket_stability.csv", index=False)
    summary.to_csv(output_dir / "factor_stability_summary.csv", index=False)

    write_report(output_dir, selected_factors, daily, bucket, summary)

    print("\nStability summary:")
    print(summary[[
        "factor",
        "day_count",
        "daily_mean_rankic_avg",
        "daily_rankic_ir",
        "positive_day_ratio",
        "daily_top20_excess_avg",
        "daily_bottom20_excess_avg",
        "worst_daily_rankic",
        "best_daily_rankic",
    ]].to_string(index=False))

    print(f"\nsaved outputs under: {output_dir}")


if __name__ == "__main__":
    main()
