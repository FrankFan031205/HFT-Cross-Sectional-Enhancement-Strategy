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


def parse_datetime_parts(s):
    s = str(s)

    if "_" in s:
        date_part, time_part = s.split("_", 1)
        date = date_part[:8]
        time_part = time_part.replace(":", "")
        hh = time_part[0:2]
        mm = time_part[2:4]
        ss = time_part[4:6] if len(time_part) >= 6 else "00"
        return date, f"{hh}:{mm}:{ss}"

    if " " in s:
        date_part, time_part = s.split(" ", 1)
        date = date_part.replace("-", "")[:8]
        time = time_part[:8]
        return date, time

    date = s[:8]
    time_part = s[8:]
    hh = time_part[0:2] if len(time_part) >= 2 else "00"
    mm = time_part[2:4] if len(time_part) >= 4 else "00"
    ss = time_part[4:6] if len(time_part) >= 6 else "00"
    return date, f"{hh}:{mm}:{ss}"


def assign_bucket(time_str, buckets):
    t = str(time_str)[:8]

    for b in buckets:
        if b["start"] <= t < b["end"]:
            return b["name"]

    return "other"


def safe_ir(mean_value, std_value):
    if pd.isna(std_value) or abs(std_value) < 1e-12:
        return np.nan
    return mean_value / std_value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="CrossSectionalModel/config/cs_score_eval_h60.yaml",
    )
    args = parser.parse_args()

    cfg = load_config(resolve_path(args.config))

    score_path = resolve_path(cfg["data"]["score_path"])
    output_dir = resolve_path(cfg["data"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    col_cfg = cfg["columns"]
    eval_cfg = cfg["evaluation"]

    datetime_col = col_cfg["datetime_col"]
    label_col = col_cfg["label_col"]
    label_rank_col = col_cfg["label_rank_col"]
    label_excess_col = col_cfg["label_excess_col"]
    score_col = col_cfg["score_col"]
    rank_col = col_cfg["rank_col"]
    group_col = col_cfg["group_col"]

    top_q = float(eval_cfg.get("top_quantile", 0.2))
    bottom_q = float(eval_cfg.get("bottom_quantile", 0.2))
    min_obs = int(eval_cfg.get("min_obs_per_datetime", 30))

    top_cut = 1.0 - top_q
    bottom_cut = bottom_q

    print(f"input: {score_path}")
    print(f"output dir: {output_dir}")

    lf = (
        pl.scan_csv(
            score_path,
            infer_schema_length=10000,
            ignore_errors=True,
            null_values=["", "nan", "NaN", "NULL", "None"],
        )
        .select([
            datetime_col,
            label_col,
            label_rank_col,
            label_excess_col,
            score_col,
            rank_col,
            group_col,
        ])
        .with_columns([
            pl.col(label_col).cast(pl.Float64, strict=False),
            pl.col(label_rank_col).cast(pl.Float64, strict=False),
            pl.col(label_excess_col).cast(pl.Float64, strict=False),
            pl.col(score_col).cast(pl.Float64, strict=False),
            pl.col(rank_col).cast(pl.Float64, strict=False),
        ])
        .filter(
            pl.col(label_col).is_not_null()
            & pl.col(label_rank_col).is_not_null()
            & pl.col(score_col).is_not_null()
            & pl.col(rank_col).is_not_null()
        )
    )

    by_time = (
        lf.group_by(datetime_col)
        .agg([
            pl.len().alias("nobs"),
            pl.corr(rank_col, label_rank_col).alias("rankic"),

            pl.when(pl.col(rank_col) >= top_cut)
              .then(pl.col(label_col))
              .otherwise(None)
              .mean()
              .alias("top20_return"),

            pl.when(pl.col(rank_col) >= top_cut)
              .then(pl.col(label_excess_col))
              .otherwise(None)
              .mean()
              .alias("top20_excess_return"),

            pl.when(pl.col(rank_col) <= bottom_cut)
              .then(pl.col(label_col))
              .otherwise(None)
              .mean()
              .alias("bottom20_return"),

            pl.when(pl.col(rank_col) <= bottom_cut)
              .then(pl.col(label_excess_col))
              .otherwise(None)
              .mean()
              .alias("bottom20_excess_return"),

            pl.col(label_col).mean().alias("universe_return"),
            pl.col(label_excess_col).mean().alias("universe_excess_return"),
        ])
        .filter(pl.col("nobs") >= min_obs)
        .collect()
        .to_pandas()
    )

    parts = by_time[datetime_col].apply(parse_datetime_parts)
    by_time["date"] = parts.apply(lambda x: x[0])
    by_time["time"] = parts.apply(lambda x: x[1])
    by_time["intraday_bucket"] = by_time["time"].apply(
        lambda x: assign_bucket(x, cfg.get("intraday_buckets", []))
    )

    by_time["top_minus_bottom_return"] = by_time["top20_return"] - by_time["bottom20_return"]
    by_time["top_minus_bottom_excess"] = by_time["top20_excess_return"] - by_time["bottom20_excess_return"]

    overall = pd.DataFrame([{
        "mean_rankic": by_time["rankic"].mean(),
        "std_rankic": by_time["rankic"].std(ddof=1),
        "rankic_ir": safe_ir(by_time["rankic"].mean(), by_time["rankic"].std(ddof=1)),
        "positive_ratio": (by_time["rankic"] > 0).mean(),
        "datetime_count": len(by_time),
        "avg_nobs": by_time["nobs"].mean(),
        "top20_return": by_time["top20_return"].mean(),
        "top20_excess_return": by_time["top20_excess_return"].mean(),
        "bottom20_return": by_time["bottom20_return"].mean(),
        "bottom20_excess_return": by_time["bottom20_excess_return"].mean(),
        "top_minus_bottom_return": by_time["top_minus_bottom_return"].mean(),
        "top_minus_bottom_excess": by_time["top_minus_bottom_excess"].mean(),
    }])

    daily = (
        by_time.groupby("date")
        .agg(
            daily_mean_rankic=("rankic", "mean"),
            daily_rankic_std=("rankic", "std"),
            daily_positive_ratio=("rankic", lambda x: (x > 0).mean()),
            daily_top20_return=("top20_return", "mean"),
            daily_top20_excess_return=("top20_excess_return", "mean"),
            daily_bottom20_return=("bottom20_return", "mean"),
            daily_bottom20_excess_return=("bottom20_excess_return", "mean"),
            daily_top_minus_bottom_excess=("top_minus_bottom_excess", "mean"),
            datetime_count=("rankic", "count"),
            avg_nobs=("nobs", "mean"),
        )
        .reset_index()
    )

    daily["daily_rankic_ir"] = daily.apply(
        lambda r: safe_ir(r["daily_mean_rankic"], r["daily_rankic_std"]),
        axis=1,
    )

    bucket = (
        by_time.groupby("intraday_bucket")
        .agg(
            bucket_mean_rankic=("rankic", "mean"),
            bucket_rankic_std=("rankic", "std"),
            bucket_positive_ratio=("rankic", lambda x: (x > 0).mean()),
            bucket_top20_return=("top20_return", "mean"),
            bucket_top20_excess_return=("top20_excess_return", "mean"),
            bucket_bottom20_return=("bottom20_return", "mean"),
            bucket_bottom20_excess_return=("bottom20_excess_return", "mean"),
            bucket_top_minus_bottom_excess=("top_minus_bottom_excess", "mean"),
            datetime_count=("rankic", "count"),
            avg_nobs=("nobs", "mean"),
        )
        .reset_index()
    )

    group_summary = (
        lf.group_by(group_col)
        .agg([
            pl.len().alias("count"),
            pl.col(label_col).mean().alias("avg_label"),
            pl.col(label_excess_col).mean().alias("avg_excess"),
            pl.col(label_rank_col).mean().alias("avg_label_rank"),
            pl.col(rank_col).mean().alias("avg_cs_rank"),
        ])
        .collect()
        .to_pandas()
        .sort_values("avg_cs_rank", ascending=False)
    )

    by_time.to_csv(output_dir / "cs_score_by_datetime.csv", index=False)
    overall.to_csv(output_dir / "cs_score_overall_summary.csv", index=False)
    daily.to_csv(output_dir / "cs_score_daily_summary.csv", index=False)
    bucket.to_csv(output_dir / "cs_score_intraday_bucket_summary.csv", index=False)
    group_summary.to_csv(output_dir / "cs_score_group_summary.csv", index=False)

    report_path = output_dir / "cs_score_eval_report.md"

    with open(report_path, "w") as f:
        f.write("# Cross-sectional Score Evaluation Report\n\n")

        f.write("## Overall Summary\n\n")
        f.write(overall.to_markdown(index=False))
        f.write("\n\n")

        f.write("## Group Summary\n\n")
        f.write(group_summary.to_markdown(index=False))
        f.write("\n\n")

        f.write("## Daily Summary\n\n")
        f.write(daily.to_markdown(index=False))
        f.write("\n\n")

        f.write("## Intraday Bucket Summary\n\n")
        f.write(bucket.to_markdown(index=False))
        f.write("\n\n")

        f.write("## Interpretation\n\n")
        f.write("- `mean_rankic` measures whether the combined score ranks future returns correctly.\n")
        f.write("- `top20_excess_return` is the key long-only selection metric.\n")
        f.write("- `bottom20_excess_return` measures weak-stock avoidance value.\n")
        f.write("- `daily_positive_ratio` and bucket results measure stability across days and intraday sessions.\n")

    print("\nOverall summary:")
    print(overall.to_string(index=False))

    print("\nGroup summary:")
    print(group_summary.to_string(index=False))

    print("\nDaily summary:")
    print(daily.to_string(index=False))

    print("\nIntraday bucket summary:")
    print(bucket.to_string(index=False))

    print(f"\nsaved outputs under: {output_dir}")
    print(f"saved report: {report_path}")


if __name__ == "__main__":
    main()
