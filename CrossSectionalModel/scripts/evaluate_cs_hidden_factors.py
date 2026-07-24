import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import yaml


def repo_root():
    return Path(__file__).resolve().parents[2]


def resolve_path(p):
    p = Path(p)
    return p if p.is_absolute() else repo_root() / p


def safe_ir(mean_value, std_value):
    if pd.isna(std_value) or abs(std_value) < 1e-12:
        return np.nan
    return mean_value / std_value


def parse_date(x):
    return str(x)[:8]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="CrossSectionalModel/outputs/hidden_factors/hidden_cs_models_h60_202410_100.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="CrossSectionalModel/outputs/hidden_factors/eval_hidden_cs_models_h60_202410_100",
    )
    args = parser.parse_args()

    input_path = resolve_path(args.input)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    datetime_col = "datetime"
    label_col = "label_60"
    label_rank_col = "label_cs_60_rank"
    label_excess_col = "label_cs_60_demean"

    factor_cols = [
        "hidden_cs_ridge_h60",
        "hidden_cs_lgbm_h60",
    ]

    print("input:", input_path)
    print("output_dir:", output_dir)

    lf = (
        pl.scan_csv(
            input_path,
            infer_schema_length=10000,
            ignore_errors=True,
            null_values=["", "nan", "NaN", "NULL", "None"],
        )
        .select([
            datetime_col,
            label_col,
            label_rank_col,
            label_excess_col,
        ] + factor_cols)
        .with_columns([
            pl.col(label_col).cast(pl.Float64, strict=False),
            pl.col(label_rank_col).cast(pl.Float64, strict=False),
            pl.col(label_excess_col).cast(pl.Float64, strict=False),
            *[pl.col(f).cast(pl.Float64, strict=False) for f in factor_cols],
        ])
        .filter(
            pl.col(label_col).is_not_null()
            & pl.col(label_rank_col).is_not_null()
        )
    )

    rank_exprs = []
    for f in factor_cols:
        rank_exprs.append(
            (
                pl.col(f).rank(method="average").over(datetime_col)
                / pl.col(f).count().over(datetime_col)
            ).alias(f"{f}__rank")
        )

    lf = lf.with_columns(rank_exprs)

    agg_exprs = []

    for f in factor_cols:
        r = f"{f}__rank"

        agg_exprs.extend([
            pl.len().alias("nobs") if f == factor_cols[0] else pl.lit(None).alias(f"{f}__dummy"),
            pl.corr(r, label_rank_col).alias(f"{f}__rankic"),

            pl.when(pl.col(r) >= 0.8)
              .then(pl.col(label_excess_col))
              .otherwise(None)
              .mean()
              .alias(f"{f}__top20_excess_return"),

            pl.when(pl.col(r) <= 0.2)
              .then(pl.col(label_excess_col))
              .otherwise(None)
              .mean()
              .alias(f"{f}__bottom20_excess_return"),

            pl.when(pl.col(r) >= 0.8)
              .then(pl.col(label_col))
              .otherwise(None)
              .mean()
              .alias(f"{f}__top20_return"),

            pl.when(pl.col(r) <= 0.2)
              .then(pl.col(label_col))
              .otherwise(None)
              .mean()
              .alias(f"{f}__bottom20_return"),
        ])

    by_time = (
        lf.group_by(datetime_col)
        .agg(agg_exprs)
        .collect()
        .to_pandas()
    )

    by_time["date"] = by_time[datetime_col].astype(str).str.slice(0, 8)

    summary_rows = []
    daily_rows = []

    for f in factor_cols:
        rankic_col = f"{f}__rankic"
        top_excess_col = f"{f}__top20_excess_return"
        bottom_excess_col = f"{f}__bottom20_excess_return"
        top_return_col = f"{f}__top20_return"
        bottom_return_col = f"{f}__bottom20_return"

        tmp = by_time[[
            datetime_col,
            "date",
            "nobs",
            rankic_col,
            top_excess_col,
            bottom_excess_col,
            top_return_col,
            bottom_return_col,
        ]].copy()

        tmp = tmp.rename(columns={
            rankic_col: "rankic",
            top_excess_col: "top20_excess_return",
            bottom_excess_col: "bottom20_excess_return",
            top_return_col: "top20_return",
            bottom_return_col: "bottom20_return",
        })

        tmp["factor"] = f

        mean_rankic = tmp["rankic"].mean()
        std_rankic = tmp["rankic"].std(ddof=1)

        summary_rows.append({
            "factor": f,
            "mean_rankic": mean_rankic,
            "std_rankic": std_rankic,
            "rankic_ir": safe_ir(mean_rankic, std_rankic),
            "positive_ratio": (tmp["rankic"] > 0).mean(),
            "datetime_count": len(tmp),
            "avg_nobs": tmp["nobs"].mean(),
            "top20_return": tmp["top20_return"].mean(),
            "top20_excess_return": tmp["top20_excess_return"].mean(),
            "bottom20_return": tmp["bottom20_return"].mean(),
            "bottom20_excess_return": tmp["bottom20_excess_return"].mean(),
            "top_minus_bottom_excess": (
                tmp["top20_excess_return"] - tmp["bottom20_excess_return"]
            ).mean(),
        })

        daily = (
            tmp.groupby(["factor", "date"])
            .agg(
                daily_mean_rankic=("rankic", "mean"),
                daily_rankic_std=("rankic", "std"),
                daily_positive_ratio=("rankic", lambda x: (x > 0).mean()),
                daily_top20_excess=("top20_excess_return", "mean"),
                daily_bottom20_excess=("bottom20_excess_return", "mean"),
                datetime_count=("rankic", "count"),
            )
            .reset_index()
        )

        daily_rows.append(daily)

    summary = pd.DataFrame(summary_rows)
    daily = pd.concat(daily_rows, axis=0, ignore_index=True)

    summary = summary.sort_values(
        ["mean_rankic", "top20_excess_return"],
        ascending=[False, False],
    )

    summary.to_csv(output_dir / "hidden_cs_factor_summary.csv", index=False)
    daily.to_csv(output_dir / "hidden_cs_factor_daily.csv", index=False)
    by_time.to_csv(output_dir / "hidden_cs_factor_by_datetime.csv", index=False)

    report_path = output_dir / "hidden_cs_factor_eval_report.md"

    with open(report_path, "w") as f:
        f.write("# Hidden Cross-sectional Factor Evaluation Report\n\n")
        f.write("## Overall Summary\n\n")
        f.write(summary.to_markdown(index=False))
        f.write("\n\n")
        f.write("## Daily Summary\n\n")
        f.write(daily.to_markdown(index=False))
        f.write("\n\n")

    print("\nOverall summary:")
    print(summary.to_string(index=False))

    print("\nDaily summary:")
    print(daily.to_string(index=False))

    print("\nsaved summary:", output_dir / "hidden_cs_factor_summary.csv")
    print("saved daily:", output_dir / "hidden_cs_factor_daily.csv")
    print("saved report:", report_path)


if __name__ == "__main__":
    main()
