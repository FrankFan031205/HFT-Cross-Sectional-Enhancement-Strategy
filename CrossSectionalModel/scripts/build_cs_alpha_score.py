import argparse
from pathlib import Path
import polars as pl
import yaml


def repo_root():
    return Path(__file__).resolve().parents[2]


def resolve_path(p):
    p = Path(p)
    return p if p.is_absolute() else repo_root() / p


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def normalize_keys(lf, datetime_col="datetime", symbol_col="securityid"):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hidden-path",
        type=str,
        default="CrossSectionalModel/outputs/hidden_factors/hidden_cs_models_h60_202410_100.csv",
    )
    parser.add_argument(
        "--manual-score-path",
        type=str,
        default="CrossSectionalModel/outputs/cs_score_h60_202410_100.csv",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="CrossSectionalModel/outputs/alpha_scores/cs_alpha_score_h60_202410_100.csv",
    )
    parser.add_argument(
        "--full-output-path",
        type=str,
        default="CrossSectionalModel/outputs/alpha_scores/cs_alpha_score_full_h60_202410_100.csv",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="lgbm",
        choices=["lgbm", "ridge", "ensemble"],
    )
    args = parser.parse_args()

    hidden_path = resolve_path(args.hidden_path)
    manual_path = resolve_path(args.manual_score_path)
    output_path = resolve_path(args.output_path)
    full_output_path = resolve_path(args.full_output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    full_output_path.parent.mkdir(parents=True, exist_ok=True)

    print("hidden input:", hidden_path)
    print("manual input:", manual_path)
    print("output:", output_path)
    print("full output:", full_output_path)
    print("method:", args.method)

    h = (
        pl.scan_csv(
            hidden_path,
            infer_schema_length=10000,
            ignore_errors=True,
            null_values=["", "nan", "NaN", "NULL", "None"],
        )
        .select([
            "datetime",
            "securityid",
            "hidden_cs_ridge_h60",
            "hidden_cs_lgbm_h60",
        ])
    )

    s = (
        pl.scan_csv(
            manual_path,
            infer_schema_length=10000,
            ignore_errors=True,
            null_values=["", "nan", "NaN", "NULL", "None"],
        )
        .select([
            "datetime",
            "securityid",
            "cs_score_z",
            "cs_rank",
        ])
    )

    h = normalize_keys(h)
    s = normalize_keys(s)

    df = h.join(s, on=["datetime", "securityid"], how="inner")

    df = df.with_columns([
        pl.col("hidden_cs_ridge_h60").cast(pl.Float64, strict=False),
        pl.col("hidden_cs_lgbm_h60").cast(pl.Float64, strict=False),
        pl.col("cs_score_z").cast(pl.Float64, strict=False),
        pl.col("cs_rank").cast(pl.Float64, strict=False),
    ])

    df = df.with_columns([
        (
            pl.col("hidden_cs_lgbm_h60").rank(method="average").over("datetime")
            / pl.col("hidden_cs_lgbm_h60").count().over("datetime")
        ).alias("hidden_cs_lgbm_rank"),

        (
            pl.col("hidden_cs_ridge_h60").rank(method="average").over("datetime")
            / pl.col("hidden_cs_ridge_h60").count().over("datetime")
        ).alias("hidden_cs_ridge_rank"),
    ])

    if args.method == "lgbm":
        alpha_raw = pl.col("hidden_cs_lgbm_h60")
        alpha_source = "hidden_cs_lgbm_h60"

    elif args.method == "ridge":
        alpha_raw = pl.col("hidden_cs_ridge_h60")
        alpha_source = "hidden_cs_ridge_h60"

    else:
        alpha_raw = (
            0.5 * pl.col("hidden_cs_lgbm_rank")
            + 0.3 * pl.col("hidden_cs_ridge_rank")
            + 0.2 * pl.col("cs_rank")
        )
        alpha_source = "ensemble_lgbm_ridge_manual"

    df = df.with_columns([
        alpha_raw.alias("alpha_score_raw"),
        pl.lit(alpha_source).alias("alpha_source"),
    ])

    df = df.with_columns([
        (
            (pl.col("alpha_score_raw") - pl.col("alpha_score_raw").mean().over("datetime"))
            / (pl.col("alpha_score_raw").std().over("datetime") + 1e-8)
        ).alias("alpha_score"),

        (
            pl.col("alpha_score_raw").rank(method="average").over("datetime")
            / pl.col("alpha_score_raw").count().over("datetime")
        ).alias("alpha_rank"),
    ])

    slim = df.select([
        "datetime",
        "securityid",
        "alpha_score",
        "alpha_rank",
        "alpha_source",
    ])

    full = df.select([
        "datetime",
        "securityid",
        "alpha_score",
        "alpha_rank",
        "alpha_source",
        "alpha_score_raw",
        "hidden_cs_lgbm_h60",
        "hidden_cs_lgbm_rank",
        "hidden_cs_ridge_h60",
        "hidden_cs_ridge_rank",
        "cs_score_z",
        "cs_rank",
    ])

    slim_out = slim.collect()
    full_out = full.collect()

    slim_out.write_csv(output_path)
    full_out.write_csv(full_output_path)

    print("slim shape:", slim_out.shape)
    print("full shape:", full_out.shape)
    print("saved:", output_path)
    print("saved:", full_output_path)


if __name__ == "__main__":
    main()
