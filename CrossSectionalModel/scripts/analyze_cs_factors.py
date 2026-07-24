import argparse
from pathlib import Path

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


def safe_round(x, n=6):
    if pd.isna(x):
        return ""
    return round(float(x), n)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="CrossSectionalModel/config/cs_analysis_h60.yaml"
    )
    args = parser.parse_args()

    cfg = load_config(resolve_path(args.config))

    eval_path = resolve_path(cfg["data"]["eval_path"])
    output_dir = resolve_path(cfg["data"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    top_n = int(cfg["analysis"].get("top_n", 20))
    suspicious_keywords = cfg["analysis"].get("suspicious_keywords", [])

    if not eval_path.exists():
        raise FileNotFoundError(f"eval file not found: {eval_path}")

    df = pd.read_csv(eval_path)

    numeric_cols = [
        "mean_rankic",
        "std_rankic",
        "rankic_ir",
        "positive_ratio",
        "valid_ic_count",
        "avg_nobs_per_datetime",
        "top20_return",
        "top20_excess_return",
        "bottom20_return",
        "bottom20_excess_return",
        "top_minus_bottom_return",
        "top_minus_bottom_excess",
        "missing_ratio",
        "zero_ratio",
        "std_all",
    ]

    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["abs_mean_rankic"] = df["mean_rankic"].abs()
    df["is_positive_alpha"] = (
        (df["mean_rankic"] > 0)
        & (df["top20_excess_return"] > 0)
        & (df["bottom20_excess_return"] < 0)
    )

    df["suspicious"] = False
    df["suspicious_reason"] = ""

    for kw in suspicious_keywords:
        mask = df["factor"].str.lower().str.contains(kw.lower(), na=False)
        df.loc[mask, "suspicious"] = True
        df.loc[mask, "suspicious_reason"] = (
            df.loc[mask, "suspicious_reason"].astype(str) + f"{kw};"
        )

    category_summary = (
        df.groupby("category")
        .agg(
            factor_count=("factor", "count"),
            positive_alpha_count=("is_positive_alpha", "sum"),
            mean_rankic_avg=("mean_rankic", "mean"),
            mean_rankic_max=("mean_rankic", "max"),
            rankic_ir_avg=("rankic_ir", "mean"),
            positive_ratio_avg=("positive_ratio", "mean"),
            top20_excess_avg=("top20_excess_return", "mean"),
            top20_excess_max=("top20_excess_return", "max"),
            bottom20_excess_avg=("bottom20_excess_return", "mean"),
            missing_ratio_avg=("missing_ratio", "mean"),
            zero_ratio_avg=("zero_ratio", "mean"),
        )
        .reset_index()
        .sort_values(["mean_rankic_avg", "top20_excess_avg"], ascending=[False, False])
    )

    top_by_rankic = df.sort_values(
        ["mean_rankic", "rankic_ir", "top20_excess_return"],
        ascending=[False, False, False]
    ).head(top_n)

    top_by_top_excess = df.sort_values(
        ["top20_excess_return", "mean_rankic", "rankic_ir"],
        ascending=[False, False, False]
    ).head(top_n)

    top_by_ir = df.sort_values(
        ["rankic_ir", "mean_rankic", "top20_excess_return"],
        ascending=[False, False, False]
    ).head(top_n)

    suspicious = df[df["suspicious"]].sort_values(
        ["mean_rankic", "top20_excess_return"],
        ascending=[False, False]
    )

    all_sorted = df.sort_values(
        ["mean_rankic", "rankic_ir", "top20_excess_return"],
        ascending=[False, False, False]
    )

    category_summary.to_csv(output_dir / "category_summary.csv", index=False)
    top_by_rankic.to_csv(output_dir / "top_by_rankic.csv", index=False)
    top_by_top_excess.to_csv(output_dir / "top_by_top20_excess.csv", index=False)
    top_by_ir.to_csv(output_dir / "top_by_rankic_ir.csv", index=False)
    suspicious.to_csv(output_dir / "suspicious_factors.csv", index=False)
    all_sorted.to_csv(output_dir / "all_factors_sorted.csv", index=False)

    report_path = output_dir / "cs_factor_analysis_report.md"

    with open(report_path, "w") as f:
        f.write("# Cross-sectional Factor Analysis Report\n\n")

        f.write("## Overview\n\n")
        f.write(f"- Total factors: {len(df)}\n")
        f.write(f"- Positive alpha factors: {int(df['is_positive_alpha'].sum())}\n")
        f.write(f"- Average mean RankIC: {safe_round(df['mean_rankic'].mean())}\n")
        f.write(f"- Best mean RankIC: {safe_round(df['mean_rankic'].max())}\n")
        f.write(f"- Average top20 excess return: {safe_round(df['top20_excess_return'].mean())}\n")
        f.write(f"- Best top20 excess return: {safe_round(df['top20_excess_return'].max())}\n\n")

        f.write("## Category Summary\n\n")
        f.write(category_summary.to_markdown(index=False))
        f.write("\n\n")

        f.write("## Top Factors by Mean RankIC\n\n")
        show_cols = [
            "factor",
            "category",
            "mean_rankic",
            "rankic_ir",
            "positive_ratio",
            "top20_excess_return",
            "bottom20_excess_return",
            "missing_ratio",
            "zero_ratio",
        ]
        f.write(top_by_rankic[show_cols].to_markdown(index=False))
        f.write("\n\n")

        f.write("## Top Factors by Top20 Excess Return\n\n")
        f.write(top_by_top_excess[show_cols].to_markdown(index=False))
        f.write("\n\n")

        f.write("## Suspicious Factors Requiring Leakage Check\n\n")
        if len(suspicious) == 0:
            f.write("No suspicious factors detected by keyword rules.\n\n")
        else:
            f.write(suspicious[[
                "factor",
                "category",
                "mean_rankic",
                "top20_excess_return",
                "suspicious_reason",
            ]].to_markdown(index=False))
            f.write("\n\n")

        f.write("## Notes\n\n")
        f.write("- `mean_rankic` measures cross-sectional ranking power.\n")
        f.write("- `top20_excess_return` measures long-only top group value.\n")
        f.write("- `bottom20_excess_return` measures weak-stock avoidance value.\n")
        f.write("- `missing_ratio` and `zero_ratio` are factor quality checks.\n")
        f.write("- Factors with names containing `ret`, `check`, `future`, or `label` should be manually checked for look-ahead bias.\n")

    print(f"saved category summary: {output_dir / 'category_summary.csv'}")
    print(f"saved top by rankic: {output_dir / 'top_by_rankic.csv'}")
    print(f"saved top by top20 excess: {output_dir / 'top_by_top20_excess.csv'}")
    print(f"saved suspicious factors: {output_dir / 'suspicious_factors.csv'}")
    print(f"saved report: {report_path}")

    print("\nCategory summary:")
    print(category_summary.to_string(index=False))

    print("\nTop factors by RankIC:")
    print(top_by_rankic[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()
