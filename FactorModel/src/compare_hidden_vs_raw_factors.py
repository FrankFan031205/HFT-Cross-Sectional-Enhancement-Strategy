import os
import argparse
import yaml
import numpy as np
import pandas as pd


def load_features(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)["features"]


def ensure_dir(path):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def corr_pair(x, y, method):
    tmp = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()

    if len(tmp) < 3:
        return np.nan

    if tmp["x"].nunique() < 2:
        return np.nan

    if tmp["y"].nunique() < 2:
        return np.nan

    return tmp["x"].corr(tmp["y"], method=method)


def eval_one_factor(df, factor_col, label_col, group_col, n_bins=5):
    overall_ic = corr_pair(df[factor_col], df[label_col], "pearson")
    overall_rankic = corr_pair(df[factor_col], df[label_col], "spearman")

    rows = []

    for dt, g in df.groupby(group_col, sort=False):
        if len(g) < n_bins:
            continue

        ic = corr_pair(g[factor_col], g[label_col], "pearson")
        rankic = corr_pair(g[factor_col], g[label_col], "spearman")

        row = {
            "datetime": dt,
            "n": len(g),
            "ic": ic,
            "rankic": rankic,
        }

        try:
            q = pd.qcut(g[factor_col], q=n_bins, labels=False, duplicates="drop")
            tmp = g[[factor_col, label_col]].copy()
            tmp["quantile"] = q

            qret = tmp.groupby("quantile")[label_col].mean()

            for i in range(n_bins):
                row[f"group_{i + 1}"] = qret.get(i, np.nan)

            row["long_short"] = qret.get(n_bins - 1, np.nan) - qret.get(0, np.nan)

        except Exception:
            for i in range(n_bins):
                row[f"group_{i + 1}"] = np.nan
            row["long_short"] = np.nan

        rows.append(row)

    ts = pd.DataFrame(rows)

    res = {
        "factor": factor_col,
        "rows": len(df),
        "num_datetimes": ts["datetime"].nunique() if len(ts) else 0,
        "overall_ic": overall_ic,
        "overall_rankic": overall_rankic,
        "mean_cs_ic": ts["ic"].mean() if len(ts) else np.nan,
        "mean_cs_rankic": ts["rankic"].mean() if len(ts) else np.nan,
        "cs_icir": ts["ic"].mean() / (ts["ic"].std() + 1e-12) if len(ts) else np.nan,
        "cs_rankicir": ts["rankic"].mean() / (ts["rankic"].std() + 1e-12) if len(ts) else np.nan,
        "mean_long_short": ts["long_short"].mean() if len(ts) else np.nan,
        "long_short_ir": ts["long_short"].mean() / (ts["long_short"].std() + 1e-12) if len(ts) else np.nan,
    }

    for i in range(n_bins):
        col = f"group_{i + 1}"
        res[f"mean_{col}"] = ts[col].mean() if len(ts) else np.nan

    return res


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--processed", default="data/processed/factor_features_202410_100_processed.csv")
    parser.add_argument("--feature_yaml", default="data/raw/feature_cols_202410_100.yaml")

    parser.add_argument("--lgbm_hidden", default="outputs/hidden_factor_lgbm_h60_202410_100.csv")
    parser.add_argument("--ridge_hidden", default="outputs/hidden_factor_ridge_h60_202410_100.csv")

    parser.add_argument("--date_col", default="date")
    parser.add_argument("--datetime_col", default="datetime")
    parser.add_argument("--symbol_col", default="securityid")
    parser.add_argument("--label_col", default="label_60")

    parser.add_argument("--start_date", type=int, default=20241030)
    parser.add_argument("--end_date", type=int, default=20241031)
    parser.add_argument("--n_bins", type=int, default=5)

    parser.add_argument("--output", default="outputs/compare_hidden_vs_raw_h60_202410_100.csv")

    args = parser.parse_args()

    feature_cols = load_features(args.feature_yaml)

    usecols = [args.date_col, args.datetime_col, args.symbol_col, args.label_col] + feature_cols

    print("loading processed data:", args.processed)
    df = pd.read_csv(args.processed, usecols=usecols)

    df = df[
        (df[args.date_col] >= args.start_date)
        & (df[args.date_col] <= args.end_date)
    ].copy()

    print("test shape:", df.shape)
    print("date range:", df[args.date_col].min(), df[args.date_col].max())
    print("num raw factors:", len(feature_cols))

    results = []

    print("evaluating raw factors...")
    for i, f in enumerate(feature_cols):
        print(i + 1, "/", len(feature_cols), f)
        res = eval_one_factor(
            df=df,
            factor_col=f,
            label_col=args.label_col,
            group_col=args.datetime_col,
            n_bins=args.n_bins,
        )
        res["type"] = "raw_factor"
        results.append(res)

    hidden_specs = [
        ("hidden_lgbm", args.lgbm_hidden, "hidden_factor_lgbm_h60"),
        ("hidden_ridge", args.ridge_hidden, "hidden_factor_ridge_h60"),
    ]

    for factor_type, path, factor_col in hidden_specs:
        if not os.path.exists(path):
            print("skip missing hidden file:", path)
            continue

        print("evaluating hidden factor:", factor_col)

        hdf = pd.read_csv(path)
        hdf = hdf[
            (hdf[args.date_col] >= args.start_date)
            & (hdf[args.date_col] <= args.end_date)
        ].copy()

        res = eval_one_factor(
            df=hdf,
            factor_col=factor_col,
            label_col=args.label_col,
            group_col=args.datetime_col,
            n_bins=args.n_bins,
        )
        res["type"] = factor_type
        results.append(res)

    out = pd.DataFrame(results)

    cols = ["type"] + [c for c in out.columns if c != "type"]
    out = out[cols]

    out = out.sort_values(
        ["mean_cs_rankic", "mean_long_short"],
        ascending=False,
    )

    ensure_dir(args.output)
    out.to_csv(args.output, index=False)

    print("saved comparison:", args.output)

    print("\nTop 20 by mean_cs_rankic:")
    show_cols = [
        "type",
        "factor",
        "mean_cs_ic",
        "mean_cs_rankic",
        "cs_rankicir",
        "mean_long_short",
        "long_short_ir",
        "mean_group_1",
        "mean_group_2",
        "mean_group_3",
        "mean_group_4",
        "mean_group_5",
    ]

    print(out[show_cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
