import os
import glob
import numpy as np
import pandas as pd

START_DATE = 20241030
END_DATE = 20241031
LABEL_COL = "label_60"
N_BINS = 5

files = sorted(glob.glob("outputs/hidden_factor_*_h60_202410_100.csv"))


def corr_pair(x, y, method):
    tmp = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()

    if len(tmp) < 3:
        return np.nan

    if tmp["x"].nunique() < 2 or tmp["y"].nunique() < 2:
        return np.nan

    return tmp["x"].corr(tmp["y"], method=method)


def eval_one(path):
    cols = pd.read_csv(path, nrows=0).columns.tolist()
    factor_cols = [c for c in cols if c.startswith("hidden_factor_")]

    if len(factor_cols) != 1:
        raise RuntimeError(f"{path}: cannot identify factor column, got {factor_cols}")

    factor_col = factor_cols[0]

    usecols = ["date", "datetime", "securityid", LABEL_COL, factor_col]
    df = pd.read_csv(path, usecols=usecols, dtype={"securityid": str, "datetime": str})

    df = df[(df["date"] >= START_DATE) & (df["date"] <= END_DATE)].copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[factor_col, LABEL_COL])

    overall_ic = corr_pair(df[factor_col], df[LABEL_COL], "pearson")
    overall_rankic = corr_pair(df[factor_col], df[LABEL_COL], "spearman")

    rows = []

    for dt, g in df.groupby("datetime", sort=False):
        if len(g) < N_BINS:
            continue

        ic = corr_pair(g[factor_col], g[LABEL_COL], "pearson")
        rankic = corr_pair(g[factor_col], g[LABEL_COL], "spearman")

        row = {
            "datetime": dt,
            "n": len(g),
            "ic": ic,
            "rankic": rankic,
        }

        try:
            tmp = g[[factor_col, LABEL_COL]].copy()
            tmp["quantile"] = pd.qcut(
                tmp[factor_col],
                q=N_BINS,
                labels=False,
                duplicates="drop",
            )

            qret = tmp.groupby("quantile")[LABEL_COL].mean()

            for i in range(N_BINS):
                row[f"group_{i + 1}"] = qret.get(i, np.nan)

            row["long_short"] = qret.get(N_BINS - 1, np.nan) - qret.get(0, np.nan)

        except Exception:
            for i in range(N_BINS):
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
        "file": os.path.basename(path),
    }

    for i in range(N_BINS):
        res[f"mean_group_{i + 1}"] = ts[f"group_{i + 1}"].mean() if len(ts) else np.nan

    return res


def main():
    print("files:")
    for f in files:
        print(" ", f)

    results = []

    for f in files:
        print("evaluating:", f)
        results.append(eval_one(f))

    res = pd.DataFrame(results)

    cols = [
        "factor",
        "rows",
        "num_datetimes",
        "overall_ic",
        "overall_rankic",
        "mean_cs_ic",
        "mean_cs_rankic",
        "cs_icir",
        "cs_rankicir",
        "mean_long_short",
        "long_short_ir",
        "mean_group_1",
        "mean_group_2",
        "mean_group_3",
        "mean_group_4",
        "mean_group_5",
        "file",
    ]

    res = res[cols].sort_values(
        ["mean_cs_rankic", "mean_long_short"],
        ascending=False,
    )

    out = "outputs/compare_all_hidden_factors_h60_202410_100.csv"
    res.to_csv(out, index=False)

    print("\n=== Comparison sorted by mean_cs_rankic ===")
    print(res.to_string(index=False))
    print("\nsaved:", out)


if __name__ == "__main__":
    main()