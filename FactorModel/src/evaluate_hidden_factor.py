import os
import argparse
import yaml
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def ensure_dir(path):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def calc_corr(x, y):
    mask = np.isfinite(x) & np.isfinite(y)

    if mask.sum() < 3:
        return np.nan, np.nan

    x = x[mask]
    y = y[mask]

    if np.unique(x).size < 2:
        return np.nan, np.nan

    if np.unique(y).size < 2:
        return np.nan, np.nan

    try:
        ic = pearsonr(x, y)[0]
    except Exception:
        ic = np.nan

    try:
        rankic = spearmanr(x, y)[0]
    except Exception:
        rankic = np.nan

    return ic, rankic


def eval_by_datetime(df, factor_col, label_col, group_col, n_bins):
    rows = []

    for dt, g in df.groupby(group_col, sort=False):
        if len(g) < n_bins:
            continue

        x = g[factor_col].values.astype(float)
        y = g[label_col].values.astype(float)

        ic, rankic = calc_corr(x, y)

        row = {
            "datetime": dt,
            "n": len(g),
            "ic": ic,
            "rankic": rankic,
        }

        try:
            tmp = g[[factor_col, label_col]].copy()
            tmp["quantile"] = pd.qcut(
                tmp[factor_col],
                q=n_bins,
                labels=False,
                duplicates="drop",
            )

            qret = tmp.groupby("quantile")[label_col].mean()

            for i in range(n_bins):
                row[f"group_{i + 1}"] = qret.get(i, np.nan)

            row["long_short"] = qret.get(n_bins - 1, np.nan) - qret.get(0, np.nan)

        except Exception:
            for i in range(n_bins):
                row[f"group_{i + 1}"] = np.nan
            row["long_short"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def summarize(df, ts, factor_col, label_col, n_bins):
    overall_ic, overall_rankic = calc_corr(
        df[factor_col].values.astype(float),
        df[label_col].values.astype(float),
    )

    res = {
        "factor": factor_col,
        "label": label_col,
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

    return pd.DataFrame([res])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    data_cfg = cfg["data"]
    eval_cfg = cfg["evaluation"]
    out_cfg = cfg["output"]

    path = data_cfg["factor_data_path"]

    date_col = data_cfg.get("date_col", "date")
    datetime_col = data_cfg["datetime_col"]
    symbol_col = data_cfg["symbol_col"]
    factor_col = data_cfg["factor_col"]
    label_col = data_cfg["label_col"]

    start_date = int(data_cfg["start_date"])
    end_date = int(data_cfg["end_date"])

    group_col = eval_cfg["cross_sectional_ic"].get("group_col", datetime_col)
    n_bins = int(eval_cfg["quantile_return"].get("n_bins", 5))

    print("loading:", path, flush=True)
    df = pd.read_csv(path)

    needed = [date_col, datetime_col, symbol_col, factor_col, label_col]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise RuntimeError(f"missing columns: {missing}")

    df = df[
        (df[date_col] >= start_date)
        & (df[date_col] <= end_date)
    ].copy()

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[factor_col, label_col])

    print("eval shape:", df.shape, flush=True)
    print("date range:", df[date_col].min(), df[date_col].max(), flush=True)
    print("num stocks:", df[symbol_col].nunique(), flush=True)
    print("factor:", factor_col, flush=True)
    print("label:", label_col, flush=True)

    ts = eval_by_datetime(
        df=df,
        factor_col=factor_col,
        label_col=label_col,
        group_col=group_col,
        n_bins=n_bins,
    )

    result = summarize(
        df=df,
        ts=ts,
        factor_col=factor_col,
        label_col=label_col,
        n_bins=n_bins,
    )

    result_path = out_cfg["result_path"]
    ts_path = out_cfg.get(
        "timeseries_path",
        result_path.replace(".csv", "_timeseries.csv"),
    )

    ensure_dir(result_path)
    ensure_dir(ts_path)

    result.to_csv(result_path, index=False)
    ts.to_csv(ts_path, index=False)

    print(result.T, flush=True)
    print("saved summary:", result_path, flush=True)
    print("saved timeseries:", ts_path, flush=True)


if __name__ == "__main__":
    main()