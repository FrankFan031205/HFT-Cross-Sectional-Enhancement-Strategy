import os
import argparse
import yaml
import numpy as np
import pandas as pd


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_features(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)["features"]


def ensure_dir(path):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def winsorize_by_train(df, cols, train_mask, lower_q, upper_q):
    bounds = {}

    for c in cols:
        low = df.loc[train_mask, c].quantile(lower_q)
        high = df.loc[train_mask, c].quantile(upper_q)
        bounds[c] = (low, high)
        df[c] = df[c].clip(low, high)

    return df, bounds


def cross_sectional_zscore(df, cols, group_col, eps):
    def zscore(x):
        return (x - x.mean()) / (x.std(ddof=0) + eps)

    df[cols] = df.groupby(group_col, sort=False)[cols].transform(zscore)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    data_cfg = cfg["data"]
    pp_cfg = cfg["preprocess"]

    raw_path = data_cfg["raw_data_path"]
    out_path = data_cfg["processed_data_path"]

    date_col = data_cfg.get("date_col", "date")
    datetime_col = data_cfg["datetime_col"]
    symbol_col = data_cfg["symbol_col"]
    label_col = data_cfg["label_col"]

    feature_cols = load_features(data_cfg["feature_cols_path"])

    print("loading raw data:", raw_path)
    df = pd.read_csv(raw_path)

    print("raw shape:", df.shape)
    print("num features:", len(feature_cols))

    required_cols = [date_col, datetime_col, symbol_col, label_col] + feature_cols
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"missing columns: {missing}")

    df = df[required_cols].copy()

    train_mask = (
        (df[date_col] >= data_cfg["train_start"])
        & (df[date_col] <= data_cfg["train_end"])
    )

    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    df[label_col] = df[label_col].replace([np.inf, -np.inf], np.nan)

    if pp_cfg["missing"]["method"] == "fill_zero":
        df[feature_cols] = df[feature_cols].fillna(0)
    elif pp_cfg["missing"]["method"] == "fill_median":
        med = df.loc[train_mask, feature_cols].median()
        df[feature_cols] = df[feature_cols].fillna(med)
    else:
        raise RuntimeError("unknown missing method")

    df = df.dropna(subset=[label_col])

    if pp_cfg["clip"]["enable"]:
        df, _ = winsorize_by_train(
            df,
            feature_cols,
            train_mask,
            pp_cfg["clip"]["lower_quantile"],
            pp_cfg["clip"]["upper_quantile"],
        )

    if pp_cfg["label"]["winsorize"]:
        df, _ = winsorize_by_train(
            df,
            [label_col],
            train_mask,
            pp_cfg["label"]["lower_quantile"],
            pp_cfg["label"]["upper_quantile"],
        )

    if pp_cfg["normalize"]["enable"]:
        method = pp_cfg["normalize"]["method"]
        if method != "cross_sectional_zscore":
            raise RuntimeError(f"unsupported normalize method: {method}")

        group_col = pp_cfg["normalize"]["group_col"]
        eps = float(pp_cfg["normalize"]["eps"])

        print("cross-sectional zscore by:", group_col)
        df = cross_sectional_zscore(df, feature_cols, group_col, eps)
        df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
        df[feature_cols] = df[feature_cols].fillna(0)

    ensure_dir(out_path)
    df.to_csv(out_path, index=False)

    print("saved processed data:", out_path)
    print("processed shape:", df.shape)
    print("date range:", df[date_col].min(), df[date_col].max())
    print("num stocks:", df[symbol_col].nunique())


if __name__ == "__main__":
    main()