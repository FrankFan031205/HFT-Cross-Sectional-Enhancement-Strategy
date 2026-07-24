import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def ensure_dir_for_file(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def parse_dt(s):
    s = s.astype(str)
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")

    m1 = s.str.match(r"^\d{8}_\d{9}$", na=False)
    if m1.any():
        out.loc[m1] = pd.to_datetime(
            s.loc[m1],
            format="%Y%m%d_%H%M%S%f",
            errors="coerce",
        )

    m2 = (~m1) & s.str.match(r"^\d{4}-\d{2}-\d{2}", na=False)
    if m2.any():
        out.loc[m2] = pd.to_datetime(
            s.loc[m2],
            errors="coerce",
        )

    return out


def fit_ols(train_path, factor_col, label_col):
    print("loading train calibration data:", train_path, flush=True)

    usecols = ["date", "datetime", "securityid", factor_col, label_col]
    df = pd.read_csv(train_path, usecols=usecols, low_memory=False)

    x = pd.to_numeric(df[factor_col], errors="coerce")
    y = pd.to_numeric(df[label_col], errors="coerce")

    m = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
    x = x[m].astype(float).values
    y = y[m].astype(float).values

    x_mean = x.mean()
    y_mean = y.mean()
    var_x = ((x - x_mean) ** 2).mean()

    if var_x <= 1e-18:
        raise RuntimeError("factor variance too small, cannot fit OLS")

    beta = ((x - x_mean) * (y - y_mean)).mean() / var_x
    alpha = y_mean - beta * x_mean

    pred = alpha + beta * x
    corr = np.corrcoef(pred, y)[0, 1]

    print("OLS calibration:", flush=True)
    print("  n:", len(x), flush=True)
    print("  alpha:", alpha, flush=True)
    print("  beta:", beta, flush=True)
    print("  corr(pred,label):", corr, flush=True)
    print("  x mean/std:", x.mean(), x.std(), flush=True)
    print("  y mean/std:", y.mean(), y.std(), flush=True)

    return alpha, beta


def load_oos_hidden(path, factor_col, label_col):
    print("loading OOS hidden factor:", path, flush=True)
    usecols = ["date", "datetime", "securityid", factor_col, label_col]
    df = pd.read_csv(path, usecols=usecols, low_memory=False)

    df["date"] = df["date"].astype(int)
    df["securityid"] = pd.to_numeric(df["securityid"], errors="coerce").astype("Int64")
    df["datetime_raw"] = df["datetime"].astype(str)
    df["dt_key"] = parse_dt(df["datetime"])

    df[factor_col] = pd.to_numeric(df[factor_col], errors="coerce")
    df[label_col] = pd.to_numeric(df[label_col], errors="coerce")

    print("OOS hidden shape:", df.shape, flush=True)
    print("OOS date:", df["date"].min(), df["date"].max(), flush=True)
    print("OOS symbols:", df["securityid"].nunique(), flush=True)
    print("OOS dt missing:", df["dt_key"].isna().mean(), flush=True)

    return df


def market_usecols(path, label_col):
    header = pd.read_csv(path, nrows=0)
    lower_to_orig = {str(c).lower(): c for c in header.columns}

    need = [
        "date",
        "datetime",
        "securityid",
        "bid1",
        "ask1",
        "mid_price",
        "spread",
        "bid1_volume",
        "ask1_volume",
        label_col.lower(),
    ]

    cols = []
    missing = []

    for c in need:
        if c in lower_to_orig:
            cols.append(lower_to_orig[c])
        else:
            missing.append(c)

    if missing:
        raise RuntimeError(f"{path} missing columns: {missing}")

    return cols


def load_market_for_hidden(oos, market_dir, pattern, label_col, chunksize):
    parts = []

    for date, hday in oos.groupby("date", sort=True):
        file_path = os.path.join(market_dir, pattern.format(date=int(date)))

        if not os.path.exists(file_path):
            print("WARNING market file missing:", file_path, flush=True)
            continue

        needed_dt = set(hday["dt_key"].dropna().unique())
        needed_sid = set(hday["securityid"].dropna().astype(int).unique())

        print("loading market:", file_path, flush=True)
        print("  needed dt:", len(needed_dt), "needed sid:", len(needed_sid), flush=True)

        usecols = market_usecols(file_path, label_col)
        day_parts = []

        for i, chunk in enumerate(pd.read_csv(file_path, usecols=usecols, chunksize=chunksize, low_memory=False)):
            chunk.columns = [str(c).lower() for c in chunk.columns]

            chunk["securityid"] = pd.to_numeric(chunk["securityid"], errors="coerce").astype("Int64")
            chunk = chunk[chunk["securityid"].isin(needed_sid)].copy()

            if chunk.empty:
                continue

            chunk["dt_key"] = parse_dt(chunk["datetime"])
            chunk = chunk[chunk["dt_key"].isin(needed_dt)].copy()

            if chunk.empty:
                continue

            day_parts.append(chunk)

            if i % 20 == 0:
                print("  chunk", i, "matched rows:", len(chunk), flush=True)

        if not day_parts:
            print("  WARNING no market rows matched for date:", date, flush=True)
            continue

        mday = pd.concat(day_parts, ignore_index=True)

        mday = (
            mday.sort_values(["securityid", "dt_key"])
                .drop_duplicates(["securityid", "dt_key"], keep="first")
        )

        print("  market matched rows:", len(mday), flush=True)

        parts.append(mday)

    if not parts:
        raise RuntimeError("No market rows matched OOS hidden factor.")

    mkt = pd.concat(parts, ignore_index=True)
    return mkt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--chunksize", type=int, default=2_000_000)
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    factor_col = "hidden_factor_cs_dl_feature_transformer_h120"
    label_col = cfg["pricing"]["label_col"]

    train_path = cfg["model"]["train_hidden_factor_path"]
    oos_hidden_path = cfg["model"]["oos_hidden_factor_path"]
    out_path = cfg["pricing"]["oos_priced_dataset_path"]

    market_dir = cfg["data"]["market_data_dir"]
    pattern = cfg["data"]["market_file_pattern"]

    ensure_dir_for_file(out_path)

    print("=" * 100, flush=True)
    print("Frozen pricing calibration for feature_transformer_h120 OOS", flush=True)
    print("train hidden:", train_path, flush=True)
    print("oos hidden:", oos_hidden_path, flush=True)
    print("market dir:", market_dir, flush=True)
    print("out:", out_path, flush=True)
    print("=" * 100, flush=True)

    alpha, beta = fit_ols(train_path, factor_col, label_col)

    oos = load_oos_hidden(oos_hidden_path, factor_col, label_col)

    oos["pred_ret"] = alpha + beta * oos[factor_col]

    mkt = load_market_for_hidden(
        oos=oos,
        market_dir=market_dir,
        pattern=pattern,
        label_col=label_col,
        chunksize=args.chunksize,
    )

    mkt["securityid"] = pd.to_numeric(mkt["securityid"], errors="coerce").astype("Int64")

    keep_market = [
        "date",
        "datetime",
        "securityid",
        "dt_key",
        "bid1",
        "ask1",
        "mid_price",
        "spread",
        "bid1_volume",
        "ask1_volume",
        label_col,
    ]

    mkt = mkt[keep_market].copy()

    print("merging hidden + market...", flush=True)

    merged = oos.merge(
        mkt,
        on=["date", "securityid", "dt_key"],
        how="left",
        suffixes=("", "_mkt"),
    )

    missing_mid = merged["mid_price"].isna().mean()
    print("merged shape:", merged.shape, flush=True)
    print("missing mid:", missing_mid, flush=True)

    if missing_mid > 0.05:
        raise RuntimeError(f"too many missing market rows: {missing_mid}")

    # prefer market datetime format for downstream scripts
    merged["datetime"] = merged["datetime_mkt"].fillna(merged["datetime_raw"])

    for c in ["bid1", "ask1", "mid_price", "spread", "bid1_volume", "ask1_volume", label_col]:
        merged[c] = pd.to_numeric(merged[c], errors="coerce")

    merged["fair_price"] = merged["mid_price"] * (1.0 + merged["pred_ret"])
    merged["spread_bps"] = merged["spread"] / merged["mid_price"].replace(0, np.nan) * 10000.0

    merged["buy_edge_bps"] = (merged["fair_price"] - merged["ask1"]) / merged["mid_price"].replace(0, np.nan) * 10000.0
    merged["sell_edge_bps"] = (merged["bid1"] - merged["fair_price"]) / merged["mid_price"].replace(0, np.nan) * 10000.0

    merged["valid_market"] = (
        merged["mid_price"].notna()
        & merged["bid1"].notna()
        & merged["ask1"].notna()
        & (merged["mid_price"] > 0)
        & (merged["bid1"] > 0)
        & (merged["ask1"] > 0)
        & (merged["ask1"] >= merged["bid1"])
    )

    merged["score_for_rank"] = merged["pred_ret"]

    out_cols = [
        "date",
        "datetime",
        "securityid",
        factor_col,
        "pred_ret",
        "fair_price",
        "buy_edge_bps",
        "sell_edge_bps",
        "score_for_rank",
        "mid_price",
        "bid1",
        "ask1",
        "spread",
        "spread_bps",
        "bid1_volume",
        "ask1_volume",
        label_col,
        "valid_market",
    ]

    out = merged[out_cols].copy()
    out.to_csv(out_path, index=False)

    cal_path = str(Path(out_path).with_suffix(".calibration.yaml"))
    with open(cal_path, "w") as f:
        yaml.safe_dump(
            {
                "calibration_mode": "frozen_train_ols",
                "train_path": train_path,
                "factor_col": factor_col,
                "label_col": label_col,
                "alpha": float(alpha),
                "beta": float(beta),
            },
            f,
            sort_keys=False,
        )

    print("saved priced dataset:", out_path, flush=True)
    print("saved calibration:", cal_path, flush=True)
    print("shape:", out.shape, flush=True)
    print("date:", out["date"].min(), out["date"].max(), flush=True)
    print("symbols:", out["securityid"].nunique(), flush=True)
    print("valid market rate:", out["valid_market"].mean(), flush=True)
    print("pred_ret describe:", flush=True)
    print(out["pred_ret"].describe(), flush=True)
    print("buy_edge_bps describe:", flush=True)
    print(out["buy_edge_bps"].describe(), flush=True)


if __name__ == "__main__":
    main()
