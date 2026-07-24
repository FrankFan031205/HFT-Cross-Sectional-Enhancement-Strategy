import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def clean_date(s):
    return (
        s.astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.strip()
    )


def clean_securityid(s):
    return (
        s.astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.strip()
        .str.zfill(6)
    )


def make_time_key_from_datetime(x):
    s = x.astype(str).str.strip()
    out = s.copy()

    mask_underscore = s.str.contains("_", regex=False)
    out.loc[mask_underscore] = s.loc[mask_underscore].str.split("_").str[-1]

    mask_space = s.str.contains(" ", regex=False)
    out.loc[mask_space] = s.loc[mask_space].str.split(" ").str[-1]

    out = (
        out.astype(str)
        .str.replace(":", "", regex=False)
        .str.replace(".", "", regex=False)
        .str.replace(r"\D", "", regex=True)
    )

    def fix_token(v):
        v = str(v)
        if len(v) == 6:
            return v + "000"
        return v[-9:].zfill(9)

    return out.map(fix_token)


def normalize_key(df):
    df = df.copy()

    if "date" not in df.columns:
        raise KeyError("missing date column")

    if "securityid" not in df.columns:
        if "code" in df.columns:
            df = df.rename(columns={"code": "securityid"})
        elif "symbol" in df.columns:
            df = df.rename(columns={"symbol": "securityid"})
        else:
            raise KeyError("missing securityid / code / symbol column")

    if "datetime" not in df.columns:
        if "timestamp" in df.columns:
            df = df.rename(columns={"timestamp": "datetime"})
        elif "time" in df.columns:
            df["datetime"] = (
                df["date"].astype(str)
                + "_"
                + df["time"].astype(str).str.zfill(9)
            )
        else:
            raise KeyError("missing datetime / timestamp / time column")

    df["date"] = clean_date(df["date"])
    df["securityid"] = clean_securityid(df["securityid"])
    df["datetime"] = df["datetime"].astype(str)
    df["time_key"] = make_time_key_from_datetime(df["datetime"])

    return df


def infer_hidden_factor_col(path):
    sample = pd.read_csv(path, nrows=5)
    hidden_cols = [c for c in sample.columns if c.startswith("hidden_factor")]

    if len(hidden_cols) == 0:
        raise ValueError(f"no hidden_factor column found in {path}")

    if len(hidden_cols) > 1:
        print(f"warning: multiple hidden_factor columns in {path}, using {hidden_cols[0]}")

    return hidden_cols[0]


def load_one_factor(path):
    factor_col = infer_hidden_factor_col(path)

    sample_cols = pd.read_csv(path, nrows=5).columns.tolist()

    usecols = ["date", "datetime", "securityid", factor_col]

    if "split" in sample_cols:
        usecols.append("split")

    df = pd.read_csv(path, usecols=usecols)
    df = normalize_key(df)

    keep_cols = ["date", "securityid", "time_key", factor_col]

    if "split" in df.columns:
        keep_cols.append("split")

    df = df[keep_cols].drop_duplicates(
        ["date", "securityid", "time_key"],
        keep="last",
    )

    return df, factor_col


def load_factors(factor_files):
    wide = None
    signal_cols = []
    split_added = False

    for path in factor_files:
        print("loading factor:", path.name)

        df, factor_col = load_one_factor(path)

        if wide is None:
            wide = df
            split_added = "split" in df.columns
        else:
            merge_cols = ["date", "securityid", "time_key", factor_col]

            if "split" in df.columns and not split_added:
                merge_cols.append("split")
                split_added = True

            df = df[merge_cols]

            wide = wide.merge(
                df,
                on=["date", "securityid", "time_key"],
                how="outer",
            )

        signal_cols.append(factor_col)
        print("factor shape:", df.shape, "factor_col:", factor_col)

    signal_cols = sorted(set(signal_cols))

    return wide, signal_cols


def create_split_by_date(df):
    dates = sorted(df["date"].unique())
    n = len(dates)

    if n < 3:
        raise ValueError("not enough dates to create split")

    train_end = int(n * 0.7)
    valid_end = int(n * 0.85)

    train_dates = set(dates[:train_end])
    valid_dates = set(dates[train_end:valid_end])

    def assign_split(d):
        if d in train_dates:
            return "train"
        if d in valid_dates:
            return "valid"
        return "test"

    df = df.copy()
    df["split"] = df["date"].map(assign_split)

    return df


def merge_market_with_factors(market_path, factor_wide, target_col, chunksize):
    market_source = Path(market_path)

    if market_source.is_dir():
        market_files = sorted(market_source.glob("market_return_*.csv"))
        if len(market_files) == 0:
            raise FileNotFoundError(f"no market_return_*.csv found in {market_source}")
    else:
        market_files = [market_source]

    first_header = pd.read_csv(market_files[0], nrows=5).columns.tolist()

    market_cols = [
        "date",
        "datetime",
        "securityid",
        "bid1",
        "ask1",
        "mid_price",
        "spread",
        "limit_up_price",
        "limit_down_price",
        "marketValue",
        "turnoverRate",
        "volatility_60",
        target_col,
    ]

    market_cols = [c for c in market_cols if c in first_header]

    if target_col not in market_cols:
        raise KeyError(f"target_col not found in market file: {target_col}")

    parts = []

    print("market source:", market_source)
    print("num market files:", len(market_files))

    for file_id, fp in enumerate(market_files):
        print("streaming market file:", file_id, fp.name)

        for chunk_id, chunk in enumerate(
            pd.read_csv(fp, usecols=market_cols, chunksize=chunksize)
        ):
            chunk = normalize_key(chunk)

            merged = chunk.merge(
                factor_wide,
                on=["date", "securityid", "time_key"],
                how="inner",
            )

            if len(merged) > 0:
                parts.append(merged)

            print(
                "file:",
                fp.name,
                "chunk:",
                chunk_id,
                "chunk_rows:",
                len(chunk),
                "matched:",
                len(merged),
            )

    if len(parts) == 0:
        raise ValueError("merged dataset is empty")

    out = pd.concat(parts, ignore_index=True)

    if "split" not in out.columns:
        out = create_split_by_date(out)

    return out


def calc_linear_calibration(df, signal_col, target_col, split_col, calibrate_splits, method):
    train = df[df[split_col].isin(calibrate_splits)]
    train = train[[signal_col, target_col]].replace([np.inf, -np.inf], np.nan).dropna()

    if len(train) == 0:
        raise ValueError(f"no calibration data for {signal_col}")

    x = train[signal_col].astype(float)
    y = train[target_col].astype(float)

    x_mean = x.mean()
    y_mean = y.mean()

    var_x = ((x - x_mean) ** 2).mean()

    if var_x == 0 or pd.isna(var_x):
        raise ValueError(f"zero signal variance for {signal_col}")

    beta = ((x - x_mean) * (y - y_mean)).mean() / var_x
    intercept = y_mean - beta * x_mean

    signal_rank_ic_calib = x.corr(y, method="spearman")

    if method == "rank_guarded":
        if pd.notna(signal_rank_ic_calib) and signal_rank_ic_calib != 0:
            desired_sign = 1.0 if signal_rank_ic_calib > 0 else -1.0
            beta = desired_sign * abs(beta)
            intercept = y_mean - beta * x_mean

    return {
        "intercept": intercept,
        "beta": beta,
        "n_calibration": len(train),
        "calibration_signal_rank_ic": signal_rank_ic_calib,
    }


def calc_metrics(df, signal_col, pred_col, target_col, split_col):
    rows = []

    for split, g in df.groupby(split_col):
        tmp = g[[signal_col, pred_col, target_col]].replace([np.inf, -np.inf], np.nan).dropna()

        if len(tmp) == 0:
            continue

        err = tmp[pred_col] - tmp[target_col]

        rows.append(
            {
                "signal": signal_col.replace("hidden_factor_", ""),
                "split": split,
                "n": len(tmp),
                "signal_rank_ic": tmp[signal_col].corr(tmp[target_col], method="spearman"),
                "pred_ret_rank_ic": tmp[pred_col].corr(tmp[target_col], method="spearman"),
                "pred_ret_pearson_ic": tmp[pred_col].corr(tmp[target_col], method="pearson"),
                "mse": (err ** 2).mean(),
                "mae": err.abs().mean(),
                "pred_ret_mean": tmp[pred_col].mean(),
                "pred_ret_std": tmp[pred_col].std(),
                "target_mean": tmp[target_col].mean(),
                "target_std": tmp[target_col].std(),
            }
        )

    return rows


def run_pricing_for_signal(
    merged,
    signal_col,
    target_col,
    output_root,
    tick_size,
    return_type,
    calibrate_splits,
    calibration_method,
):
    signal_name = signal_col.replace("hidden_factor_", "")

    cols = [
        "date",
        "datetime",
        "securityid",
        "time_key",
        "split",
        "bid1",
        "ask1",
        "mid_price",
        "spread",
        "limit_up_price",
        "limit_down_price",
        "marketValue",
        "turnoverRate",
        "volatility_60",
        signal_col,
        target_col,
    ]

    cols = [c for c in cols if c in merged.columns]

    df = merged[cols].copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[signal_col, target_col, "mid_price"])

    if len(df) == 0:
        print("skip empty signal after dropna:", signal_col)
        return None, None

    calib = calc_linear_calibration(
        df=df,
        signal_col=signal_col,
        target_col=target_col,
        split_col="split",
        calibrate_splits=calibrate_splits,
        method=calibration_method,
    )

    pred_col = f"pred_ret_{signal_name}"
    fair_col = f"fair_price_{signal_name}"
    alpha_price_col = f"alpha_price_{signal_name}"
    alpha_ticks_col = f"alpha_ticks_{signal_name}"
    signal_out_col = f"signal_{signal_name}"

    df[signal_out_col] = df[signal_col]
    df[pred_col] = calib["intercept"] + calib["beta"] * df[signal_col]

    if return_type == "simple":
        df[fair_col] = df["mid_price"] * (1.0 + df[pred_col])
    elif return_type == "log":
        df[fair_col] = df["mid_price"] * np.exp(df[pred_col])
    else:
        raise ValueError(f"unknown return_type: {return_type}")

    df[alpha_price_col] = df[fair_col] - df["mid_price"]
    df[alpha_ticks_col] = df[alpha_price_col] / tick_size
    df["target_ret"] = df[target_col]

    pricing_dir = output_root / "pricing" / signal_name
    pricing_dir.mkdir(parents=True, exist_ok=True)

    out_cols = [
        "date",
        "datetime",
        "securityid",
        "split",
        "bid1",
        "ask1",
        "mid_price",
        "spread",
        "marketValue",
        "turnoverRate",
        "volatility_60",
        signal_out_col,
        pred_col,
        fair_col,
        alpha_price_col,
        alpha_ticks_col,
        "target_ret",
    ]

    out_cols = [c for c in out_cols if c in df.columns]

    df[out_cols].to_csv(pricing_dir / "priced_dataset.csv", index=False)

    metric_rows = calc_metrics(
        df=df,
        signal_col=signal_col,
        pred_col=pred_col,
        target_col=target_col,
        split_col="split",
    )

    calib_row = {
        "signal": signal_name,
        "raw_col": signal_col,
        "target_col": target_col,
        "calibration_method": calibration_method,
        "calibrate_splits": ",".join(calibrate_splits),
        "intercept": calib["intercept"],
        "beta": calib["beta"],
        "n_calibration": calib["n_calibration"],
        "calibration_signal_rank_ic": calib["calibration_signal_rank_ic"],
        "n_pricing_rows": len(df),
    }

    return calib_row, metric_rows


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--market", required=True)
    parser.add_argument("--factor_dir", required=True)
    parser.add_argument("--factor_glob", required=True)
    parser.add_argument("--target", default="ret_120")
    parser.add_argument("--run_name", required=True)

    parser.add_argument("--tick_size", type=float, default=0.01)
    parser.add_argument("--return_type", default="simple")
    parser.add_argument("--calibration", default="ols", choices=["ols", "rank_guarded"])
    parser.add_argument("--calibrate_splits", default="train,valid")
    parser.add_argument("--chunksize", type=int, default=2000000)
    parser.add_argument("--save_merged", action="store_true")

    args = parser.parse_args()

    market_path = Path(args.market)
    factor_dir = Path(args.factor_dir)
    output_root = Path("output") / args.run_name
    report_dir = output_root / "reports"

    output_root.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    factor_files = sorted(factor_dir.glob(args.factor_glob))

    if len(factor_files) == 0:
        raise FileNotFoundError(f"no factor files matched: {factor_dir}/{args.factor_glob}")

    print("num factor files:", len(factor_files))

    factor_wide, signal_cols = load_factors(factor_files)

    print("factor_wide shape:", factor_wide.shape)
    print("signals:")
    for c in signal_cols:
        print(" ", c)

    merged = merge_market_with_factors(
        market_path=market_path,
        factor_wide=factor_wide,
        target_col=args.target,
        chunksize=args.chunksize,
    )

    print("merged shape:", merged.shape)
    print("date range:", merged["date"].min(), merged["date"].max())
    print("num dates:", merged["date"].nunique())
    print("num stocks:", merged["securityid"].nunique())

    if args.save_merged:
        merged_path = output_root / "merged_pricing_dataset.csv"
        merged.to_csv(merged_path, index=False)
        print("saved merged:", merged_path)

    calibrate_splits = [x.strip() for x in args.calibrate_splits.split(",") if x.strip()]

    calibration_rows = []
    metric_rows = []

    for signal_col in signal_cols:
        print("pricing signal:", signal_col)

        try:
            calib_row, rows = run_pricing_for_signal(
                merged=merged,
                signal_col=signal_col,
                target_col=args.target,
                output_root=output_root,
                tick_size=args.tick_size,
                return_type=args.return_type,
                calibrate_splits=calibrate_splits,
                calibration_method=args.calibration,
            )
        except Exception as e:
            print("failed signal:", signal_col, "error:", e)
            continue

        if calib_row is not None:
            calibration_rows.append(calib_row)

        if rows is not None:
            metric_rows.extend(rows)

    calibration = pd.DataFrame(calibration_rows)
    metrics = pd.DataFrame(metric_rows)

    calibration.to_csv(report_dir / "pricing_calibration.csv", index=False)
    metrics.to_csv(report_dir / "pricing_metrics_by_signal.csv", index=False)

    print("saved:", report_dir / "pricing_calibration.csv")
    print("saved:", report_dir / "pricing_metrics_by_signal.csv")

    if len(metrics) > 0:
        print("metrics:")
        print(metrics.to_string(index=False))

    if len(calibration) > 0:
        print("calibration:")
        print(calibration.to_string(index=False))


if __name__ == "__main__":
    main()
