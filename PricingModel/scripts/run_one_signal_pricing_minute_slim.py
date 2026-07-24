from pathlib import Path
import argparse
import re
import numpy as np
import pandas as pd


def clean_date(s):
    return s.astype(str).str.replace(r"\.0$", "", regex=True).str.strip()


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

    mask = out.str.contains("_", regex=False)
    out.loc[mask] = out.loc[mask].str.split("_").str[-1]

    mask = out.str.contains(" ", regex=False)
    out.loc[mask] = out.loc[mask].str.split(" ").str[-1]

    out = (
        out.astype(str)
        .str.replace(":", "", regex=False)
        .str.replace(".", "", regex=False)
        .str.replace(r"\D", "", regex=True)
    )

    def fix(v):
        v = str(v)
        if len(v) == 6:
            return v + "000"
        return v[-9:].zfill(9)

    return out.map(fix)


def normalize_key(df):
    df = df.copy()

    if "securityid" not in df.columns:
        if "SecurityID" in df.columns:
            df = df.rename(columns={"SecurityID": "securityid"})
        elif "code" in df.columns:
            df = df.rename(columns={"code": "securityid"})
        elif "symbol" in df.columns:
            df = df.rename(columns={"symbol": "securityid"})
        else:
            raise KeyError("missing securityid / SecurityID / code / symbol")

    if "datetime" not in df.columns:
        if "timestamp" in df.columns:
            df = df.rename(columns={"timestamp": "datetime"})
        elif "time" in df.columns:
            df["datetime"] = df["date"].astype(str) + "_" + df["time"].astype(str).str.zfill(9)
        else:
            raise KeyError("missing datetime / timestamp / time")

    df["date"] = clean_date(df["date"])
    df["securityid"] = clean_securityid(df["securityid"])
    df["time_key"] = make_time_key_from_datetime(df["datetime"])

    return df


def keep_minute_boundary(df):
    # time_key = HHMMSSmmm; 每分钟决策点要求 SSmmm = 00000
    return df[df["time_key"].astype(str).str[-5:] == "00000"].copy()


def infer_factor_col(path):
    cols = pd.read_csv(path, nrows=5).columns.tolist()
    hidden = [c for c in cols if c.startswith("hidden_factor")]
    if not hidden:
        raise ValueError(f"no hidden_factor column found in {path}")
    return hidden[0]


def load_factor_minute(path, factor_col, chunksize):
    path = Path(path)
    header = pd.read_csv(path, nrows=5).columns.tolist()

    usecols = ["date", "datetime", "securityid", factor_col]
    missing = [c for c in usecols if c not in header]
    if missing:
        raise KeyError(f"{path.name} missing columns: {missing}")

    parts = []

    for i, chunk in enumerate(pd.read_csv(path, usecols=usecols, chunksize=chunksize)):
        chunk = normalize_key(chunk)
        chunk = keep_minute_boundary(chunk)
        chunk = chunk.dropna(subset=[factor_col])
        chunk = chunk[["date", "securityid", "time_key", factor_col]]

        if len(chunk):
            parts.append(chunk)

        print(f"[factor] chunk={i}, minute_rows={len(chunk)}", flush=True)

    if not parts:
        raise ValueError("empty factor after minute filter")

    factor = pd.concat(parts, ignore_index=True)
    factor = factor.drop_duplicates(["date", "securityid", "time_key"], keep="last")

    print("[factor] final shape:", factor.shape, flush=True)
    print("[factor] date range:", factor["date"].min(), factor["date"].max(), flush=True)
    print("[factor] stocks:", factor["securityid"].nunique(), flush=True)

    return factor


def date_from_market_filename(name):
    m = re.search(r"market_return_(\d{8})_", name)
    return m.group(1) if m else None


def load_merged_market_minute(market_dir, market_glob, factor, factor_col, target, chunksize):
    market_dir = Path(market_dir)
    files = sorted(market_dir.glob(market_glob))

    if not files:
        raise FileNotFoundError(f"no market files: {market_dir}/{market_glob}")

    factor_dates = set(factor["date"].unique())

    base_cols = [
        "date", "datetime", "securityid",
        "bid1", "ask1", "mid_price", "spread",
        "limit_up_price", "limit_down_price",
        "marketValue", "turnoverRate", "volatility_60",
        target,
    ]

    parts = []

    for fp in files:
        d = date_from_market_filename(fp.name)
        if d is not None and d not in factor_dates:
            continue

        header = pd.read_csv(fp, nrows=5).columns.tolist()
        if target not in header:
            raise KeyError(f"{fp.name} missing target {target}")

        usecols = [c for c in base_cols if c in header]

        print("[market] streaming:", fp.name, flush=True)

        f_date = factor[factor["date"] == d] if d is not None else factor
        if len(f_date) == 0:
            continue

        for j, chunk in enumerate(pd.read_csv(fp, usecols=usecols, chunksize=chunksize)):
            chunk = normalize_key(chunk)
            chunk = keep_minute_boundary(chunk)

            merged = chunk.merge(
                f_date,
                on=["date", "securityid", "time_key"],
                how="inner",
            )

            merged = merged.replace([np.inf, -np.inf], np.nan)
            merged = merged.dropna(subset=[factor_col, target, "mid_price"])

            if len(merged):
                parts.append(merged)

            print(f"  chunk={j}, matched={len(merged)}", flush=True)

    if not parts:
        raise ValueError("empty merged dataset")

    df = pd.concat(parts, ignore_index=True)

    print("[merged] shape:", df.shape, flush=True)
    print("[merged] date range:", df["date"].min(), df["date"].max(), flush=True)
    print("[merged] stocks:", df["securityid"].nunique(), flush=True)

    return df


def sampled_corr(x, y, method="spearman", n=2_000_000, seed=1):
    tmp = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(tmp) == 0:
        return np.nan

    if len(tmp) > n:
        tmp = tmp.sample(n=n, random_state=seed)

    return tmp["x"].corr(tmp["y"], method=method)


def fit_calibration(df, factor_col, target, method):
    tmp = df[[factor_col, target]].replace([np.inf, -np.inf], np.nan).dropna()

    x = tmp[factor_col].astype(float).to_numpy()
    y = tmp[target].astype(float).to_numpy()

    x_mean = x.mean()
    y_mean = y.mean()

    var_x = ((x - x_mean) ** 2).mean()
    cov_xy = ((x - x_mean) * (y - y_mean)).mean()

    beta = cov_xy / var_x if var_x > 0 else 0.0
    rank_ic = sampled_corr(tmp[factor_col], tmp[target], method="spearman")

    if method == "rank_guarded":
        if np.isfinite(rank_ic) and rank_ic != 0:
            beta = np.sign(rank_ic) * abs(beta)

    intercept = y_mean - beta * x_mean

    return intercept, beta, rank_ic, len(tmp)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--market_dir", required=True)
    parser.add_argument("--market_glob", required=True)
    parser.add_argument("--factor", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--factor_col", default="")
    parser.add_argument("--signal", default="")
    parser.add_argument("--calibration", default="rank_guarded", choices=["ols", "rank_guarded"])
    parser.add_argument("--chunksize", type=int, default=2_000_000)
    parser.add_argument("--tick_size", type=float, default=0.01)

    args = parser.parse_args()

    factor_path = Path(args.factor)
    factor_col = args.factor_col if args.factor_col else infer_factor_col(factor_path)
    signal = args.signal if args.signal else factor_col.replace("hidden_factor_", "")

    out_base = Path("output") / args.run_name
    pricing_dir = out_base / "pricing" / signal
    report_dir = out_base / "reports"

    pricing_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    print("factor:", factor_path, flush=True)
    print("factor_col:", factor_col, flush=True)
    print("signal:", signal, flush=True)
    print("target:", args.target, flush=True)
    print("run_name:", args.run_name, flush=True)

    factor = load_factor_minute(
        path=factor_path,
        factor_col=factor_col,
        chunksize=args.chunksize,
    )

    df = load_merged_market_minute(
        market_dir=args.market_dir,
        market_glob=args.market_glob,
        factor=factor,
        factor_col=factor_col,
        target=args.target,
        chunksize=args.chunksize,
    )

    intercept, beta, rank_ic_calib, n_calib = fit_calibration(
        df=df,
        factor_col=factor_col,
        target=args.target,
        method=args.calibration,
    )

    pred_col = f"pred_ret_{signal}"
    fair_col = f"fair_price_{signal}"
    alpha_price_col = f"alpha_price_{signal}"
    alpha_ticks_col = f"alpha_ticks_{signal}"
    signal_col = f"signal_{signal}"

    df[signal_col] = df[factor_col]
    df[pred_col] = intercept + beta * df[factor_col]
    df[fair_col] = df["mid_price"] * (1.0 + df[pred_col])
    df[alpha_price_col] = df[fair_col] - df["mid_price"]
    df[alpha_ticks_col] = df[alpha_price_col] / args.tick_size
    df["target_ret"] = df[args.target]
    df["split"] = "insample_minute"

    signal_rank_ic = sampled_corr(df[signal_col], df["target_ret"], method="spearman")
    pred_rank_ic = sampled_corr(df[pred_col], df["target_ret"], method="spearman")
    pred_pearson = sampled_corr(df[pred_col], df["target_ret"], method="pearson")

    err = df[pred_col] - df["target_ret"]

    metrics = {
        "signal": signal,
        "split": "insample_minute",
        "n": len(df),
        "signal_rank_ic": signal_rank_ic,
        "pred_ret_rank_ic": pred_rank_ic,
        "pred_ret_pearson_ic": pred_pearson,
        "mse": float((err ** 2).mean()),
        "mae": float(err.abs().mean()),
        "pred_ret_mean": float(df[pred_col].mean()),
        "pred_ret_std": float(df[pred_col].std()),
        "target_mean": float(df["target_ret"].mean()),
        "target_std": float(df["target_ret"].std()),
        "date_min": df["date"].min(),
        "date_max": df["date"].max(),
        "num_dates": df["date"].nunique(),
        "num_stocks": df["securityid"].nunique(),
    }

    calib = {
        "signal": signal,
        "raw_col": factor_col,
        "target_col": args.target,
        "calibration_method": args.calibration,
        "intercept": intercept,
        "beta": beta,
        "calibration_signal_rank_ic": rank_ic_calib,
        "n_pricing_rows": n_calib,
    }

    out_cols = [
        "date", "datetime", "securityid", "split",
        "bid1", "ask1", "mid_price", "spread",
        "marketValue", "turnoverRate", "volatility_60",
        signal_col, pred_col, fair_col, alpha_price_col, alpha_ticks_col,
        "target_ret",
    ]
    out_cols = [c for c in out_cols if c in df.columns]

    priced_path = pricing_dir / "priced_dataset.csv"
    df[out_cols].to_csv(priced_path, index=False)

    pd.DataFrame([metrics]).to_csv(report_dir / "pricing_metrics_by_signal.csv", index=False)
    pd.DataFrame([calib]).to_csv(report_dir / "pricing_calibration.csv", index=False)

    print("saved priced:", priced_path, flush=True)
    print("saved metrics:", report_dir / "pricing_metrics_by_signal.csv", flush=True)
    print("saved calib:", report_dir / "pricing_calibration.csv", flush=True)
    print(pd.DataFrame([metrics]).to_string(index=False), flush=True)
    print(pd.DataFrame([calib]).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
