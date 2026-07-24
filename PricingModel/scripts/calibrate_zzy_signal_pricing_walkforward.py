# -*- coding: utf-8 -*-
"""
Walk-forward PricingModel adapter for external ZZY signals.

Input:
  TakerOptimizer zzy optimizer_input_YYYYMMDD.parquet

Use:
  signal_z as external alpha signal
  fwd_ret_label as calibration target only on previous days

Output:
  pricing_zzy_pm_h5_res.csv with:
    pred_ret_zzy_pm_h5_res
    fair_price_zzy_pm_h5_res
    buy_edge_bps_zzy_pm_h5_res
    sell_edge_bps_zzy_pm_h5_res

No current-day label leakage:
  For each target day D, fit calibration only on dates < D.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl


def fit_linear(x, y, max_abs_bps_per_z=30.0, max_abs_intercept_bps=20.0):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]

    if len(x) < 10000:
        return 0.0, 10.0 / 10000.0, len(x), "fallback_small_train"

    # Robust clipping for calibration stability.
    x = np.clip(x, -8.0, 8.0)
    lo, hi = np.nanpercentile(y, [0.5, 99.5])
    y = np.clip(y, lo, hi)

    vx = np.var(x)
    if vx <= 1e-12:
        return 0.0, 10.0 / 10000.0, len(x), "fallback_low_var"

    b = np.mean((x - x.mean()) * (y - y.mean())) / vx
    a = y.mean() - b * x.mean()

    b = float(np.clip(b, -max_abs_bps_per_z / 10000.0, max_abs_bps_per_z / 10000.0))
    a = float(np.clip(a, -max_abs_intercept_bps / 10000.0, max_abs_intercept_bps / 10000.0))

    return a, b, len(x), "walk_forward_ols"


def safe_corr(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 1000:
        return np.nan
    aa = a[m]
    bb = b[m]
    if np.std(aa) <= 1e-12 or np.std(bb) <= 1e-12:
        return np.nan
    return float(np.corrcoef(aa, bb)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model-name", default="zzy_pm_h5_res")

    ap.add_argument("--price-multiplier", type=float, default=0.01)
    ap.add_argument("--rebalance-ts-step", type=int, default=60)
    ap.add_argument("--max-spread-bps", type=float, default=50.0)

    ap.add_argument("--min-train-days", type=int, default=5)
    ap.add_argument("--max-train-days", type=int, default=10)
    ap.add_argument("--fallback-bps-per-z", type=float, default=10.0)
    ap.add_argument("--max-abs-bps-per-z", type=float, default=30.0)
    ap.add_argument("--max-abs-intercept-bps", type=float, default=20.0)
    ap.add_argument("--max-abs-pred-bps", type=float, default=200.0)
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    pricing_dir = out_dir / "pricing"
    report_dir = out_dir / "reports"
    pricing_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    files = [str(x) for x in sorted(in_dir.glob("optimizer_input_*.parquet"))]
    if not files:
        raise SystemExit(f"no optimizer_input_*.parquet under {in_dir}")

    lf0 = pl.scan_parquet(files)
    max_ts_real = lf0.select(pl.col("ts_real").cast(pl.Int64).max()).collect().item()
    div = 1000 if max_ts_real > 235959 else 1

    date_s = pl.col("date").cast(pl.Utf8)
    hhmmss = (pl.col("ts_real").cast(pl.Int64) // div).cast(pl.Int64)
    time_s = hhmmss.cast(pl.Utf8).str.zfill(6)

    datetime_str = pl.concat_str([
        date_s.str.slice(0, 4), pl.lit("-"),
        date_s.str.slice(4, 2), pl.lit("-"),
        date_s.str.slice(6, 2), pl.lit(" "),
        time_s.str.slice(0, 2), pl.lit(":"),
        time_s.str.slice(2, 2), pl.lit(":"),
        time_s.str.slice(4, 2),
    ])

    pred_col = f"pred_ret_{args.model_name}"
    fair_col = f"fair_price_{args.model_name}"
    buy_edge_col = f"buy_edge_bps_{args.model_name}"
    sell_edge_col = f"sell_edge_bps_{args.model_name}"

    df = (
        lf0.filter(
            (pl.col("ts") % int(args.rebalance_ts_step) == 0)
            & (pl.col("exec_spread_bps") <= float(args.max_spread_bps))
            & (pl.col("exec_mid_price") > 0)
            & (pl.col("exec_buy_price") > 0)
            & (pl.col("exec_sell_price") > 0)
            & (pl.col("volume_60s") > 0)
            & pl.col("signal_z").is_not_null()
            & pl.col("fwd_ret_label").is_not_null()
        )
        .with_columns([
            hhmmss.alias("minute"),
            datetime_str.alias("datetime"),
            pl.col("SecurityID").cast(pl.Utf8).alias("securityid"),

            (pl.col("exec_mid_price") * float(args.price_multiplier)).alias("mid_price"),
            (pl.col("exec_mid_price") * float(args.price_multiplier)).alias("price"),
            (pl.col("exec_mid_price") * float(args.price_multiplier)).alias("last_price"),
            (pl.col("exec_buy_price") * float(args.price_multiplier)).alias("ask_price"),
            (pl.col("exec_sell_price") * float(args.price_multiplier)).alias("bid_price"),

            ((pl.col("exec_buy_price") - pl.col("exec_sell_price")) * float(args.price_multiplier)).alias("spread"),
            pl.col("exec_spread_bps").alias("spread_bps_audit"),

            pl.col("signal_z").alias("signal_z"),
            pl.col("fwd_ret_label").alias("target_ret"),
            pl.col("volume_60s").alias("volume_60s"),
            (pl.col("max_participation_notional") * float(args.price_multiplier)).alias("capacity_notional"),
        ])
        .select([
            "date", "datetime", "minute", "securityid",
            "signal_z", "target_ret",
            "mid_price", "price", "last_price", "ask_price", "bid_price",
            "spread", "spread_bps_audit", "volume_60s", "capacity_notional",
        ])
        .collect()
        .sort(["date", "minute", "securityid"])
    )

    pdf = df.to_pandas()
    dates = sorted(pdf["date"].unique().tolist())

    pdf[pred_col] = np.nan
    pdf["pricing_calib_intercept_bps"] = np.nan
    pdf["pricing_calib_bps_per_z"] = np.nan
    pdf["pricing_calib_train_n"] = 0
    pdf["pricing_calib_mode"] = ""

    metrics = []

    for i, d in enumerate(dates):
        target_idx = pdf["date"].values == d
        train_dates = dates[max(0, i - args.max_train_days):i]

        if len(train_dates) < args.min_train_days:
            a = 0.0
            b = args.fallback_bps_per_z / 10000.0
            n = 0
            mode = "fallback_not_enough_days"
        else:
            train_idx = pdf["date"].isin(train_dates).values
            a, b, n, mode = fit_linear(
                pdf.loc[train_idx, "signal_z"].values,
                pdf.loc[train_idx, "target_ret"].values,
                max_abs_bps_per_z=args.max_abs_bps_per_z,
                max_abs_intercept_bps=args.max_abs_intercept_bps,
            )

        pred = a + b * pdf.loc[target_idx, "signal_z"].values
        pred = np.clip(pred, -args.max_abs_pred_bps / 10000.0, args.max_abs_pred_bps / 10000.0)

        pdf.loc[target_idx, pred_col] = pred
        pdf.loc[target_idx, "pricing_calib_intercept_bps"] = a * 10000.0
        pdf.loc[target_idx, "pricing_calib_bps_per_z"] = b * 10000.0
        pdf.loc[target_idx, "pricing_calib_train_n"] = int(n)
        pdf.loc[target_idx, "pricing_calib_mode"] = mode

        y = pdf.loc[target_idx, "target_ret"].values
        x = pdf.loc[target_idx, "signal_z"].values

        metrics.append({
            "date": int(d),
            "mode": mode,
            "train_dates": ",".join(map(str, train_dates)),
            "train_n": int(n),
            "intercept_bps": a * 10000.0,
            "bps_per_z": b * 10000.0,
            "signal_target_ic": safe_corr(x, y),
            "pred_target_ic": safe_corr(pred, y),
            "target_mean_bps": float(np.nanmean(y) * 10000.0),
            "target_std_bps": float(np.nanstd(y) * 10000.0),
            "pred_mean_bps": float(np.nanmean(pred) * 10000.0),
            "pred_std_bps": float(np.nanstd(pred) * 10000.0),
        })

    pdf[fair_col] = pdf["mid_price"] * (1.0 + pdf[pred_col])
    pdf[buy_edge_col] = (pdf[fair_col] / pdf["ask_price"] - 1.0) * 10000.0
    pdf[sell_edge_col] = (pdf["bid_price"] / pdf[fair_col] - 1.0) * 10000.0

    pricing_cols = [
        "date", "datetime", "minute", "securityid",
        pred_col, fair_col, buy_edge_col, sell_edge_col,
        "mid_price", "price", "last_price", "ask_price", "bid_price",
        "spread", "spread_bps_audit", "volume_60s", "capacity_notional",
        "pricing_calib_intercept_bps", "pricing_calib_bps_per_z",
        "pricing_calib_train_n", "pricing_calib_mode",
    ]

    pricing_path = pricing_dir / f"pricing_{args.model_name}.csv"
    metrics_path = report_dir / f"pricing_metrics_{args.model_name}.csv"

    pdf[pricing_cols].to_csv(pricing_path, index=False)
    pd.DataFrame(metrics).to_csv(metrics_path, index=False)

    print("[saved pricing]", pricing_path)
    print("[saved metrics]", metrics_path)
    print("[shape]", pdf.shape)
    print("[date-minute]", pdf[["date", "minute"]].drop_duplicates().shape[0])
    print("[per date]")
    print(pdf[["date", "minute"]].drop_duplicates().groupby("date").size().describe())
    print("[metrics tail]")
    print(pd.DataFrame(metrics).tail(10))


if __name__ == "__main__":
    main()
