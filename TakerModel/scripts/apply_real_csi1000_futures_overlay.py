# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def pick_col(df, candidates, required=True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"cannot find any of {candidates}; columns={list(df.columns)}")
    return None


def compound_curve(x):
    x = pd.Series(x).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return (1.0 + x).cumprod() - 1.0


def compound_return(x):
    x = pd.Series(x).replace([np.inf, -np.inf], np.nan).dropna()
    if x.empty:
        return np.nan
    return float((1.0 + x).prod() - 1.0)


def make_ticks(df):
    first = df.groupby("date", as_index=False)["bar_index"].min()
    n = len(first)
    step = 1 if n <= 12 else 2 if n <= 24 else max(1, n // 12)
    ticks = first.iloc[::step]
    return ticks["bar_index"].tolist(), ticks["date"].astype(str).tolist()


def load_futures(path):
    fut = pd.read_csv(path, low_memory=False)

    dt_col = pick_col(fut, ["datetime", "execution_datetime", "timestamp", "time"])
    fut["datetime"] = pd.to_datetime(fut[dt_col])

    if "futures_ret" in fut.columns:
        fut["futures_ret"] = pd.to_numeric(fut["futures_ret"], errors="coerce")
    else:
        px_col = pick_col(
            fut,
            ["futures_price", "close", "last_price", "LastPrice", "price", "mid_price"]
        )
        fut["futures_price"] = pd.to_numeric(fut[px_col], errors="coerce")
        fut = fut.sort_values("datetime")
        fut["date"] = fut["datetime"].dt.strftime("%Y%m%d").astype(int)

        # 不计算隔夜跳空，每天第一分钟收益设为 0
        fut["futures_ret"] = fut.groupby("date")["futures_price"].pct_change()
        fut["futures_ret"] = fut["futures_ret"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    fut = fut[["datetime", "futures_ret"]].copy()
    fut = fut.dropna(subset=["datetime"])
    fut["futures_ret"] = fut["futures_ret"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    fut = fut.drop_duplicates("datetime").sort_values("datetime").reset_index(drop=True)

    return fut


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curve-csv", required=True)
    ap.add_argument("--futures-csv", required=True)
    ap.add_argument("--out-dir", required=True)

    ap.add_argument("--target-total-gross", type=float, default=1.0)
    ap.add_argument("--leverage", type=float, default=10.0)
    ap.add_argument("--futures-fee-rate", type=float, default=0.000023)
    ap.add_argument("--allow-short-futures", type=int, default=0)

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    curve = pd.read_csv(args.curve_csv, low_memory=False)
    curve["datetime"] = pd.to_datetime(curve["datetime"])
    curve = curve.sort_values(["date", "datetime"]).reset_index(drop=True)

    fut = load_futures(args.futures_csv)

    df = curve.merge(fut, on="datetime", how="left")
    df["futures_ret"] = df["futures_ret"].fillna(0.0)

    if "held_gross_prev" in df.columns:
        gross_col = "held_gross_prev"
    elif "target_gross" in df.columns:
        gross_col = "target_gross"
    else:
        raise KeyError("curve csv must contain held_gross_prev or target_gross")

    df["stock_gross"] = pd.to_numeric(df[gross_col], errors="coerce").fillna(0.0)

    raw_notional = float(args.target_total_gross) - df["stock_gross"]
    if int(args.allow_short_futures) == 1:
        df["futures_notional"] = raw_notional
    else:
        df["futures_notional"] = raw_notional.clip(lower=0.0)

    prev_notional = df["futures_notional"].shift(1)
    prev_notional.iloc[0] = df["futures_notional"].iloc[0]

    df["futures_turnover"] = (df["futures_notional"] - prev_notional).abs()
    df["futures_margin_used"] = df["futures_notional"].abs() / float(args.leverage)

    df["futures_pnl_ret"] = df["futures_notional"] * df["futures_ret"]
    df["futures_fee_ret"] = df["futures_turnover"] * float(args.futures_fee_rate)

    df["actual_ret_with_futures"] = (
        df["actual_ret"]
        + df["futures_pnl_ret"]
        - df["futures_fee_ret"]
    )

    df["actualret_with_futures"] = compound_curve(df["actual_ret_with_futures"])
    df["benchmarkret_full"] = compound_curve(df["benchmark_ret"])
    df["alpharet_with_futures"] = df["actualret_with_futures"] - df["benchmarkret_full"]

    df["alpha_ret_with_futures"] = df["actual_ret_with_futures"] - df["benchmark_ret"]
    df["alpharet_with_futures_compound"] = compound_curve(df["alpha_ret_with_futures"])

    if "actualret_clean" in df.columns:
        df["actualret_no_futures"] = df["actualret_clean"]
    elif "actualret" in df.columns:
        df["actualret_no_futures"] = df["actualret"]
    else:
        df["actualret_no_futures"] = compound_curve(df["actual_ret"])

    if "benchmarkret_clean" in df.columns:
        df["benchmarkret_no_futures"] = df["benchmarkret_clean"]
    elif "benchmarkret" in df.columns:
        df["benchmarkret_no_futures"] = df["benchmarkret"]
    else:
        df["benchmarkret_no_futures"] = compound_curve(df["benchmark_ret"])

    df["alpharet_no_futures"] = df["actualret_no_futures"] - df["benchmarkret_no_futures"]

    df["bar_index"] = np.arange(len(df))

    out_csv = out_dir / "curve_with_real_csi1000_futures_overlay.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    daily = (
        df.groupby("date", as_index=False)
        .agg(
            actual_return_with_futures=("actual_ret_with_futures", compound_return),
            benchmark_return=("benchmark_ret", compound_return),
            stock_actual_return=("actual_ret", compound_return),
            avg_stock_gross=("stock_gross", "mean"),
            avg_futures_notional=("futures_notional", "mean"),
            avg_margin_used=("futures_margin_used", "mean"),
            futures_turnover=("futures_turnover", "sum"),
            futures_fee_ret=("futures_fee_ret", "sum"),
            n_minutes=("datetime", "nunique"),
        )
    )

    daily["excess_return_with_futures"] = daily["actual_return_with_futures"] - daily["benchmark_return"]
    daily["actualret_with_futures"] = compound_curve(daily["actual_return_with_futures"])
    daily["benchmarkret"] = compound_curve(daily["benchmark_return"])
    daily["alpharet_with_futures"] = daily["actualret_with_futures"] - daily["benchmarkret"]

    ex = daily["excess_return_with_futures"].dropna()
    if len(ex) >= 2 and ex.std(ddof=1) > 0:
        daily_sharpe = ex.mean() / ex.std(ddof=1) * np.sqrt(252)
        daily_tstat = ex.mean() / ex.std(ddof=1) * np.sqrt(len(ex))
    else:
        daily_sharpe = np.nan
        daily_tstat = np.nan

    daily["daily_excess_sharpe_with_futures"] = daily_sharpe
    daily["daily_excess_tstat_with_futures"] = daily_tstat

    daily_csv = out_dir / "daily_with_real_csi1000_futures_overlay.csv"
    daily.to_csv(daily_csv, index=False, encoding="utf-8-sig")

    tick_pos, tick_lab = make_ticks(df)

    plt.figure(figsize=(14, 6))
    plt.plot(df["bar_index"], df["actualret_with_futures"] * 100.0, label="actualret_with_futures")
    plt.plot(df["bar_index"], df["benchmarkret_full"] * 100.0, label="benchmarkret_full")
    plt.plot(df["bar_index"], df["alpharet_with_futures"] * 100.0, label="alpharet_with_futures")
    plt.axhline(0.0, linewidth=0.8)
    plt.xticks(tick_pos, tick_lab, rotation=45)
    plt.title("Pure-CS NAV with Real CSI1000 Futures Overlay")
    plt.xlabel("trading minute index")
    plt.ylabel("cumulative return (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "curve_with_real_csi1000_futures_overlay.png", dpi=180)
    plt.close()

    plt.figure(figsize=(14, 5))
    plt.plot(df["bar_index"], df["alpharet_no_futures"] * 100.0, label="alpha_no_futures")
    plt.plot(df["bar_index"], df["alpharet_with_futures"] * 100.0, label="alpha_with_futures")
    plt.axhline(0.0, linewidth=0.8)
    plt.xticks(tick_pos, tick_lab, rotation=45)
    plt.title("Alpha Before and After Real CSI1000 Futures Overlay")
    plt.xlabel("trading minute index")
    plt.ylabel("cumulative alpha (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "alpha_before_after_real_csi1000_futures_overlay.png", dpi=180)
    plt.close()

    summary = pd.DataFrame([{
        "target_total_gross": float(args.target_total_gross),
        "leverage": float(args.leverage),
        "futures_fee_rate": float(args.futures_fee_rate),
        "avg_stock_gross": df["stock_gross"].mean(),
        "avg_futures_notional": df["futures_notional"].mean(),
        "avg_margin_used": df["futures_margin_used"].mean(),
        "total_futures_turnover": df["futures_turnover"].sum(),
        "total_futures_fee_return": df["futures_fee_ret"].sum(),
        "final_actualret_no_futures": df["actualret_no_futures"].iloc[-1],
        "final_alpharet_no_futures": df["alpharet_no_futures"].iloc[-1],
        "final_actualret_with_futures": df["actualret_with_futures"].iloc[-1],
        "final_benchmarkret_full": df["benchmarkret_full"].iloc[-1],
        "final_alpharet_with_futures": df["alpharet_with_futures"].iloc[-1],
        "final_alpharet_with_futures_compound": df["alpharet_with_futures_compound"].iloc[-1],
        "daily_excess_sharpe_with_futures": daily_sharpe,
        "daily_excess_tstat_with_futures": daily_tstat,
        "n_minutes": len(df),
        "n_days": daily["date"].nunique(),
    }])

    summary_csv = out_dir / "summary_with_real_csi1000_futures_overlay.csv"
    summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    print("\n===== real CSI1000 futures overlay summary =====")
    print(summary.to_string(index=False))
    print("\n[saved]", summary_csv)


if __name__ == "__main__":
    main()
