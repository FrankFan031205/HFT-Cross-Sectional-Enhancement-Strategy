# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def pick_col(df, candidates, name, required=True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"cannot find {name}; candidates={candidates}; columns={df.columns.tolist()}")
    return None


def compound(x):
    x = pd.to_numeric(x, errors="coerce").fillna(0.0)
    return float((1.0 + x).prod() - 1.0)


def cumret(x):
    x = pd.to_numeric(x, errors="coerce").fillna(0.0)
    return (1.0 + x).cumprod() - 1.0


def daily_sharpe(df, ret_col, bench_col):
    x = df.copy()
    x["date"] = pd.to_datetime(x["datetime"]).dt.strftime("%Y%m%d").astype(int)
    daily = x.groupby("date").agg(
        actual_day=(ret_col, compound),
        bench_day=(bench_col, compound),
    )
    daily["excess_day"] = daily["actual_day"] - daily["bench_day"]
    y = daily["excess_day"].dropna()
    if len(y) < 2 or y.std(ddof=1) == 0:
        return np.nan
    return float(y.mean() / y.std(ddof=1) * np.sqrt(252))


def normalize_datetime(df):
    df = df.copy()

    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        return df

    date_col = pick_col(df, ["date", "trading_date", "Date"], "date")
    time_col = pick_col(df, ["time", "timestamp", "ts_real", "ts", "Time"], "time")

    def fmt_time(v):
        s = str(int(v))
        # 93000000 -> 09:30:00, 930000 -> 09:30:00
        if len(s) >= 8:
            s = s.zfill(9)[:6]
        else:
            s = s.zfill(6)
        return f"{s[:2]}:{s[2:4]}:{s[4:6]}"

    df["datetime"] = pd.to_datetime(
        df[date_col].astype(int).astype(str) + " " + df[time_col].map(fmt_time)
    )
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curve-csv", required=True)
    ap.add_argument("--futures-csv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", required=True)

    ap.add_argument("--target-total-exposure", type=float, default=1.0)
    ap.add_argument("--max-overlay-gross", type=float, default=0.25)
    ap.add_argument("--futures-fee-rate", type=float, default=0.000023)
    ap.add_argument("--futures-leverage", type=float, default=10.0)
    ap.add_argument("--stock-gross-col", default="")
    ap.add_argument("--futures-ret-col", default="")
    ap.add_argument("--futures-price-col", default="")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    curve = pd.read_csv(args.curve_csv)
    curve["datetime"] = pd.to_datetime(curve["datetime"])
    curve = curve.sort_values("datetime").reset_index(drop=True)

    stock_ret_col = pick_col(curve, ["actual_ret", "stock_actual_ret", "strategy_ret"], "stock return")
    bench_ret_col = pick_col(curve, ["benchmark_ret", "bench_ret"], "benchmark return")

    if args.stock_gross_col:
        gross_col = args.stock_gross_col
    else:
        gross_col = pick_col(
            curve,
            [
                "gross_prev_to_capital",
                "actual_gross_prev_to_capital",
                "gross_after_to_capital",
                "actual_gross_after",
                "actual_gross",
                "stock_gross",
                "target_gross",
                "gross_prev",
                "gross",
            ],
            "stock gross",
            required=False,
        )

    if gross_col is None:
        raise KeyError(
            "curve csv has no stock gross column. Please inspect columns and pass --stock-gross-col. "
            f"columns={curve.columns.tolist()}"
        )

    fut = pd.read_csv(args.futures_csv)
    fut = normalize_datetime(fut)
    fut = fut.sort_values("datetime").reset_index(drop=True)
    fut["minute_dt"] = fut["datetime"].dt.floor("min")

    if args.futures_ret_col:
        fut_ret_col = args.futures_ret_col
    elif "futures_ret" in fut.columns:
        fut_ret_col = "futures_ret"
    else:
        fut_ret_col = ""

    if fut_ret_col:
        fut_min = (
            fut.groupby("minute_dt", as_index=False)
            .agg(futures_ret=(fut_ret_col, "last"))
        )
    else:
        price_col = args.futures_price_col or pick_col(
            fut,
            ["price", "close", "close_price", "lastprice", "last_price", "mid_price", "settle", "settlement"],
            "futures price",
        )
        fut_min = (
            fut.groupby("minute_dt", as_index=False)
            .agg(futures_price=(price_col, "last"))
        )
        fut_min["date"] = fut_min["minute_dt"].dt.strftime("%Y%m%d").astype(int)
        fut_min["futures_ret"] = fut_min.groupby("date")["futures_price"].pct_change().fillna(0.0)

    curve["minute_dt"] = curve["datetime"].dt.floor("min")
    df = curve.merge(fut_min[["minute_dt", "futures_ret"]], on="minute_dt", how="left")
    df["futures_ret"] = pd.to_numeric(df["futures_ret"], errors="coerce").fillna(0.0)

    df["stock_ret"] = pd.to_numeric(df[stock_ret_col], errors="coerce").fillna(0.0)
    df["benchmark_ret"] = pd.to_numeric(df[bench_ret_col], errors="coerce").fillna(0.0)
    df["stock_gross"] = pd.to_numeric(df[gross_col], errors="coerce").fillna(0.0)

    # 核心逻辑：用剩余现金暴露做多 CSI1000 期货
    df["target_futures_overlay_gross"] = (
        args.target_total_exposure - df["stock_gross"]
    ).clip(lower=0.0, upper=float(args.max_overlay_gross))

    # 上一分钟已经持有的期货 overlay 产生本分钟 PnL
    df["futures_overlay_prev_gross"] = df["target_futures_overlay_gross"].shift(1).fillna(0.0)
    df["futures_pnl_ret"] = df["futures_overlay_prev_gross"] * df["futures_ret"]

    # 本分钟调期货仓的成本
    df["futures_overlay_turnover"] = (
        df["target_futures_overlay_gross"] - df["futures_overlay_prev_gross"]
    ).abs()
    df["futures_fee_ret"] = df["futures_overlay_turnover"] * float(args.futures_fee_rate)

    df["actual_ret_with_futures"] = (
        df["stock_ret"] + df["futures_pnl_ret"] - df["futures_fee_ret"]
    )

    df["stock_actualret"] = cumret(df["stock_ret"])
    df["benchmarkret"] = cumret(df["benchmark_ret"])
    df["actualret_with_futures"] = cumret(df["actual_ret_with_futures"])

    df["stock_alpharet"] = df["stock_actualret"] - df["benchmarkret"]
    df["alpharet_with_futures"] = df["actualret_with_futures"] - df["benchmarkret"]

    stock_return = compound(df["stock_ret"])
    benchmark_return = compound(df["benchmark_ret"])
    actual_return_with_futures = compound(df["actual_ret_with_futures"])

    summary = pd.DataFrame([{
        "tag": args.tag,
        "stock_return": stock_return,
        "benchmark_return": benchmark_return,
        "stock_alpha_return": stock_return - benchmark_return,
        "actual_return_with_futures": actual_return_with_futures,
        "alpha_return_with_futures": actual_return_with_futures - benchmark_return,
        "daily_excess_sharpe_stock": daily_sharpe(df, "stock_ret", "benchmark_ret"),
        "daily_excess_sharpe_with_futures": daily_sharpe(df, "actual_ret_with_futures", "benchmark_ret"),
        "avg_stock_gross": float(df["stock_gross"].mean()),
        "avg_futures_overlay_gross": float(df["target_futures_overlay_gross"].mean()),
        "max_futures_overlay_gross": float(df["target_futures_overlay_gross"].max()),
        "avg_total_exposure": float((df["stock_gross"] + df["target_futures_overlay_gross"]).mean()),
        "futures_overlay_turnover": float(df["futures_overlay_turnover"].sum()),
        "futures_fee_return": float(df["futures_fee_ret"].sum()),
        "futures_fee_rate": float(args.futures_fee_rate),
        "futures_leverage": float(args.futures_leverage),
        "avg_futures_margin_usage": float((df["target_futures_overlay_gross"] / args.futures_leverage).mean()),
        "n_rows": len(df),
        "start": str(df["datetime"].iloc[0]),
        "end": str(df["datetime"].iloc[-1]),
    }])

    curve_out = out_dir / f"{args.tag}_stock_plus_csi1000_futures_overlay_curve.csv"
    summary_out = out_dir / f"{args.tag}_stock_plus_csi1000_futures_overlay_summary.csv"
    png_out = out_dir / f"{args.tag}_stock_plus_csi1000_futures_overlay_nav.png"

    df.to_csv(curve_out, index=False)
    summary.to_csv(summary_out, index=False)

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(df["datetime"], df["stock_actualret"] * 100.0, label="stock strategy")
    ax.plot(df["datetime"], df["actualret_with_futures"] * 100.0, label="stock + CSI1000 futures overlay")
    ax.plot(df["datetime"], df["benchmarkret"] * 100.0, label="CSI2000 benchmark")
    ax.plot(df["datetime"], df["alpharet_with_futures"] * 100.0, label="excess with futures")
    ax.axhline(0, linewidth=1)
    ax.set_title("Stock + CSI1000 Futures Overlay vs CSI2000 Benchmark")
    ax.set_xlabel("datetime")
    ax.set_ylabel("cumulative return / excess (%)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    s = summary.iloc[0]
    txt = (
        f"Stock alpha: {s['stock_alpha_return']*100:.2f}%\n"
        f"With futures alpha: {s['alpha_return_with_futures']*100:.2f}%\n"
        f"Benchmark: {s['benchmark_return']*100:.2f}%\n"
        f"Avg stock gross: {s['avg_stock_gross']*100:.2f}%\n"
        f"Avg futures overlay: {s['avg_futures_overlay_gross']*100:.2f}%\n"
        f"Sharpe with futures: {s['daily_excess_sharpe_with_futures']:.2f}"
    )
    ax.text(
        1.02, 0.95, txt,
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round", alpha=0.15),
    )

    fig.tight_layout()
    fig.savefig(png_out, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print("curve:", curve_out)
    print("summary:", summary_out)
    print("png:", png_out)
    print("\n===== stock + futures overlay summary =====")
    print(summary.T.to_string())


if __name__ == "__main__":
    main()
