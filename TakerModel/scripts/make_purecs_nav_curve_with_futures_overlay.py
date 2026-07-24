import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_any(path):
    path = Path(path)
    suf = "".join(path.suffixes).lower()
    if suf.endswith(".parquet"):
        return pd.read_parquet(path)
    if suf.endswith(".csv") or suf.endswith(".csv.gz"):
        return pd.read_csv(path, low_memory=False)
    raise ValueError(f"unsupported file: {path}")


def pick_col(df, candidates, required=True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"cannot find any of {candidates}; available={list(df.columns)}")
    return None


def compound(x):
    x = pd.Series(x).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return float((1.0 + x).prod() - 1.0)


def safe_std(x):
    x = pd.Series(x).dropna()
    if len(x) <= 1:
        return np.nan
    return float(x.std(ddof=1))


def annualized_daily_sharpe(excess_daily):
    x = pd.Series(excess_daily).dropna()
    if len(x) <= 1:
        return np.nan
    s = safe_std(x)
    if pd.isna(s) or s == 0:
        return np.nan
    return float(x.mean() / s * np.sqrt(252.0))


def pct_str(x):
    if pd.isna(x):
        return "NA"
    return f"{x * 100:.2f}%"


def num_str(x, nd=2):
    if pd.isna(x):
        return "NA"
    return f"{x:.{nd}f}"


def load_stock_curve(path):
    df = read_any(path).copy()

    dt_col = pick_col(df, ["datetime", "execution_datetime", "dt", "ts"])
    df["datetime"] = pd.to_datetime(df[dt_col])

    date_col = pick_col(df, ["date", "execution_date"], required=False)
    if date_col is not None:
        df["date"] = pd.to_numeric(df[date_col], errors="coerce").astype("Int64")
    else:
        df["date"] = df["datetime"].dt.strftime("%Y%m%d").astype(int)

    actual_ret_col = pick_col(
        df,
        ["actual_ret", "actual_strategy_ret", "strategy_ret", "actual_step_ret", "ret"]
    )
    benchmark_ret_col = pick_col(
        df,
        ["benchmark_ret", "full_benchmark_ret", "benchmark_return", "bench_ret"]
    )

    gross_col = pick_col(
        df,
        [
            "gross_prev_to_equity",
            "avg_gross_prev_to_equity",
            "actual_gross_after_to_equity",
            "actual_gross",
            "avg_actual_gross",
            "held_gross",
            "gross_prev_to_capital",
            "avg_gross_prev_to_capital",
        ]
    )

    target_gross_col = pick_col(
        df,
        ["target_gross", "avg_target_gross", "target_total_gross"],
        required=False,
    )

    out = pd.DataFrame({
        "date": df["date"].astype(int),
        "datetime": df["datetime"],
        "stock_actual_ret": pd.to_numeric(df[actual_ret_col], errors="coerce").fillna(0.0),
        "benchmark_ret": pd.to_numeric(df[benchmark_ret_col], errors="coerce").fillna(0.0),
        "stock_gross": pd.to_numeric(df[gross_col], errors="coerce").fillna(0.0),
    })

    # Keep stock rebalance indicator for rebalance-only futures adjustment.
    turnover_col = pick_col(df, ["turnover_weight", "turnover_to_capital", "turnover"], required=False)
    if turnover_col is not None:
        out["turnover_weight"] = pd.to_numeric(df[turnover_col], errors="coerce").fillna(0.0)
    else:
        out["turnover_weight"] = 0.0

    if target_gross_col is not None:
        out["target_gross"] = pd.to_numeric(df[target_gross_col], errors="coerce")
    else:
        out["target_gross"] = np.nan

    out = out.sort_values(["date", "datetime"]).reset_index(drop=True)
    out["minute_dt"] = out["datetime"].dt.floor("min")
    return out


def load_futures(path):
    df = read_any(path).copy()

    dt_col = pick_col(df, ["datetime", "dt", "ts", "timestamp"])
    df["datetime"] = pd.to_datetime(df[dt_col], errors="coerce")

    date_col = pick_col(df, ["date", "trading_date"], required=False)
    if date_col is not None:
        df["date"] = pd.to_numeric(df[date_col], errors="coerce").astype("Int64")
    else:
        df["date"] = df["datetime"].dt.strftime("%Y%m%d").astype(int)

    price_col = pick_col(
        df,
        ["futures_price", "last_price", "lastprice", "close", "price", "mid_price", "tmid"]
    )
    contract_col = pick_col(df, ["contract", "SecurityID", "securityid", "symbol"], required=False)

    out = pd.DataFrame({
        "date": df["date"].astype(int),
        "datetime": df["datetime"],
        "fut_price": pd.to_numeric(df[price_col], errors="coerce"),
    })

    if contract_col is not None:
        out["contract"] = df[contract_col].astype(str)
    else:
        out["contract"] = ""

    out = out.dropna(subset=["datetime", "fut_price"]).copy()
    out = out.sort_values(["date", "datetime"]).reset_index(drop=True)
    out["minute_dt"] = out["datetime"].dt.floor("min")
    return out


def align_futures_to_stock(stock_df, fut_df):
    left = stock_df.sort_values(["date", "minute_dt"]).reset_index(drop=True).copy()
    right = fut_df.sort_values(["date", "minute_dt"]).reset_index(drop=True).copy()

    merged = pd.merge_asof(
        left,
        right[["date", "minute_dt", "fut_price", "contract"]].sort_values(["date", "minute_dt"]),
        on="minute_dt",
        by="date",
        direction="backward",
        tolerance=pd.Timedelta("10min"),
    )

    merged["fut_price"] = merged.groupby("date")["fut_price"].ffill().bfill()
    merged["contract"] = merged.groupby("date")["contract"].ffill().bfill().fillna("")

    merged["fut_ret"] = (
        merged.groupby("date")["fut_price"]
        .pct_change()
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    return merged


def find_start_index(df, target_total_gross, start_gross_ratio, min_consecutive):
    if "target_gross" in df.columns and df["target_gross"].notna().any():
        tgt = df["target_gross"].fillna(target_total_gross)
    else:
        tgt = pd.Series(target_total_gross, index=df.index)

    cond = (df["stock_gross"] >= start_gross_ratio * tgt) & (df["stock_gross"] > 0)
    arr = cond.to_numpy()

    need = max(int(min_consecutive), 1)
    for i in range(0, max(len(arr) - need + 1, 1)):
        if arr[i:i + need].all():
            return i
    return 0


def build_overlay_curve(stock_df, fut_df, args):
    df = align_futures_to_stock(stock_df, fut_df)

    df["target_total_gross"] = float(args.target_total_gross)
    df["stock_gross"] = pd.to_numeric(df["stock_gross"], errors="coerce").fillna(0.0).clip(lower=0.0)

    # 期货目标补仓：补足到目标总仓位
    df["desired_futures_overlay_gross"] = (df["target_total_gross"] - df["stock_gross"]).clip(lower=0.0)
    if args.max_futures_overlay_gross is not None:
        df["desired_futures_overlay_gross"] = df["desired_futures_overlay_gross"].clip(
            upper=float(args.max_futures_overlay_gross)
        )

    # 更真实的口径：只在股票 rebalance / 有 turnover 的时点调整期货；
    # 其他分钟保持原期货仓位，不按股票市值浮动每分钟调仓。
    if "turnover_weight" in df.columns:
        rebalance_mask = df["turnover_weight"].fillna(0.0).abs() > 1e-12
    else:
        rebalance_mask = pd.Series(False, index=df.index)

    if len(df) > 0:
        rebalance_mask.iloc[0] = True

    df["futures_overlay_gross"] = np.nan
    df.loc[rebalance_mask, "futures_overlay_gross"] = df.loc[rebalance_mask, "desired_futures_overlay_gross"]
    df["futures_overlay_gross"] = df["futures_overlay_gross"].ffill().fillna(0.0)

    # 当前分钟收益用上一分钟已经持有的期货仓位，避免 lookahead。
    df["futures_overlay_gross_prev"] = df["futures_overlay_gross"].shift(1).fillna(0.0)

    # 只有调整期货时才产生期货换手。
    df["futures_turnover_gross"] = 0.0
    prev_overlay = df["futures_overlay_gross"].shift(1).fillna(0.0)
    df.loc[rebalance_mask, "futures_turnover_gross"] = (
        df.loc[rebalance_mask, "futures_overlay_gross"] - prev_overlay.loc[rebalance_mask]
    ).abs()

    # 期货收益与手续费
    df["futures_fee_return"] = df["futures_turnover_gross"] * float(args.futures_fee_rate)
    df["futures_pnl_return"] = df["futures_overlay_gross_prev"] * df["fut_ret"]

    # 总组合收益
    df["actual_ret_with_futures"] = (
        df["stock_actual_ret"]
        + df["futures_pnl_return"]
        - df["futures_fee_return"]
    )
    df["alpha_ret"] = df["actual_ret_with_futures"] - df["benchmark_ret"]

    # 记录期货保证金占用（仅展示）
    df["futures_margin_used"] = df["futures_overlay_gross"] / float(args.futures_leverage)

    # 去掉最开头那段未真正进入有效持仓的窗口
    start_idx = find_start_index(
        df=df,
        target_total_gross=float(args.target_total_gross),
        start_gross_ratio=float(args.start_gross_ratio),
        min_consecutive=int(args.min_consecutive),
    )
    df = df.iloc[start_idx:].copy().reset_index(drop=True)

    # 重新压缩横轴
    df["bar_index"] = np.arange(len(df))

    # 重新计算 cumulative curve
    df["cum_actualret"] = (1.0 + df["actual_ret_with_futures"]).cumprod() - 1.0
    df["cum_benchmarkret"] = (1.0 + df["benchmark_ret"]).cumprod() - 1.0
    df["cum_alpharet"] = df["cum_actualret"] - df["cum_benchmarkret"]
    df["cum_alpharet_compound"] = (1.0 + df["alpha_ret"]).cumprod() - 1.0

    return df


def build_daily(df):
    daily = (
        df.groupby("date", as_index=False)
        .agg(
            actual_day_ret=("actual_ret_with_futures", compound),
            benchmark_day_ret=("benchmark_ret", compound),
            alpha_day_ret=("alpha_ret", compound),
            avg_stock_gross=("stock_gross", "mean"),
            avg_futures_overlay_gross=("futures_overlay_gross", "mean"),
            avg_margin_used=("futures_margin_used", "mean"),
            futures_turnover_gross=("futures_turnover_gross", "sum"),
            futures_fee_return=("futures_fee_return", "sum"),
        )
        .sort_values("date")
        .reset_index(drop=True)
    )
    daily["daily_excess"] = daily["actual_day_ret"] - daily["benchmark_day_ret"]
    return daily


def build_summary(df, daily, args):
    summary = pd.DataFrame([{
        "start_date": int(df["date"].iloc[0]),
        "end_date": int(df["date"].iloc[-1]),
        "start_datetime": str(df["datetime"].iloc[0]),
        "end_datetime": str(df["datetime"].iloc[-1]),
        "target_total_gross": float(args.target_total_gross),
        "actual_return": compound(df["actual_ret_with_futures"]),
        "benchmark_return": compound(df["benchmark_ret"]),
        "alpha_return": compound(df["actual_ret_with_futures"]) - compound(df["benchmark_ret"]),
        "alpha_compound_return": compound(df["alpha_ret"]),
        "daily_excess_sharpe": annualized_daily_sharpe(daily["daily_excess"]),
        "avg_stock_gross": float(df["stock_gross"].mean()),
        "avg_futures_overlay_gross": float(df["futures_overlay_gross"].mean()),
        "avg_margin_used": float(df["futures_margin_used"].mean()),
        "total_futures_turnover_gross": float(df["futures_turnover_gross"].sum()),
        "total_futures_fee_return": float(df["futures_fee_return"].sum()),
        "futures_fee_rate": float(args.futures_fee_rate),
        "futures_leverage": float(args.futures_leverage),
        "n_minutes": int(len(df)),
        "n_days": int(df["date"].nunique()),
    }])
    return summary


def plot_curve(df, summary, out_png, title):
    fig = plt.figure(figsize=(18, 9))
    gs = fig.add_gridspec(1, 2, width_ratios=[6.2, 1.6], wspace=0.05)

    ax = fig.add_subplot(gs[0, 0])
    ax_info = fig.add_subplot(gs[0, 1])
    ax_info.axis("off")

    x = df["bar_index"].to_numpy()
    ax.plot(x, df["cum_actualret"] * 100.0, label="actualret with futures")
    ax.plot(x, df["cum_benchmarkret"] * 100.0, label="benchmarkret")
    ax.plot(x, df["cum_alpharet"] * 100.0, label="alpharet")

    ax.axhline(0.0, linewidth=0.8, linestyle="--")
    ax.grid(True, alpha=0.3)
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_ylabel("cumulative return (%)", fontsize=12)
    ax.set_xlabel("trading minute index, initial pre-position window removed", fontsize=11)

    # x轴按交易日打点
    first_of_day = df.groupby("date", as_index=False).first()[["date", "bar_index"]]
    if len(first_of_day) > 0:
        step = max(1, len(first_of_day) // 10)
        tick_df = first_of_day.iloc[::step].copy()
        ax.set_xticks(tick_df["bar_index"].tolist())
        ax.set_xticklabels(tick_df["date"].astype(str).tolist(), rotation=45, ha="right")

    ax.legend(loc="lower left", fontsize=11)

    row = summary.iloc[0]
    text = (
        "Summary\n\n"
        f"Strategy Return\n{pct_str(row['actual_return'])}\n\n"
        f"Benchmark Return\n{pct_str(row['benchmark_return'])}\n\n"
        f"Alpha Return\n{pct_str(row['alpha_return'])}\n\n"
        f"Daily Excess Sharpe\n{num_str(row['daily_excess_sharpe'], 2)}"
    )

    ax_info.text(
        0.02, 0.95, text,
        va="top", ha="left", fontsize=12,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="0.6", alpha=0.95)
    )

    plt.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stock-curve", required=True, help="stock-only minute curve csv/parquet")
    ap.add_argument("--futures", required=True, help="CSI1000 futures minute csv/parquet")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", default="purecs_with_futures_overlay")
    ap.add_argument("--target-total-gross", type=float, default=0.95)
    ap.add_argument("--max-futures-overlay-gross", type=float, default=None)
    ap.add_argument("--futures-fee-rate", type=float, default=0.000023)  # 0.23 bps
    ap.add_argument("--futures-leverage", type=float, default=10.0)
    ap.add_argument("--start-gross-ratio", type=float, default=0.90)
    ap.add_argument("--min-consecutive", type=int, default=5)
    ap.add_argument("--title", default="Pure-CS NAV with CSI1000 Futures Overlay")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stock_df = load_stock_curve(args.stock_curve)
    fut_df = load_futures(args.futures)

    curve = build_overlay_curve(stock_df, fut_df, args)
    daily = build_daily(curve)
    summary = build_summary(curve, daily, args)

    curve_out = out_dir / f"{args.tag}_nav_curve_minute_stock_plus_futures.csv"
    daily_out = out_dir / f"{args.tag}_nav_curve_daily_stock_plus_futures.csv"
    summary_out = out_dir / f"{args.tag}_nav_curve_summary_stock_plus_futures.csv"
    png_out = out_dir / f"{args.tag}_nav_curve_stock_plus_futures.png"

    curve.to_csv(curve_out, index=False)
    daily.to_csv(daily_out, index=False)
    summary.to_csv(summary_out, index=False)

    plot_curve(curve, summary, png_out, args.title)

    print("===== final summary =====")
    print(summary.T.to_string(header=False))
    print()
    print("[saved curve ]", curve_out)
    print("[saved daily ]", daily_out)
    print("[saved summ  ]", summary_out)
    print("[saved png   ]", png_out)


if __name__ == "__main__":
    main()
