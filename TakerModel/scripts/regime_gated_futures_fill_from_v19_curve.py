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


def compound(s):
    s = pd.to_numeric(s, errors="coerce").fillna(0.0)
    return float((1.0 + s).prod() - 1.0)


def daily_sharpe(df, actual_col, bench_col):
    x = df.copy()
    x["date"] = pd.to_datetime(x["datetime"]).dt.strftime("%Y%m%d").astype(int)
    d = x.groupby("date").agg(
        actual_day=(actual_col, compound),
        bench_day=(bench_col, compound),
    )
    e = d["actual_day"] - d["bench_day"]
    if len(e) < 2 or e.std(ddof=1) == 0:
        return np.nan
    return float(e.mean() / e.std(ddof=1) * np.sqrt(252))


def make_daily_table(df, actual_col, bench_col):
    x = df.copy()
    x["date"] = pd.to_datetime(x["datetime"]).dt.strftime("%Y%m%d").astype(int)
    d = x.groupby("date").agg(
        actual=(actual_col, compound),
        bench=(bench_col, compound),
        avg_stock_gross=("stock_gross", "mean"),
        avg_futures=("target_futures_overlay_gross", "mean"),
        fut_fee=("futures_fee_ret", "sum"),
        up_regime_ratio=("up_regime", "mean"),
    )
    d["excess"] = d["actual"] - d["bench"]
    return d.reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-curve", required=True, help="v19 curve with futures_ret and stock_gross columns")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", required=True)

    ap.add_argument("--target-total-exposure", type=float, default=1.0)
    ap.add_argument("--max-overlay-gross", type=float, default=0.15)
    ap.add_argument("--futures-fee-rate", type=float, default=0.000023)

    # regime：用上一分钟的 benchmark 日内累计收益判断是否补 beta
    ap.add_argument("--threshold", type=float, default=0.0)
    ap.add_argument("--min-bars", type=int, default=10)

    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    x = pd.read_csv(args.base_curve)
    x["datetime"] = pd.to_datetime(x["datetime"])
    x = x.sort_values("datetime").reset_index(drop=True)
    x["date"] = x["datetime"].dt.strftime("%Y%m%d").astype(int)
    x["bar_in_day"] = x.groupby("date").cumcount()

    # base actual / benchmark 用 v18 stock-only + overnight，不用已有 v19 futures actual
    actual_base_col = pick_col(
        x,
        ["actual_ret_with_overnight"],
        "base actual return",
    )
    bench_eval_col = pick_col(
        x,
        ["benchmark_ret_with_overnight"],
        "benchmark return with overnight",
    )

    # regime signal 用 no-overnight intraday benchmark return
    bench_signal_col = pick_col(
        x,
        ["benchmark_ret", "bench_ret"],
        "intraday benchmark return",
        required=False,
    )
    if bench_signal_col is None:
        # fallback：如果没有原始 intraday benchmark，就用 with_overnight，但第一根置 0
        bench_signal_col = bench_eval_col

    x["bench_signal_ret"] = pd.to_numeric(x[bench_signal_col], errors="coerce").fillna(0.0)
    x.loc[x["bar_in_day"] == 0, "bench_signal_ret"] = 0.0

    x["bench_cum_intraday"] = (
        x.groupby("date")["bench_signal_ret"]
        .transform(lambda s: (1.0 + s).cumprod() - 1.0)
    )

    # 用上一分钟信号，避免同一根 lookahead
    x["bench_cum_signal_lag"] = x.groupby("date")["bench_cum_intraday"].shift(1).fillna(0.0)

    x["up_regime"] = (
        (x["bench_cum_signal_lag"] > float(args.threshold))
        & (x["bar_in_day"] >= int(args.min_bars))
    )

    x["stock_gross"] = pd.to_numeric(x["stock_gross"], errors="coerce").fillna(0.0)
    x["futures_ret"] = pd.to_numeric(x["futures_ret"], errors="coerce").fillna(0.0)

    fill = (float(args.target_total_exposure) - x["stock_gross"]).clip(
        lower=0.0,
        upper=float(args.max_overlay_gross),
    )

    x["target_futures_overlay_gross"] = np.where(x["up_regime"], fill, 0.0)

    # 不持 futures 过夜
    x.loc[x["bar_in_day"] == 0, "target_futures_overlay_gross"] = 0.0

    # 当前分钟收益由上一分钟持仓产生
    x["futures_overlay_prev_gross"] = (
        x.groupby("date")["target_futures_overlay_gross"]
        .shift(1)
        .fillna(0.0)
    )

    x["overlay_turnover"] = (
        x.groupby("date")["target_futures_overlay_gross"]
        .diff()
        .abs()
        .fillna(x["target_futures_overlay_gross"].abs())
    )
    x.loc[x["bar_in_day"] == 0, "overlay_turnover"] = 0.0

    x["futures_fee_ret"] = x["overlay_turnover"] * float(args.futures_fee_rate)

    x["actual_ret_v21"] = (
        pd.to_numeric(x[actual_base_col], errors="coerce").fillna(0.0)
        + x["futures_overlay_prev_gross"] * x["futures_ret"]
        - x["futures_fee_ret"]
    )

    x["benchmark_ret_v21"] = pd.to_numeric(x[bench_eval_col], errors="coerce").fillna(0.0)

    x["actualret_v21"] = (1.0 + x["actual_ret_v21"]).cumprod() - 1.0
    x["benchmarkret_v21"] = (1.0 + x["benchmark_ret_v21"]).cumprod() - 1.0
    x["alpharet_v21"] = x["actualret_v21"] - x["benchmarkret_v21"]

    summary = pd.DataFrame([{
        "tag": args.tag,
        "actual_return": float(x["actualret_v21"].iloc[-1]),
        "benchmark_return": float(x["benchmarkret_v21"].iloc[-1]),
        "alpha_return": float(x["alpharet_v21"].iloc[-1]),
        "daily_excess_sharpe": daily_sharpe(x, "actual_ret_v21", "benchmark_ret_v21"),
        "avg_stock_gross": float(x["stock_gross"].mean()),
        "avg_futures_overlay_gross": float(x["target_futures_overlay_gross"].mean()),
        "avg_total_exposure": float((x["stock_gross"] + x["target_futures_overlay_gross"]).mean()),
        "avg_up_regime_ratio": float(x["up_regime"].mean()),
        "avg_overlay_turnover": float(x["overlay_turnover"].mean()),
        "total_futures_fee_ret": float(x["futures_fee_ret"].sum()),
        "threshold": float(args.threshold),
        "min_bars": int(args.min_bars),
        "target_total_exposure": float(args.target_total_exposure),
        "max_overlay_gross": float(args.max_overlay_gross),
        "futures_fee_rate": float(args.futures_fee_rate),
    }])

    daily = make_daily_table(x, "actual_ret_v21", "benchmark_ret_v21")

    curve_out = out / f"{args.tag}_curve.csv"
    daily_out = out / f"{args.tag}_daily.csv"
    summary_out = out / f"{args.tag}_summary.csv"
    png_out = out / f"{args.tag}_nav.png"

    x.to_csv(curve_out, index=False)
    daily.to_csv(daily_out, index=False)
    summary.to_csv(summary_out, index=False)

    x["plot_index"] = np.arange(len(x))
    first_idx = x.groupby("date")["plot_index"].first()
    tick_dates = list(first_idx.index)[::2]
    tick_pos = [int(first_idx.loc[d]) for d in tick_dates]

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(x["plot_index"], x["actualret_v21"] * 100.0, label="strategy actualret")
    ax.plot(x["plot_index"], x["benchmarkret_v21"] * 100.0, label="CSI2000 benchmarkret")
    ax.plot(x["plot_index"], x["alpharet_v21"] * 100.0, label="alpharet")
    ax.axhline(0, linewidth=1, linestyle="--")
    ax.set_title("V21 Regime-Gated Intraday Futures Beta Fill")
    ax.set_xlabel("trading minute index")
    ax.set_ylabel("cumulative return (%)")
    ax.set_xticks(tick_pos)
    ax.set_xticklabels([str(d) for d in tick_dates], rotation=45, ha="right")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left")

    s = summary.iloc[0]
    txt = (
        "Summary\n\n"
        f"Strategy Return\n{s['actual_return']*100:.2f}%\n\n"
        f"Benchmark Return\n{s['benchmark_return']*100:.2f}%\n\n"
        f"Alpha Return\n{s['alpha_return']*100:.2f}%\n\n"
        f"Daily Excess Sharpe\n{s['daily_excess_sharpe']:.2f}"
    )
    ax.text(
        1.05, 0.95, txt,
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )

    fig.tight_layout()
    fig.savefig(png_out, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(summary.T.to_string())
    print("curve:", curve_out)
    print("daily:", daily_out)
    print("summary:", summary_out)
    print("png:", png_out)


if __name__ == "__main__":
    main()
