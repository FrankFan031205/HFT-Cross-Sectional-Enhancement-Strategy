# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def pick_col(cols, candidates):
    lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None

def load_table(path: Path):
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="curve csv/parquet path")
    ap.add_argument("--output", required=True, help="output png path")
    ap.add_argument("--title", default="Time-sliced Sleeve Pure-CS, Raw Gradual Build")
    args = ap.parse_args()

    path = Path(args.input)
    df = load_table(path)

    cols = list(df.columns)

    time_col = pick_col(cols, [
        "datetime", "ts", "timestamp", "time",
        "trading_minute_index", "minute_index", "date"
    ])

    strat_col = pick_col(cols, [
        "strategy_ret", "actual_ret", "sleeve_actualret", "actualret",
        "portfolio_ret", "ret"
    ])
    bench_col = pick_col(cols, [
        "benchmark_ret", "benchmarkret", "sleeve_benchmarkret", "bench_ret"
    ])
    alpha_col = pick_col(cols, [
        "alpha_ret", "alpharet", "sleeve_alpharet", "excess_ret", "excessret"
    ])

    if strat_col is None or bench_col is None:
        raise RuntimeError(f"Cannot detect strategy/benchmark return columns. columns={cols}")

    x = np.arange(len(df)) if time_col is None else df[time_col]

    strat = pd.to_numeric(df[strat_col], errors="coerce").fillna(0.0).to_numpy()
    bench = pd.to_numeric(df[bench_col], errors="coerce").fillna(0.0).to_numpy()

    if alpha_col is not None:
        alpha = pd.to_numeric(df[alpha_col], errors="coerce").fillna(0.0).to_numpy()
    else:
        alpha = strat - bench

    # 如果是逐期收益（小数），累计求和并转成百分比
    strat_cum = np.cumsum(strat) * 100.0
    bench_cum = np.cumsum(bench) * 100.0
    alpha_cum = np.cumsum(alpha) * 100.0

    strategy_return = strat_cum[-1]
    benchmark_return = bench_cum[-1]
    alpha_return = alpha_cum[-1]

    # 粗略算一个 daily excess sharpe（如果没有日期就退化为全样本）
    date_col = pick_col(cols, ["date"])
    if date_col is not None:
        tmp = pd.DataFrame({
            "date": df[date_col],
            "alpha": alpha
        })
        daily = tmp.groupby("date", sort=True)["alpha"].sum()
        if len(daily) > 1 and daily.std() > 1e-12:
            daily_excess_sharpe = daily.mean() / daily.std() * np.sqrt(252)
        else:
            daily_excess_sharpe = np.nan
    else:
        if len(alpha) > 1 and np.std(alpha) > 1e-12:
            daily_excess_sharpe = np.mean(alpha) / np.std(alpha) * np.sqrt(252)
        else:
            daily_excess_sharpe = np.nan

    fig = plt.figure(figsize=(16, 7))
    ax = fig.add_axes([0.07, 0.12, 0.67, 0.79])

    ax.plot(x, strat_cum, label="sleeve actualret")
    ax.plot(x, bench_cum, label="benchmarkret")
    ax.plot(x, alpha_cum, label="sleeve alpharet")
    ax.axhline(0, linestyle="--", linewidth=1)

    ax.set_title(args.title)
    ax.set_ylabel("cumulative return (%)")
    ax.set_xlabel("trading minute index" if time_col is None else time_col)
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)

    summary_text = (
        "Summary\n\n"
        f"Strategy Return\n{strategy_return:.2f}%\n\n"
        f"Benchmark Return\n{benchmark_return:.2f}%\n\n"
        f"Alpha Return\n{alpha_return:.2f}%\n\n"
        f"Daily Excess Sharpe\n"
        f"{daily_excess_sharpe:.2f}" if np.isfinite(daily_excess_sharpe) else "NA"
    )

    fig.text(
        0.79, 0.83, summary_text,
        ha="left", va="top", fontsize=14,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.85)
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[OK] plot saved to {out}")

if __name__ == "__main__":
    main()
