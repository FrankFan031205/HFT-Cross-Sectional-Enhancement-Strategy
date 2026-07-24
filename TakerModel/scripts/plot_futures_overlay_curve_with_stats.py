# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compound_curve(x):
    x = pd.Series(x).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return (1.0 + x).cumprod() - 1.0


def pick_first(row, candidates, default=np.nan):
    for c in candidates:
        if c in row.index and pd.notna(row[c]):
            return row[c]
    return default


def pct(x, nd=2):
    if pd.isna(x):
        return "NA"
    return f"{float(x) * 100:.{nd}f}%"


def num(x, nd=2):
    if pd.isna(x):
        return "NA"
    return f"{float(x):.{nd}f}"


def make_date_ticks(df):
    first = df.groupby("date", as_index=False)["bar_index"].min()
    n = len(first)

    if n <= 12:
        step = 1
    elif n <= 24:
        step = 2
    else:
        step = max(1, n // 12)

    ticks = first.iloc[::step]
    return ticks["bar_index"].tolist(), ticks["date"].astype(str).tolist()



def build_stats_text(summary_df):
    row = summary_df.iloc[0].to_dict()

    def pick(keys, default=None):
        for k in keys:
            if k in row and row[k] is not None:
                return row[k]
        return default

    strategy_ret = pick([
        "final_actualret_with_futures",
        "summary_actual_return",
        "actual_return",
        "nav_actualret",
    ], 0.0)

    benchmark_ret = pick([
        "final_benchmarkret_full",
        "summary_benchmark_return",
        "benchmark_return",
        "nav_benchmarkret",
    ], 0.0)

    alpha_ret = pick([
        "final_alpharet_with_futures",
        "summary_excess_vs_full_benchmark",
        "actual_excess_vs_full_benchmark",
        "nav_alpharet",
    ], 0.0)

    sharpe = pick([
        "daily_excess_sharpe_with_futures",
        "daily_excess_sharpe",
    ], 0.0)

    lines = [
        f"Strategy Return: {strategy_ret:.2%}",
        f"Benchmark Return: {benchmark_ret:.2%}",
        f"Alpha Return: {alpha_ret:.2%}",
        f"Daily Excess Sharpe: {sharpe:.2f}",
    ]
    return "\n".join(lines)

def normalize_curve(curve_csv):
    df = pd.read_csv(curve_csv, low_memory=False)
    df["datetime"] = pd.to_datetime(df["datetime"])

    if "date" not in df.columns:
        df["date"] = df["datetime"].dt.strftime("%Y%m%d").astype(int)

    df = df.sort_values(["date", "datetime"]).reset_index(drop=True)

    if "actualret_with_futures" not in df.columns:
        if "actual_ret_with_futures" in df.columns:
            df["actualret_with_futures"] = compound_curve(df["actual_ret_with_futures"])
        else:
            raise KeyError("curve csv must contain actualret_with_futures or actual_ret_with_futures")

    if "benchmarkret_full" not in df.columns:
        if "benchmark_ret" in df.columns:
            df["benchmarkret_full"] = compound_curve(df["benchmark_ret"])
        else:
            raise KeyError("curve csv must contain benchmarkret_full or benchmark_ret")

    if "alpharet_with_futures" not in df.columns:
        df["alpharet_with_futures"] = df["actualret_with_futures"] - df["benchmarkret_full"]

    if "alpharet_no_futures" not in df.columns:
        if "actualret_no_futures" in df.columns and "benchmarkret_no_futures" in df.columns:
            df["alpharet_no_futures"] = df["actualret_no_futures"] - df["benchmarkret_no_futures"]
        elif "alpharet" in df.columns:
            df["alpharet_no_futures"] = df["alpharet"]
        else:
            df["alpharet_no_futures"] = np.nan

    df["bar_index"] = np.arange(len(df))

    return df


def plot_main_curve(df, stats_text, out_dir, prefix):
    tick_pos, tick_lab = make_date_ticks(df)

    fig, ax = plt.subplots(figsize=(15, 7))

    ax.plot(
        df["bar_index"],
        df["actualret_with_futures"] * 100.0,
        label="actualret with CSI1000 futures",
        linewidth=1.4,
    )
    ax.plot(
        df["bar_index"],
        df["benchmarkret_full"] * 100.0,
        label="full benchmarkret",
        linewidth=1.4,
    )
    ax.plot(
        df["bar_index"],
        df["alpharet_with_futures"] * 100.0,
        label="alpharet with futures",
        linewidth=1.6,
    )

    ax.axhline(0.0, linewidth=0.8, color="gray", linestyle="--", alpha=0.6)
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lab, rotation=45)
    ax.set_title("Pure-CS NAV with Real CSI1000 Futures Overlay")
    ax.set_xlabel("trading minute index, overnight/weekend gaps removed")
    ax.set_ylabel("cumulative return (%)")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)

    ax.text(
        0.98,
        0.98,
        stats_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(
            boxstyle="round,pad=0.45",
            facecolor="white",
            edgecolor="gray",
            alpha=0.88,
        ),
    )

    fig.tight_layout()

    out_png = out_dir / f"{prefix}_curve_with_stats.png"
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

    print("[saved]", out_png)


def plot_alpha_compare(df, stats_text, out_dir, prefix):
    tick_pos, tick_lab = make_date_ticks(df)

    fig, ax = plt.subplots(figsize=(15, 6))

    if "alpharet_no_futures" in df.columns and df["alpharet_no_futures"].notna().any():
        ax.plot(
            df["bar_index"],
            df["alpharet_no_futures"] * 100.0,
            label="alpha without futures",
            linewidth=1.4,
        )

    ax.plot(
        df["bar_index"],
        df["alpharet_with_futures"] * 100.0,
        label="alpha with CSI1000 futures",
        linewidth=1.6,
    )

    ax.axhline(0.0, linewidth=0.8, color="gray", linestyle="--", alpha=0.6)
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lab, rotation=45)
    ax.set_title("Alpha Before and After CSI1000 Futures Overlay")
    ax.set_xlabel("trading minute index, overnight/weekend gaps removed")
    ax.set_ylabel("cumulative alpha (%)")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)

    ax.text(
        0.98,
        0.98,
        stats_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(
            boxstyle="round,pad=0.45",
            facecolor="white",
            edgecolor="gray",
            alpha=0.88,
        ),
    )

    fig.tight_layout()

    out_png = out_dir / f"{prefix}_alpha_compare_with_stats.png"
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

    print("[saved]", out_png)


def save_final_table(df, summary_csv, out_dir, prefix):
    s = pd.read_csv(summary_csv).iloc[0].to_dict()
    last = df.iloc[-1].to_dict()

    row = {
        "final_actualret_with_futures": last.get("actualret_with_futures", np.nan),
        "final_benchmarkret_full": last.get("benchmarkret_full", np.nan),
        "final_alpharet_with_futures": last.get("alpharet_with_futures", np.nan),
        "final_alpharet_no_futures": last.get("alpharet_no_futures", np.nan),
        "avg_stock_gross": s.get("avg_stock_gross", np.nan),
        "avg_futures_notional": s.get("avg_futures_notional", np.nan),
        "avg_margin_used": s.get("avg_margin_used", np.nan),
        "total_futures_turnover": s.get("total_futures_turnover", np.nan),
        "total_futures_fee_return": s.get("total_futures_fee_return", np.nan),
        "daily_excess_sharpe_with_futures": s.get("daily_excess_sharpe_with_futures", np.nan),
        "daily_excess_tstat_with_futures": s.get("daily_excess_tstat_with_futures", np.nan),
        "n_minutes": len(df),
    }

    out = pd.DataFrame([row])
    out_csv = out_dir / f"{prefix}_summary_for_chart.csv"
    out.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print("[saved]", out_csv)
    print(out.T.to_string(header=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curve-csv", required=True)
    ap.add_argument("--summary-csv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--prefix", default="csi1000_futures_overlay")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = normalize_curve(args.curve_csv)
    stats_text = build_stats_text(args.summary_csv)

    plot_main_curve(df, stats_text, out_dir, args.prefix)
    plot_alpha_compare(df, stats_text, out_dir, args.prefix)

    out_curve = out_dir / f"{args.prefix}_curve_for_chart.csv"
    df.to_csv(out_curve, index=False, encoding="utf-8-sig")
    print("[saved]", out_curve)

    save_final_table(df, args.summary_csv, out_dir, args.prefix)


if __name__ == "__main__":
    main()
