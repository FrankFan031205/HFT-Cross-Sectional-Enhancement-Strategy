# -*- coding: utf-8 -*-
"""
Presentation-style plot for Pure-CS NAV with CSI1000 futures overlay.

Input:
  curve_with_real_csi1000_futures_overlay.csv
  summary_with_real_csi1000_futures_overlay.csv

Output:
  <prefix>_main_curve_presentation.png
  <prefix>_alpha_compare_presentation.png
  <prefix>_chart_summary.csv
  <prefix>_curve_for_presentation.csv

This script avoids Chinese labels to prevent font/encoding issues in matplotlib.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def compound_curve(x):
    x = pd.Series(x).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return (1.0 + x).cumprod() - 1.0


def pick(row, keys, default=np.nan):
    for k in keys:
        if k in row and pd.notna(row[k]):
            return float(row[k])
    return default


def pct(x):
    if pd.isna(x):
        return "NA"
    return f"{x * 100:.2f}%"


def num(x):
    if pd.isna(x):
        return "NA"
    return f"{x:.2f}"


def load_curve(path):
    df = pd.read_csv(path, low_memory=False)
    df["datetime"] = pd.to_datetime(df["datetime"])

    if "date" not in df.columns:
        df["date"] = df["datetime"].dt.strftime("%Y%m%d").astype(int)

    df = df.sort_values(["date", "datetime"]).reset_index(drop=True)
    df["bar_index"] = np.arange(len(df))

    if "actualret_with_futures" not in df.columns:
        if "actual_ret_with_futures" in df.columns:
            df["actualret_with_futures"] = compound_curve(df["actual_ret_with_futures"])
        else:
            raise KeyError("curve csv needs actualret_with_futures or actual_ret_with_futures")

    if "benchmarkret_full" not in df.columns:
        if "benchmark_ret" in df.columns:
            df["benchmarkret_full"] = compound_curve(df["benchmark_ret"])
        else:
            raise KeyError("curve csv needs benchmarkret_full or benchmark_ret")

    if "alpharet_with_futures" not in df.columns:
        df["alpharet_with_futures"] = df["actualret_with_futures"] - df["benchmarkret_full"]

    if "alpharet_no_futures" not in df.columns:
        if "actualret_no_futures" in df.columns and "benchmarkret_no_futures" in df.columns:
            df["alpharet_no_futures"] = df["actualret_no_futures"] - df["benchmarkret_no_futures"]
        elif "alpharet" in df.columns:
            df["alpharet_no_futures"] = df["alpharet"]
        else:
            df["alpharet_no_futures"] = np.nan

    return df


def load_stats(summary_csv, df):
    s = pd.read_csv(summary_csv).iloc[0].to_dict()

    strategy_ret = pick(
        s,
        ["final_actualret_with_futures", "actualret_with_futures", "actual_return_with_futures"],
        df["actualret_with_futures"].iloc[-1],
    )

    benchmark_ret = pick(
        s,
        ["final_benchmarkret_full", "benchmarkret_full", "benchmark_return"],
        df["benchmarkret_full"].iloc[-1],
    )

    alpha_ret = pick(
        s,
        ["final_alpharet_with_futures", "alpharet_with_futures", "excess_return_with_futures"],
        df["alpharet_with_futures"].iloc[-1],
    )

    sharpe = pick(
        s,
        ["daily_excess_sharpe_with_futures", "daily_excess_sharpe"],
        np.nan,
    )

    return {
        "strategy_ret": strategy_ret,
        "benchmark_ret": benchmark_ret,
        "alpha_ret": alpha_ret,
        "sharpe": sharpe,
    }


def make_date_ticks(df):
    first = df.groupby("date", as_index=False)["bar_index"].min()
    n = len(first)

    if n <= 12:
        step = 1
    elif n <= 24:
        step = 2
    else:
        step = max(1, n // 12)

    ticks = first.iloc[::step].copy()
    return ticks["bar_index"].tolist(), ticks["date"].astype(str).tolist()


def draw_summary_card(ax, stats):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    card = patches.FancyBboxPatch(
        (0.08, 0.08),
        0.84,
        0.84,
        boxstyle="round,pad=0.035,rounding_size=0.035",
        linewidth=1.2,
        edgecolor="#D0D0D0",
        facecolor="white",
        alpha=0.98,
    )
    ax.add_patch(card)

    ax.text(
        0.18,
        0.86,
        "Summary",
        fontsize=22,
        fontweight="bold",
        ha="left",
        va="center",
        color="#111111",
    )

    metrics = [
        ("Strategy Return", pct(stats["strategy_ret"]), "#1f77b4"),
        ("Benchmark Return", pct(stats["benchmark_ret"]), "#ff7f0e"),
        ("Alpha Return", pct(stats["alpha_ret"]), "#2ca02c"),
        ("Daily Excess Sharpe", num(stats["sharpe"]), "#222222"),
    ]

    y = 0.70
    for label, value, color in metrics:
        ax.plot([0.18, 0.82], [y + 0.07, y + 0.07], color="#D8E6F3", linewidth=1.2)
        ax.text(
            0.18,
            y,
            label,
            fontsize=14,
            ha="left",
            va="center",
            color="#111111",
        )
        ax.text(
            0.18,
            y - 0.095,
            value,
            fontsize=24,
            fontweight="bold",
            ha="left",
            va="center",
            color=color,
        )
        y -= 0.205


def style_main_axis(ax, df):
    tick_pos, tick_lab = make_date_ticks(df)

    ax.axhline(0.0, linewidth=1.0, color="#B0B0B0", linestyle="--", alpha=0.75)
    ax.grid(True, color="#BEBEBE", alpha=0.28, linestyle="--", linewidth=0.8)

    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lab, rotation=45, ha="right", fontsize=11)

    ax.tick_params(axis="y", labelsize=12)
    ax.set_xlabel("Date", fontsize=15, labelpad=12)
    ax.set_ylabel("Cumulative return (%)", fontsize=16, labelpad=12)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")


def plot_main(df, stats, out_dir, prefix, fig_width, fig_height, dpi):
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = fig.add_gridspec(
        1,
        2,
        width_ratios=[5.2, 1.45],
        wspace=0.08,
        left=0.055,
        right=0.975,
        top=0.88,
        bottom=0.15,
    )

    ax = fig.add_subplot(gs[0, 0])
    ax_card = fig.add_subplot(gs[0, 1])

    ax.plot(
        df["bar_index"],
        df["actualret_with_futures"] * 100.0,
        label="Strategy NAV",
        linewidth=1.8,
        color="#1f77b4",
    )
    ax.plot(
        df["bar_index"],
        df["benchmarkret_full"] * 100.0,
        label="Full Benchmark",
        linewidth=1.8,
        color="#ff7f0e",
    )
    ax.plot(
        df["bar_index"],
        df["alpharet_with_futures"] * 100.0,
        label="Alpha",
        linewidth=2.2,
        color="#2ca02c",
    )

    style_main_axis(ax, df)

    ax.legend(
        loc="lower left",
        fontsize=13,
        frameon=True,
        framealpha=0.92,
        facecolor="white",
        edgecolor="#BBBBBB",
    )

    fig.suptitle(
        "Pure-CS NAV with CSI1000 Futures Overlay",
        fontsize=24,
        fontweight="bold",
        y=0.965,
    )

    draw_summary_card(ax_card, stats)

    out_png = out_dir / f"{prefix}_main_curve_presentation.png"
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    print("[saved]", out_png)


def plot_alpha_compare(df, stats, out_dir, prefix, fig_width, fig_height, dpi):
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = fig.add_gridspec(
        1,
        2,
        width_ratios=[5.2, 1.45],
        wspace=0.08,
        left=0.055,
        right=0.975,
        top=0.88,
        bottom=0.15,
    )

    ax = fig.add_subplot(gs[0, 0])
    ax_card = fig.add_subplot(gs[0, 1])

    if df["alpharet_no_futures"].notna().any():
        ax.plot(
            df["bar_index"],
            df["alpharet_no_futures"] * 100.0,
            label="Alpha without Futures",
            linewidth=1.9,
            color="#9467bd",
        )

    ax.plot(
        df["bar_index"],
        df["alpharet_with_futures"] * 100.0,
        label="Alpha with CSI1000 Futures",
        linewidth=2.2,
        color="#2ca02c",
    )

    style_main_axis(ax, df)
    ax.set_ylabel("Cumulative alpha (%)", fontsize=16, labelpad=12)

    ax.legend(
        loc="lower left",
        fontsize=13,
        frameon=True,
        framealpha=0.92,
        facecolor="white",
        edgecolor="#BBBBBB",
    )

    fig.suptitle(
        "Alpha Before and After CSI1000 Futures Overlay",
        fontsize=24,
        fontweight="bold",
        y=0.965,
    )

    draw_summary_card(ax_card, stats)

    out_png = out_dir / f"{prefix}_alpha_compare_presentation.png"
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    print("[saved]", out_png)


def save_summary(df, stats, out_dir, prefix):
    row = {
        "strategy_return": stats["strategy_ret"],
        "benchmark_return": stats["benchmark_ret"],
        "alpha_return": stats["alpha_ret"],
        "daily_excess_sharpe": stats["sharpe"],
        "final_alpha_without_futures": df["alpharet_no_futures"].iloc[-1]
        if df["alpharet_no_futures"].notna().any()
        else np.nan,
        "n_minutes": len(df),
    }
    out_csv = out_dir / f"{prefix}_chart_summary.csv"
    pd.DataFrame([row]).to_csv(out_csv, index=False, encoding="utf-8-sig")
    print("[saved]", out_csv)
    print(pd.DataFrame([row]).T.to_string(header=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curve-csv", required=True)
    ap.add_argument("--summary-csv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--prefix", default="futures_overlay")
    ap.add_argument("--fig-width", type=float, default=22.0)
    ap.add_argument("--fig-height", type=float, default=11.5)
    ap.add_argument("--dpi", type=int, default=220)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_curve(args.curve_csv)
    stats = load_stats(args.summary_csv, df)

    plot_main(df, stats, out_dir, args.prefix, args.fig_width, args.fig_height, args.dpi)
    plot_alpha_compare(df, stats, out_dir, args.prefix, args.fig_width, args.fig_height * 0.90, args.dpi)

    curve_out = out_dir / f"{args.prefix}_curve_for_presentation.csv"
    df.to_csv(curve_out, index=False, encoding="utf-8-sig")
    print("[saved]", curve_out)

    save_summary(df, stats, out_dir, args.prefix)


if __name__ == "__main__":
    main()
