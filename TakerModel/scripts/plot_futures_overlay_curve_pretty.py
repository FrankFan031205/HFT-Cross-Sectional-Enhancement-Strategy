# -*- coding: utf-8 -*-
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compound_curve(x):
    x = pd.Series(x).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return (1.0 + x).cumprod() - 1.0


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


def pick_value(row, keys, default=np.nan):
    for k in keys:
        if k in row and pd.notna(row[k]):
            return row[k]
    return default


def pct(x):
    if pd.isna(x):
        return "NA"
    return f"{float(x) * 100:.2f}%"


def num(x):
    if pd.isna(x):
        return "NA"
    return f"{float(x):.2f}"


def load_curve(curve_csv):
    df = pd.read_csv(curve_csv, low_memory=False)
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

    strategy_ret = pick_value(
        s,
        ["final_actualret_with_futures", "actualret_with_futures", "actual_return_with_futures"],
        df["actualret_with_futures"].iloc[-1],
    )

    benchmark_ret = pick_value(
        s,
        ["final_benchmarkret_full", "benchmarkret_full", "benchmark_return"],
        df["benchmarkret_full"].iloc[-1],
    )

    alpha_ret = pick_value(
        s,
        ["final_alpharet_with_futures", "alpharet_with_futures", "excess_return_with_futures"],
        df["alpharet_with_futures"].iloc[-1],
    )

    sharpe = pick_value(
        s,
        ["daily_excess_sharpe_with_futures", "daily_excess_sharpe"],
        np.nan,
    )

    return {
        "Strategy Return": pct(strategy_ret),
        "Benchmark Return": pct(benchmark_ret),
        "Alpha Return": pct(alpha_ret),
        "Daily Excess Sharpe": num(sharpe),
    }


def draw_stats_panel(ax, stats):
    ax.axis("off")

    title = "Summary"
    lines = [title, ""]
    for k, v in stats.items():
        lines.append(f"{k}")
        lines.append(f"  {v}")
        lines.append("")

    text = "\n".join(lines).strip()

    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        linespacing=1.35,
        bbox=dict(
            boxstyle="round,pad=0.55",
            facecolor="white",
            edgecolor="gray",
            alpha=0.95,
        ),
    )


def plot_main(df, stats, out_dir, prefix):
    tick_pos, tick_lab = make_date_ticks(df)

    fig = plt.figure(figsize=(16, 7))
    gs = fig.add_gridspec(1, 2, width_ratios=[4.7, 1.35], wspace=0.08)

    ax = fig.add_subplot(gs[0, 0])
    ax_stats = fig.add_subplot(gs[0, 1])

    ax.plot(
        df["bar_index"],
        df["actualret_with_futures"] * 100.0,
        label="Strategy NAV",
        linewidth=1.5,
    )
    ax.plot(
        df["bar_index"],
        df["benchmarkret_full"] * 100.0,
        label="Full Benchmark",
        linewidth=1.5,
    )
    ax.plot(
        df["bar_index"],
        df["alpharet_with_futures"] * 100.0,
        label="Alpha",
        linewidth=1.8,
    )

    ax.axhline(0.0, linewidth=0.8, color="gray", linestyle="--", alpha=0.65)

    ax.set_title("Pure-CS NAV with CSI1000 Futures Overlay", fontsize=14)
    ax.set_xlabel("Trading minute index, overnight/weekend gaps removed")
    ax.set_ylabel("Cumulative return (%)")

    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lab, rotation=45, ha="right")

    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower left", frameon=True)

    draw_stats_panel(ax_stats, stats)

    fig.tight_layout()

    out_png = out_dir / f"{prefix}_main_curve_pretty.png"
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print("[saved]", out_png)


def plot_alpha(df, stats, out_dir, prefix):
    tick_pos, tick_lab = make_date_ticks(df)

    fig = plt.figure(figsize=(16, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[4.7, 1.35], wspace=0.08)

    ax = fig.add_subplot(gs[0, 0])
    ax_stats = fig.add_subplot(gs[0, 1])

    if df["alpharet_no_futures"].notna().any():
        ax.plot(
            df["bar_index"],
            df["alpharet_no_futures"] * 100.0,
            label="Alpha without Futures",
            linewidth=1.4,
        )

    ax.plot(
        df["bar_index"],
        df["alpharet_with_futures"] * 100.0,
        label="Alpha with CSI1000 Futures",
        linewidth=1.8,
    )

    ax.axhline(0.0, linewidth=0.8, color="gray", linestyle="--", alpha=0.65)

    ax.set_title("Alpha Before and After Futures Overlay", fontsize=14)
    ax.set_xlabel("Trading minute index, overnight/weekend gaps removed")
    ax.set_ylabel("Cumulative alpha (%)")

    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lab, rotation=45, ha="right")

    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower left", frameon=True)

    draw_stats_panel(ax_stats, stats)

    fig.tight_layout()

    out_png = out_dir / f"{prefix}_alpha_curve_pretty.png"
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print("[saved]", out_png)


def save_compact_summary(df, stats, out_dir, prefix):
    out = {
        "final_strategy_return": df["actualret_with_futures"].iloc[-1],
        "final_benchmark_return": df["benchmarkret_full"].iloc[-1],
        "final_alpha_return": df["alpharet_with_futures"].iloc[-1],
        "final_alpha_without_futures": df["alpharet_no_futures"].iloc[-1]
        if df["alpharet_no_futures"].notna().any()
        else np.nan,
        "n_minutes": len(df),
    }

    for k, v in stats.items():
        out[k] = v

    out_df = pd.DataFrame([out])
    out_csv = out_dir / f"{prefix}_compact_summary.csv"
    out_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print("[saved]", out_csv)
    print(out_df.T.to_string(header=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curve-csv", required=True)
    ap.add_argument("--summary-csv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--prefix", default="futures_overlay")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_curve(args.curve_csv)
    stats = load_stats(args.summary_csv, df)

    plot_main(df, stats, out_dir, args.prefix)
    plot_alpha(df, stats, out_dir, args.prefix)

    curve_out = out_dir / f"{args.prefix}_curve_for_plot.csv"
    df.to_csv(curve_out, index=False, encoding="utf-8-sig")
    print("[saved]", curve_out)

    save_compact_summary(df, stats, out_dir, args.prefix)


if __name__ == "__main__":
    main()
