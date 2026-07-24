# -*- coding: utf-8 -*-
from pathlib import Path
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE = Path("/mnt/data1/fwz/HFT_010-dev_fwz_outputs/final_report")
V10_ROOT = BASE / "v10_many_horizon_mix_csi2000_warmstart_noovernight"
OUT_DIR = BASE / "v10_report_style_png"
OUT_DIR.mkdir(parents=True, exist_ok=True)


CASES = [
    {
        "name": "final_h10h20_4060",
        "mix": "mix_406000",
        "title": "T+1-aware Pure-CS NAV with CSI2000 Benchmark",
        "display": "Final: 40% h10 + 60% h20",
    },
    {
        "name": "stable_h20h30_7030",
        "mix": "mix_007030",
        "title": "T+1-aware Pure-CS NAV with CSI2000 Benchmark",
        "display": "Stable: 70% h20 + 30% h30",
    },
    {
        "name": "light_h10h20_1090",
        "mix": "mix_109000",
        "title": "T+1-aware Pure-CS NAV with CSI2000 Benchmark",
        "display": "Light: 10% h10 + 90% h20",
    },
]


def compound(ret):
    ret = pd.Series(ret).fillna(0.0)
    return float((1.0 + ret).prod() - 1.0)


def annualized_daily_sharpe(daily_alpha):
    daily_alpha = pd.Series(daily_alpha).dropna()
    if len(daily_alpha) <= 1 or daily_alpha.std(ddof=1) == 0:
        return np.nan
    return float(daily_alpha.mean() / daily_alpha.std(ddof=1) * np.sqrt(252))


def load_nav(mix):
    strategy = f"pure_cs_v10_{mix}_csi2000_warmstart_noovernight"
    p = V10_ROOT / strategy / f"{strategy}_nav_curve_benchmark_warmstart_noovernight.csv"
    if not p.exists():
        raise FileNotFoundError(p)

    df = pd.read_csv(p)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    if "actual_ret" not in df.columns:
        raise KeyError(f"missing actual_ret in {p}")
    if "benchmark_ret" not in df.columns:
        raise KeyError(f"missing benchmark_ret in {p}")

    df["actual_cum"] = (1.0 + df["actual_ret"].fillna(0.0)).cumprod() - 1.0
    df["benchmark_cum"] = (1.0 + df["benchmark_ret"].fillna(0.0)).cumprod() - 1.0
    df["alpha_cum"] = df["actual_cum"] - df["benchmark_cum"]

    return df, p


def make_plot(case):
    df, src = load_nav(case["mix"])

    strategy_return = compound(df["actual_ret"])
    benchmark_return = compound(df["benchmark_ret"])
    alpha_return = strategy_return - benchmark_return

    daily = (
        df.groupby("date", as_index=False)
          .agg(
              actual_day=("actual_ret", compound),
              benchmark_day=("benchmark_ret", compound),
          )
    )
    daily["alpha_day"] = daily["actual_day"] - daily["benchmark_day"]
    daily_excess_sharpe = annualized_daily_sharpe(daily["alpha_day"])

    fig, ax = plt.subplots(figsize=(14, 8))

    ax.plot(
        df["datetime"],
        df["actual_cum"] * 100.0,
        label="strategyret",
        linewidth=1.6,
    )
    ax.plot(
        df["datetime"],
        df["benchmark_cum"] * 100.0,
        label="benchmarkret",
        linewidth=1.6,
    )
    ax.plot(
        df["datetime"],
        df["alpha_cum"] * 100.0,
        label="alpharet",
        linewidth=1.8,
    )

    ax.axhline(0.0, linestyle="--", linewidth=1.0)

    ax.set_title(case["title"], fontsize=16, weight="bold")
    ax.set_ylabel("cumulative return (%)")
    ax.set_xlabel("trading minute index, benchmark warm-start, no overnight PnL")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.25)

    summary_text = (
        "Summary\n\n"
        f"Strategy\n{case['display']}\n\n"
        f"Strategy Return\n{strategy_return * 100.0:.2f}%\n\n"
        f"Benchmark Return\n{benchmark_return * 100.0:.2f}%\n\n"
        f"Alpha Return\n{alpha_return * 100.0:.2f}%\n\n"
        f"Daily Excess Sharpe\n{daily_excess_sharpe:.2f}"
    )

    ax.text(
        1.035,
        0.95,
        summary_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.7", alpha=0.95),
    )

    fig.autofmt_xdate(rotation=45)
    plt.subplots_adjust(right=0.76, bottom=0.14)

    out = OUT_DIR / f"{case['name']}_nav_report_style.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)

    print("[saved]", out)
    print("[source]", src)
    print(
        f"{case['name']}: "
        f"strategy={strategy_return:.6f}, "
        f"benchmark={benchmark_return:.6f}, "
        f"alpha={alpha_return:.6f}, "
        f"sharpe={daily_excess_sharpe:.6f}"
    )


for case in CASES:
    make_plot(case)

print("\n===== DONE =====")
print(OUT_DIR)
