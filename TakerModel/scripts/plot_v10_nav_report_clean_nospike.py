# -*- coding: utf-8 -*-
from pathlib import Path
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE = Path("/mnt/data1/fwz/HFT_010-dev_fwz_outputs/final_report")
V10_ROOT = BASE / "v10_many_horizon_mix_csi2000_warmstart_noovernight"
OUT_DIR = BASE / "v10_report_style_png_clean"
OUT_DIR.mkdir(parents=True, exist_ok=True)


CASES = [
    {
        "name": "final_h10h20_4060",
        "mix": "mix_406000",
        "display": "Final: 40% h10 + 60% h20",
    },
    {
        "name": "stable_h20h30_7030",
        "mix": "mix_007030",
        "display": "Stable: 70% h20 + 30% h30",
    },
    {
        "name": "light_h10h20_1090",
        "mix": "mix_109000",
        "display": "Light: 10% h10 + 90% h20",
    },
]


def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def normalize_nav(s):
    s = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).ffill()
    first = s.dropna().iloc[0]
    if first == 0:
        raise ValueError("first NAV/equity value is zero")
    return s / first


def compound(x):
    x = pd.Series(x).fillna(0.0)
    return float((1.0 + x).prod() - 1.0)


def daily_sharpe_from_nav(df):
    daily = []
    for d, g in df.groupby("date"):
        g = g.sort_values("datetime")
        actual_day = float(g["actual_nav"].iloc[-1] / g["actual_nav"].iloc[0] - 1.0)
        bench_day = float(g["benchmark_nav"].iloc[-1] / g["benchmark_nav"].iloc[0] - 1.0)
        daily.append({"date": d, "alpha_day": actual_day - bench_day})
    daily = pd.DataFrame(daily)
    if len(daily) <= 1 or daily["alpha_day"].std(ddof=1) == 0:
        return np.nan
    return float(daily["alpha_day"].mean() / daily["alpha_day"].std(ddof=1) * np.sqrt(252))


def load_clean_curve(mix):
    strategy = f"pure_cs_v10_{mix}_csi2000_warmstart_noovernight"
    p = V10_ROOT / strategy / f"{strategy}_nav_curve_benchmark_warmstart_noovernight.csv"
    if not p.exists():
        raise FileNotFoundError(p)

    raw = pd.read_csv(p)
    if "datetime" not in raw.columns:
        raise KeyError(f"missing datetime in {p}")

    raw["datetime"] = pd.to_datetime(raw["datetime"])
    if "date" not in raw.columns:
        raw["date"] = raw["datetime"].dt.strftime("%Y%m%d").astype(int)

    raw = raw.sort_values("datetime").reset_index(drop=True)

    dup_count = int(raw.duplicated("datetime").sum())
    print(f"[{mix}] raw rows={len(raw)} duplicated datetime rows={dup_count}")

    # 核心修复：同一个交易分钟只保留一行，避免同一 x 坐标竖线尖峰。
    df = raw.groupby("datetime", as_index=False).last()
    df["date"] = df["datetime"].dt.strftime("%Y%m%d").astype(int)
    df = df.sort_values("datetime").reset_index(drop=True)

    # 优先使用已经算好的 NAV / equity，不再从重复逐行 ret 重算。
    actual_nav_col = pick_col(df, [
        "actual_nav", "strategy_nav", "portfolio_nav",
        "actual_equity", "strategy_equity", "portfolio_equity", "equity",
    ])
    bench_nav_col = pick_col(df, [
        "benchmark_nav", "bench_nav",
        "benchmark_equity", "bench_equity", "benchmark_value",
    ])

    actual_ret_col = pick_col(df, ["actual_ret", "strategy_ret", "portfolio_ret"])
    bench_ret_col = pick_col(df, ["benchmark_ret", "bench_ret"])

    if actual_nav_col is not None and bench_nav_col is not None:
        df["actual_nav"] = normalize_nav(df[actual_nav_col])
        df["benchmark_nav"] = normalize_nav(df[bench_nav_col])
        source = f"nav/equity columns: {actual_nav_col}, {bench_nav_col}"
    elif actual_ret_col is not None and bench_ret_col is not None:
        # 只有在没有 NAV / equity 列时才 fallback 到 ret cumprod。
        df["actual_nav"] = (1.0 + pd.to_numeric(df[actual_ret_col], errors="coerce").fillna(0.0)).cumprod()
        df["benchmark_nav"] = (1.0 + pd.to_numeric(df[bench_ret_col], errors="coerce").fillna(0.0)).cumprod()
        source = f"ret columns after minute de-dup: {actual_ret_col}, {bench_ret_col}"
    else:
        raise KeyError(
            f"cannot find NAV/equity or return columns in {p}; columns={df.columns.tolist()}"
        )

    df["actual_cum"] = df["actual_nav"] - 1.0
    df["benchmark_cum"] = df["benchmark_nav"] - 1.0
    df["alpha_cum"] = df["actual_cum"] - df["benchmark_cum"]

    for c in ["actual_cum", "benchmark_cum", "alpha_cum"]:
        df[f"d_{c}"] = df[c].diff() * 100.0

    # 不删、不平滑，只把疑似尖峰位置输出出来。
    spike = df[
        (df["d_actual_cum"].abs() > 0.50)
        | (df["d_benchmark_cum"].abs() > 0.50)
        | (df["d_alpha_cum"].abs() > 0.30)
    ].copy()

    spike_path = OUT_DIR / f"{strategy}_spike_check.csv"
    spike_cols = [
        "date", "datetime",
        "actual_cum", "benchmark_cum", "alpha_cum",
        "d_actual_cum", "d_benchmark_cum", "d_alpha_cum",
    ]
    spike[spike_cols].to_csv(spike_path, index=False)

    print(f"[{mix}] clean rows={len(df)} source={source}")
    print(f"[{mix}] spike_check rows={len(spike)} saved={spike_path}")

    return df, p, source, dup_count, spike_path


def plot_case(case):
    df, src, source, dup_count, spike_path = load_clean_curve(case["mix"])

    strategy_return = float(df["actual_nav"].iloc[-1] - 1.0)
    benchmark_return = float(df["benchmark_nav"].iloc[-1] - 1.0)
    alpha_return = strategy_return - benchmark_return
    sharpe = daily_sharpe_from_nav(df)

    x = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(14, 8))

    ax.plot(x, df["actual_cum"] * 100.0, label="strategyret", linewidth=1.6)
    ax.plot(x, df["benchmark_cum"] * 100.0, label="benchmarkret", linewidth=1.6)
    ax.plot(x, df["alpha_cum"] * 100.0, label="alpharet", linewidth=1.8)

    ax.axhline(0.0, linestyle="--", linewidth=1.0)

    ax.set_title("T+1-aware Pure-CS NAV with CSI2000 Benchmark", fontsize=16, weight="bold")
    ax.set_ylabel("cumulative return (%)")
    ax.set_xlabel("trading minute index, benchmark warm-start, no overnight PnL")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.25)

    # 日期刻度：用 trading index，不直接用 datetime，避免隔夜间隔和重复 x。
    first_idx = df.groupby("date").head(1).index.to_numpy()
    date_labels = df.loc[first_idx, "date"].astype(str).tolist()
    step = max(1, len(first_idx) // 10)
    ax.set_xticks(first_idx[::step])
    ax.set_xticklabels(date_labels[::step], rotation=45, ha="right")

    summary_text = (
        "Summary\n\n"
        f"Strategy\n{case['display']}\n\n"
        f"Strategy Return\n{strategy_return * 100.0:.2f}%\n\n"
        f"Benchmark Return\n{benchmark_return * 100.0:.2f}%\n\n"
        f"Alpha Return\n{alpha_return * 100.0:.2f}%\n\n"
        f"Daily Excess Sharpe\n{sharpe:.2f}"
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

    plt.subplots_adjust(right=0.76, bottom=0.14)

    out = OUT_DIR / f"{case['name']}_nav_report_clean.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)

    print("[saved]", out)
    print("[source]", src)
    print("[spike_check]", spike_path)
    print(
        f"{case['name']}: "
        f"strategy={strategy_return:.6f}, "
        f"benchmark={benchmark_return:.6f}, "
        f"alpha={alpha_return:.6f}, "
        f"sharpe={sharpe:.6f}, "
        f"dup_removed={dup_count}"
    )


for case in CASES:
    plot_case(case)

print("\n===== CLEAN REPORT PNG DONE =====")
print(OUT_DIR)
