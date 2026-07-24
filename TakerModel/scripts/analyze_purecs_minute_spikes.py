# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def pick_curve_file(path: Path) -> Path:
    if path.is_file():
        return path

    candidates = [
        path / "nav_curve_minute_actualret_benchmarkret_alpharet.csv",
        path / "curve_minute_compressed_actualret_benchmarkret_alpharet.csv",
        path / "curve_minute_actualret_benchmarkret_alpharet.csv",
    ]

    for p in candidates:
        if p.exists():
            return p

    raise FileNotFoundError(f"cannot find minute curve csv under {path}")


def compound_return(x):
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


def prepare_curve(df, drop_open, drop_close):
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values(["date", "datetime"]).reset_index(drop=True)

    if "actual_ret" not in df.columns:
        raise KeyError("curve csv must contain actual_ret")
    if "benchmark_ret" not in df.columns:
        raise KeyError("curve csv must contain benchmark_ret")

    df["actual_ret"] = pd.to_numeric(df["actual_ret"], errors="coerce").fillna(0.0)
    df["benchmark_ret"] = pd.to_numeric(df["benchmark_ret"], errors="coerce").fillna(0.0)

    if "alpha_ret" in df.columns:
        df["alpha_ret"] = pd.to_numeric(df["alpha_ret"], errors="coerce").fillna(0.0)
    elif "alpha_ret_1m" in df.columns:
        df["alpha_ret"] = pd.to_numeric(df["alpha_ret_1m"], errors="coerce").fillna(0.0)
    else:
        df["alpha_ret"] = df["actual_ret"] - df["benchmark_ret"]

    if "actualret" not in df.columns:
        df["actualret"] = compound_return(df["actual_ret"])
    if "benchmarkret" not in df.columns:
        df["benchmarkret"] = compound_return(df["benchmark_ret"])
    if "alpharet" not in df.columns:
        df["alpharet"] = df["actualret"] - df["benchmarkret"]

    for c in ["execution_cost_return", "execution_cost", "target_gross", "held_gross_prev", "benchmark_gross", "n_target_names"]:
        if c not in df.columns:
            df[c] = np.nan

    df["bar_in_day"] = df.groupby("date").cumcount()
    df["bars_in_day"] = df.groupby("date")["datetime"].transform("count")
    df["bar_to_close"] = df["bars_in_day"] - 1 - df["bar_in_day"]

    df["is_open_window"] = df["bar_in_day"] < int(drop_open)
    df["is_close_window"] = df["bar_to_close"] < int(drop_close)
    df["is_open_or_close_window"] = df["is_open_window"] | df["is_close_window"]

    df["abs_alpha_ret"] = df["alpha_ret"].abs()
    df["abs_actual_ret"] = df["actual_ret"].abs()
    df["abs_benchmark_ret"] = df["benchmark_ret"].abs()

    df["bar_index"] = np.arange(len(df))

    return df


def classify_reason(row):
    alpha = abs(float(row.get("alpha_ret", 0.0)))
    actual = abs(float(row.get("actual_ret", 0.0)))
    bench = abs(float(row.get("benchmark_ret", 0.0)))
    cost = abs(float(row.get("execution_cost_return", 0.0)))
    gross = row.get("target_gross", np.nan)

    if bool(row.get("is_open_window", False)):
        return "open_window"

    if bool(row.get("is_close_window", False)):
        return "close_window"

    if cost > 0.00005 and cost > 0.20 * max(alpha, 1e-12):
        return "rebalance_cost"

    if np.isfinite(gross) and gross < 0.95 and row.get("benchmark_ret", 0.0) > 0:
        return "under_gross_when_benchmark_up"

    if bench > actual and bench > 0.50 * max(alpha, 1e-12):
        return "benchmark_move"

    if actual > bench and actual > 0.50 * max(alpha, 1e-12):
        return "portfolio_price_move"

    return "selection_or_noise"


def add_reasons(df):
    df = df.copy()
    df["spike_reason"] = df.apply(classify_reason, axis=1)
    return df


def save_tables(df, out_dir, top_n):
    cols = [
        "date",
        "datetime",
        "bar_in_day",
        "bar_to_close",
        "is_open_window",
        "is_close_window",
        "actual_ret",
        "benchmark_ret",
        "alpha_ret",
        "actualret",
        "benchmarkret",
        "alpharet",
        "execution_cost",
        "execution_cost_return",
        "target_gross",
        "held_gross_prev",
        "benchmark_gross",
        "n_target_names",
        "spike_reason",
    ]

    cols = [c for c in cols if c in df.columns]

    top_alpha = df.sort_values("abs_alpha_ret", ascending=False).head(top_n)[cols]
    top_actual = df.sort_values("abs_actual_ret", ascending=False).head(top_n)[cols]
    top_benchmark = df.sort_values("abs_benchmark_ret", ascending=False).head(top_n)[cols]

    top_alpha.to_csv(out_dir / "top_alpha_spikes.csv", index=False, encoding="utf-8-sig")
    top_actual.to_csv(out_dir / "top_actual_spikes.csv", index=False, encoding="utf-8-sig")
    top_benchmark.to_csv(out_dir / "top_benchmark_spikes.csv", index=False, encoding="utf-8-sig")

    reason = pd.DataFrame([
        {
            "spike_reason": "open_window",
            "中文解释": "尖峰发生在每日开盘附近，通常来自开盘后价格跳动、流动性变化或集合竞价后的价格修正。",
        },
        {
            "spike_reason": "close_window",
            "中文解释": "尖峰发生在每日收盘附近，通常来自收盘前流动性变化或尾盘价格跳动。",
        },
        {
            "spike_reason": "rebalance_cost",
            "中文解释": "尖峰主要来自 rebalance 时刻扣除的交易成本。",
        },
        {
            "spike_reason": "under_gross_when_benchmark_up",
            "中文解释": "benchmark 上涨，但策略仓位低于 100%，因此 full benchmark 口径下 alpha 会短暂下降。",
        },
        {
            "spike_reason": "benchmark_move",
            "中文解释": "尖峰主要来自 full benchmark 在该分钟快速上涨或下跌。",
        },
        {
            "spike_reason": "portfolio_price_move",
            "中文解释": "尖峰主要来自策略持仓股票在该分钟的价格跳动。",
        },
        {
            "spike_reason": "selection_or_noise",
            "中文解释": "尖峰主要来自策略组合与 benchmark 的横截面收益差异，或分钟级报价噪音。",
        },
    ])
    reason.to_csv(out_dir / "spike_reason_dictionary_cn.csv", index=False, encoding="utf-8-sig")

    return top_alpha, top_actual, top_benchmark


def plot_with_markers(df, top_alpha, out_dir):
    tick_pos, tick_lab = make_date_ticks(df)

    marker_idx = set(top_alpha["bar_in_day"].index.tolist()) if "bar_in_day" in top_alpha.columns else set()

    top_points = df.loc[top_alpha.index.intersection(df.index)].copy()

    plt.figure(figsize=(14, 6))
    plt.plot(df["bar_index"], df["actualret"] * 100.0, label="actualret")
    plt.plot(df["bar_index"], df["benchmarkret"] * 100.0, label="benchmarkret")
    plt.plot(df["bar_index"], df["alpharet"] * 100.0, label="alpharet")

    if not top_points.empty:
        plt.scatter(
            top_points["bar_index"],
            top_points["alpharet"] * 100.0,
            s=22,
            marker="x",
            label="top alpha spikes",
        )

    plt.axhline(0.0, linewidth=0.8)
    plt.xticks(tick_pos, tick_lab, rotation=45)
    plt.title("Pure-CS Minute Curve with Spike Markers")
    plt.xlabel("trading minute index, overnight/weekend gaps removed")
    plt.ylabel("cumulative return (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out = out_dir / "spike_actual_benchmark_alpha_curve.png"
    plt.savefig(out, dpi=180)
    plt.close()

    plt.figure(figsize=(14, 5))
    plt.plot(df["bar_index"], df["alpha_ret"] * 100.0, label="minute alpha return")
    if not top_points.empty:
        plt.scatter(
            top_points["bar_index"],
            top_points["alpha_ret"] * 100.0,
            s=22,
            marker="x",
            label="top alpha spikes",
        )
    plt.axhline(0.0, linewidth=0.8)
    plt.xticks(tick_pos, tick_lab, rotation=45)
    plt.title("Minute Alpha Return Spikes")
    plt.xlabel("trading minute index, overnight/weekend gaps removed")
    plt.ylabel("one-minute alpha return (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out2 = out_dir / "spike_minute_alpha_return.png"
    plt.savefig(out2, dpi=180)
    plt.close()

    return out, out2


def summarize(df, top_alpha, out_dir):
    reason_counts = top_alpha["spike_reason"].value_counts(dropna=False).reset_index()
    reason_counts.columns = ["spike_reason", "count_in_top_alpha_spikes"]
    reason_counts.to_csv(out_dir / "top_alpha_spike_reason_counts.csv", index=False, encoding="utf-8-sig")

    open_share = float(top_alpha["is_open_window"].mean()) if "is_open_window" in top_alpha.columns else np.nan
    close_share = float(top_alpha["is_close_window"].mean()) if "is_close_window" in top_alpha.columns else np.nan

    summary = pd.DataFrame([{
        "n_minutes": len(df),
        "final_actualret": df["actualret"].iloc[-1],
        "final_benchmarkret": df["benchmarkret"].iloc[-1],
        "final_alpharet": df["alpharet"].iloc[-1],
        "top_alpha_spikes_open_window_share": open_share,
        "top_alpha_spikes_close_window_share": close_share,
        "max_abs_alpha_ret": df["abs_alpha_ret"].max(),
        "max_abs_actual_ret": df["abs_actual_ret"].max(),
        "max_abs_benchmark_ret": df["abs_benchmark_ret"].max(),
    }])
    summary.to_csv(out_dir / "spike_summary.csv", index=False, encoding="utf-8-sig")

    return summary, reason_counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curve-csv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--drop-open-minutes", type=int, default=5)
    ap.add_argument("--drop-close-minutes", type=int, default=5)
    ap.add_argument("--top-n", type=int, default=50)
    args = ap.parse_args()

    curve_path = pick_curve_file(Path(args.curve_csv))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[read curve]", curve_path)
    df = pd.read_csv(curve_path)
    df = prepare_curve(df, args.drop_open_minutes, args.drop_close_minutes)
    df = add_reasons(df)

    df.to_csv(out_dir / "minute_curve_with_spike_reason.csv", index=False, encoding="utf-8-sig")

    top_alpha, top_actual, top_benchmark = save_tables(df, out_dir, args.top_n)
    p1, p2 = plot_with_markers(df, top_alpha, out_dir)
    summary, reason_counts = summarize(df, top_alpha, out_dir)

    print("\n===== spike summary =====")
    print(summary.to_string(index=False))

    print("\n===== top alpha spike reason counts =====")
    print(reason_counts.to_string(index=False))

    print("\n===== top 20 alpha spikes =====")
    show_cols = [
        "date", "datetime", "bar_in_day", "bar_to_close",
        "actual_ret", "benchmark_ret", "alpha_ret",
        "execution_cost_return", "target_gross",
        "spike_reason",
    ]
    show_cols = [c for c in show_cols if c in top_alpha.columns]
    print(top_alpha[show_cols].head(20).to_string(index=False))

    print("\n[saved dir]", out_dir)
    print("[saved plot]", p1)
    print("[saved plot]", p2)


if __name__ == "__main__":
    main()
