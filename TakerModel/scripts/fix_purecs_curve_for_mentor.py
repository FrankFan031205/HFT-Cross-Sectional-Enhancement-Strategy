import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compound_return(x):
    x = pd.Series(x).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return (1.0 + x).cumprod() - 1.0


def make_ticks(df):
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


def add_time_flags(df, drop_open, drop_close):
    df = df.sort_values(["date", "datetime"]).copy()
    df["bar_in_day"] = df.groupby("date").cumcount()
    df["bars_in_day"] = df.groupby("date")["datetime"].transform("count")
    df["bar_to_close"] = df["bars_in_day"] - 1 - df["bar_in_day"]
    df["is_open_window"] = df["bar_in_day"] < int(drop_open)
    df["is_close_window"] = df["bar_to_close"] < int(drop_close)
    df["is_open_or_close_window"] = df["is_open_window"] | df["is_close_window"]
    return df


def add_alpha_decomposition(df):
    df = df.copy()

    if "target_gross" in df.columns and "benchmark_gross" in df.columns:
        df["scaled_benchmark_ret"] = np.where(
            df["benchmark_gross"].abs() > 1e-12,
            df["benchmark_ret"] * df["target_gross"] / df["benchmark_gross"],
            np.nan,
        )
    else:
        df["scaled_benchmark_ret"] = np.nan

    df["selection_alpha_ret"] = df["actual_ret"] - df["scaled_benchmark_ret"]
    df["exposure_effect_ret"] = df["scaled_benchmark_ret"] - df["benchmark_ret"]
    df["full_alpha_ret"] = df["actual_ret"] - df["benchmark_ret"]

    df["selection_alpharet"] = compound_return(df["selection_alpha_ret"])
    df["exposure_effect_cum"] = compound_return(df["exposure_effect_ret"])
    df["full_alpharet_compound"] = compound_return(df["full_alpha_ret"])

    return df


def classify_spike(row):
    alpha = abs(float(row.get("full_alpha_ret", 0.0)))
    bench = abs(float(row.get("benchmark_ret", 0.0)))
    actual = abs(float(row.get("actual_ret", 0.0)))
    cost = abs(float(row.get("execution_cost_return", 0.0)))
    sel = abs(float(row.get("selection_alpha_ret", 0.0)))
    expo = abs(float(row.get("exposure_effect_ret", 0.0)))

    if bool(row.get("is_open_or_close_window", False)):
        return "open_or_close_window"
    if cost > 0.00005 and cost > 0.25 * max(alpha, 1e-12):
        return "rebalance_cost"
    if expo > sel and expo > 0.4 * max(alpha, 1e-12):
        return "gross_exposure_effect"
    if bench > actual and bench > 0.4 * max(alpha, 1e-12):
        return "benchmark_move"
    return "selection_or_price_move"


def save_spike_tables(df, out_dir, top_n):
    x = df.copy()
    x["abs_full_alpha_ret"] = x["full_alpha_ret"].abs()
    x["abs_benchmark_ret"] = x["benchmark_ret"].abs()
    x["abs_actual_ret"] = x["actual_ret"].abs()
    x["spike_reason"] = x.apply(classify_spike, axis=1)

    cols = [
        "date", "datetime",
        "actual_ret", "benchmark_ret", "full_alpha_ret",
        "selection_alpha_ret", "exposure_effect_ret",
        "actualret", "benchmarkret", "alpharet",
        "execution_cost", "execution_cost_return",
        "target_gross", "benchmark_gross",
        "is_open_window", "is_close_window",
        "spike_reason",
    ]
    cols = [c for c in cols if c in x.columns]

    top_alpha = x.sort_values("abs_full_alpha_ret", ascending=False).head(top_n)[cols]
    top_benchmark = x.sort_values("abs_benchmark_ret", ascending=False).head(top_n)[cols]

    top_alpha.to_csv(out_dir / "top_alpha_spikes_for_mentor.csv", index=False, encoding="utf-8-sig")
    top_benchmark.to_csv(out_dir / "top_benchmark_spikes_for_mentor.csv", index=False, encoding="utf-8-sig")

    reason_map = pd.DataFrame([
        {
            "spike_reason": "open_or_close_window",
            "中文解释": "尖峰发生在每日开盘或收盘附近，通常受开收盘流动性、报价跳动或集合竞价影响。",
        },
        {
            "spike_reason": "rebalance_cost",
            "中文解释": "尖峰主要来自 rebalance 时刻扣除的手续费或价差成本。",
        },
        {
            "spike_reason": "gross_exposure_effect",
            "中文解释": "尖峰主要来自仓位暴露差。benchmark 是 100% 满仓，策略仓位低于 100%，benchmark 上涨时策略可能没有完全跟上。",
        },
        {
            "spike_reason": "benchmark_move",
            "中文解释": "尖峰主要来自 benchmark 单分钟快速上涨或下跌。",
        },
        {
            "spike_reason": "selection_or_price_move",
            "中文解释": "尖峰主要来自选股组合与 benchmark 的个股收益差异，或个别分钟价格跳动。",
        },
    ])
    reason_map.to_csv(out_dir / "spike_reason_dictionary_cn.csv", index=False, encoding="utf-8-sig")

    return top_alpha, top_benchmark


def recompute_clean_curve(df):
    x = df.copy()
    x["actualret_clean"] = compound_return(x["actual_ret"])
    x["benchmarkret_clean"] = compound_return(x["benchmark_ret"])
    x["alpharet_clean"] = x["actualret_clean"] - x["benchmarkret_clean"]

    x["selection_alpharet_clean"] = compound_return(x["selection_alpha_ret"])
    x["exposure_effect_cum_clean"] = compound_return(x["exposure_effect_ret"])
    x["full_alpharet_compound_clean"] = compound_return(x["full_alpha_ret"])

    x = x.reset_index(drop=True)
    x["bar_index"] = np.arange(len(x))
    return x


def plot_main(df, out_dir, prefix):
    tick_pos, tick_lab = make_ticks(df)

    plt.figure(figsize=(14, 6))
    plt.plot(df["bar_index"], df["actualret_clean"] * 100, label="actualret")
    plt.plot(df["bar_index"], df["benchmarkret_clean"] * 100, label="benchmarkret")
    plt.plot(df["bar_index"], df["alpharet_clean"] * 100, label="alpharet")
    plt.axhline(0, linewidth=0.8)
    plt.xticks(tick_pos, tick_lab, rotation=45)
    plt.title("Pure-CS Clean Minute Curve")
    plt.xlabel("trading minute index, overnight/weekend gaps removed")
    plt.ylabel("cumulative return (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out = out_dir / f"{prefix}_actualret_benchmarkret_alpharet.png"
    plt.savefig(out, dpi=180)
    plt.close()

    plt.figure(figsize=(14, 5))
    plt.plot(df["bar_index"], df["alpharet_clean"] * 100, label="alpharet")
    plt.axhline(0, linewidth=0.8)
    plt.xticks(tick_pos, tick_lab, rotation=45)
    plt.title("Pure-CS Clean Alpha Curve")
    plt.xlabel("trading minute index, overnight/weekend gaps removed")
    plt.ylabel("cumulative alpha (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out2 = out_dir / f"{prefix}_alpharet.png"
    plt.savefig(out2, dpi=180)
    plt.close()

    return out, out2


def plot_attribution(df, out_dir, prefix):
    tick_pos, tick_lab = make_ticks(df)

    plt.figure(figsize=(14, 6))
    plt.plot(df["bar_index"], df["alpharet_clean"] * 100, label="full alpha")
    plt.plot(df["bar_index"], df["selection_alpharet_clean"] * 100, label="selection alpha")
    plt.plot(df["bar_index"], df["exposure_effect_cum_clean"] * 100, label="exposure effect")
    plt.axhline(0, linewidth=0.8)
    plt.xticks(tick_pos, tick_lab, rotation=45)
    plt.title("Pure-CS Alpha Attribution")
    plt.xlabel("trading minute index, overnight/weekend gaps removed")
    plt.ylabel("cumulative return contribution (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out = out_dir / f"{prefix}_alpha_attribution.png"
    plt.savefig(out, dpi=180)
    plt.close()
    return out


def make_cn_curve(df, out_dir, name):
    keep = [
        "date", "datetime", "bar_index",
        "actual_ret", "benchmark_ret", "full_alpha_ret",
        "selection_alpha_ret", "exposure_effect_ret",
        "actualret_clean", "benchmarkret_clean", "alpharet_clean",
        "selection_alpharet_clean", "exposure_effect_cum_clean",
        "execution_cost", "execution_cost_return",
        "target_gross", "benchmark_gross",
        "is_open_window", "is_close_window",
    ]
    keep = [c for c in keep if c in df.columns]
    cn = df[keep].rename(columns={
        "date": "日期",
        "datetime": "时间",
        "bar_index": "连续交易分钟序号",
        "actual_ret": "策略分钟收益",
        "benchmark_ret": "满仓基准分钟收益",
        "full_alpha_ret": "分钟超额收益_策略减满仓基准",
        "selection_alpha_ret": "选股超额收益_相对同仓位基准",
        "exposure_effect_ret": "仓位暴露贡献",
        "actualret_clean": "策略累计收益",
        "benchmarkret_clean": "满仓基准累计收益",
        "alpharet_clean": "累计超额收益",
        "selection_alpharet_clean": "累计选股超额",
        "exposure_effect_cum_clean": "累计仓位暴露贡献",
        "execution_cost": "执行成本_CNY",
        "execution_cost_return": "执行成本收益率",
        "target_gross": "策略目标仓位",
        "benchmark_gross": "基准仓位",
        "is_open_window": "是否开盘窗口",
        "is_close_window": "是否收盘窗口",
    })
    cn.to_csv(out_dir / name, index=False, encoding="utf-8-sig")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curve-csv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--drop-open-minutes", type=int, default=5)
    ap.add_argument("--drop-close-minutes", type=int, default=5)
    ap.add_argument("--top-n", type=int, default=50)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.curve_csv)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values(["date", "datetime"]).reset_index(drop=True)

    needed = ["actual_ret", "benchmark_ret", "actualret", "benchmarkret", "alpharet"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"missing columns in curve csv: {missing}")

    df = add_time_flags(df, args.drop_open_minutes, args.drop_close_minutes)
    df = add_alpha_decomposition(df)

    top_alpha, top_benchmark = save_spike_tables(df, out_dir, args.top_n)

    raw = recompute_clean_curve(df)
    raw.to_csv(out_dir / "mentor_curve_raw_with_attribution.csv", index=False, encoding="utf-8-sig")
    make_cn_curve(raw, out_dir, "mentor_curve_raw_with_attribution_cn.csv")
    plot_main(raw, out_dir, "mentor_raw")
    plot_attribution(raw, out_dir, "mentor_raw")

    clean = df[~df["is_open_or_close_window"]].copy()
    clean = recompute_clean_curve(clean)
    clean.to_csv(out_dir / "mentor_curve_clean_ex_open_close.csv", index=False, encoding="utf-8-sig")
    make_cn_curve(clean, out_dir, "mentor_curve_clean_ex_open_close_cn.csv")
    plot_main(clean, out_dir, "mentor_clean_ex_open_close")
    plot_attribution(clean, out_dir, "mentor_clean_ex_open_close")

    print("\n===== raw final =====")
    print("n bars:", len(raw))
    print("actualret:", raw["actualret_clean"].iloc[-1])
    print("benchmarkret:", raw["benchmarkret_clean"].iloc[-1])
    print("alpharet:", raw["alpharet_clean"].iloc[-1])
    print("selection_alpharet:", raw["selection_alpharet_clean"].iloc[-1])
    print("exposure_effect:", raw["exposure_effect_cum_clean"].iloc[-1])

    print("\n===== clean ex open/close final =====")
    print("n bars:", len(clean))
    print("actualret:", clean["actualret_clean"].iloc[-1])
    print("benchmarkret:", clean["benchmarkret_clean"].iloc[-1])
    print("alpharet:", clean["alpharet_clean"].iloc[-1])
    print("selection_alpharet:", clean["selection_alpharet_clean"].iloc[-1])
    print("exposure_effect:", clean["exposure_effect_cum_clean"].iloc[-1])

    print("\n===== top alpha spikes =====")
    cols = ["date", "datetime", "actual_ret", "benchmark_ret", "full_alpha_ret", "selection_alpha_ret", "exposure_effect_ret", "execution_cost_return", "target_gross", "spike_reason"]
    cols = [c for c in cols if c in top_alpha.columns]
    print(top_alpha[cols].head(20).to_string(index=False))

    print("\n[saved dir]", out_dir)


if __name__ == "__main__":
    main()
