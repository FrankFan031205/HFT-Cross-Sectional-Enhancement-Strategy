"""
Generate boss-readable report files for Pure-CS strategy.

Outputs:
  1. 中文字段解释表
  2. summary 中文长表
  3. robustness 中文汇总表
  4. rebalance-level curve CSV
  5. daily-level curve CSV
  6. curve PNG: actualret / benchmarkret / alpharet

Current main use case:
  purecs_h20_gross90_min5000_volclip1
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def find_one(root: Path, pattern: str) -> Path:
    files = sorted(root.glob(pattern))
    if not files:
        raise FileNotFoundError(f"cannot find {pattern} under {root}")
    if len(files) > 1:
        print("[WARN] multiple files found, using first:")
        for f in files[:10]:
            print("  ", f)
    return files[0]


def compound(x):
    x = pd.Series(x).replace([np.inf, -np.inf], np.nan).dropna()
    if x.empty:
        return np.nan
    return float((1.0 + x).prod() - 1.0)


def pct(x):
    if pd.isna(x):
        return ""
    return f"{100.0 * float(x):.4f}%"


def money(x):
    if pd.isna(x):
        return ""
    return f"{float(x):,.2f}"


def num(x, nd=4):
    if pd.isna(x):
        return ""
    return f"{float(x):.{nd}f}"


FIELD_INFO = {
    "actual_return": ("实际策略收益率", "策略经过执行、手续费、价差成本后的总收益率。", "收益率"),
    "benchmark_return": ("满仓等权基准收益率", "同一可交易股票池、同一 rebalance 频率下，100% 等权 benchmark 的收益。", "收益率"),
    "same_gross_benchmark_return": ("同仓位等权基准收益率", "把 benchmark 按策略 target gross 缩放后的收益，更适合与非 100% gross 策略比较。", "收益率"),
    "actual_excess_vs_full_benchmark": ("相对满仓基准超额收益", "actual_return - benchmark_return。", "收益率"),
    "actual_excess_vs_same_gross_benchmark": ("相对同仓位基准超额收益", "actual_return - same_gross_benchmark_return，主报告建议看这一列。", "收益率"),
    "no_cost_excess_vs_same_gross_benchmark": ("无成本同仓位超额收益", "不考虑执行成本时，策略相对同仓位 benchmark 的超额收益。", "收益率"),
    "total_net_pnl": ("策略净 PnL", "策略最终净收益金额，已经扣除手续费和价差成本。", "CNY"),
    "total_turnover": ("总成交额", "回测期间总交易 notional。", "CNY"),
    "turnover_to_capital": ("换手倍数", "total_turnover / capital。", "倍"),
    "total_cost": ("总交易成本", "total_fee + total_spread_cost_est。", "CNY"),
    "total_fee": ("总手续费", "按成交额和 fee_bps 估算的显性手续费。", "CNY"),
    "total_spread_cost_est": ("估算价差成本", "taker 买在 ask、卖在 bid 相对 mid 的价差成本估计。", "CNY"),
    "avg_target_gross": ("平均目标仓位", "optimizer 目标权重的平均 gross。", "比例"),
    "avg_actual_gross": ("平均实际仓位", "执行后实际持仓的平均 gross。", "比例"),
    "avg_actual_net": ("平均实际净仓位", "执行后实际持仓的平均 net。long-only 下接近 gross。", "比例"),
    "avg_n_hold": ("平均持仓股票数", "每个 rebalance 时点实际持有的平均股票数量。", "只"),
    "num_trade_events": ("交易事件数", "买卖交易事件数量。", "次"),
    "total_blocked_small": ("小单过滤次数", "低于 min_trade_notional 而没有执行的交易次数。", "次"),
    "max_drawdown": ("最大回撤", "按策略 accounting equity 计算的最大回撤金额。", "CNY"),
    "daily_excess_sharpe_same_gross": ("日频超额 Sharpe", "基于日频 actual_excess_vs_same_gross_benchmark 计算并年化。", "Sharpe"),
    "daily_excess_tstat_same_gross": ("日频超额 t-stat", "基于日频超额收益序列计算的 t 值。", "t-stat"),
    "daily_excess_mean": ("日均超额收益", "日频同仓位 benchmark 超额收益均值。", "收益率"),
    "daily_excess_std": ("日频超额波动", "日频同仓位 benchmark 超额收益标准差。", "收益率"),
}


def build_field_dictionary(out_dir: Path):
    rows = []
    for k, (cn, meaning, unit) in FIELD_INFO.items():
        rows.append({
            "field": k,
            "中文名": cn,
            "单位": unit,
            "含义": meaning,
            "老板阅读提示": "主结果优先看 actual_excess_vs_same_gross_benchmark 和 daily_excess_sharpe_same_gross。"
        })
    df = pd.DataFrame(rows)
    out = out_dir / "field_dictionary_cn.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print("[saved]", out)
    return df


def build_summary_cn(summary_path: Path, out_dir: Path, capital: float):
    s = pd.read_csv(summary_path)
    row = s.iloc[0].to_dict()

    ordered_fields = [
        "actual_return",
        "benchmark_return",
        "same_gross_benchmark_return",
        "actual_excess_vs_full_benchmark",
        "actual_excess_vs_same_gross_benchmark",
        "no_cost_excess_vs_same_gross_benchmark",
        "total_net_pnl",
        "total_turnover",
        "turnover_to_capital",
        "total_cost",
        "total_fee",
        "total_spread_cost_est",
        "avg_target_gross",
        "avg_actual_gross",
        "avg_actual_net",
        "avg_n_hold",
        "num_trade_events",
        "total_blocked_small",
        "max_drawdown",
    ]

    rows = []
    for f in ordered_fields:
        if f not in row:
            continue
        cn, meaning, unit = FIELD_INFO.get(f, (f, "", ""))

        v = row[f]
        if "return" in f or "excess" in f or "gross" in f or f in ["avg_actual_net"]:
            show = pct(v)
        elif "pnl" in f or "cost" in f or "fee" in f or "turnover" == f or f == "total_turnover" or f == "max_drawdown":
            show = money(v)
        else:
            show = num(v, 4)

        rows.append({
            "field": f,
            "中文名": cn,
            "数值": v,
            "展示值": show,
            "单位": unit,
            "解释": meaning,
        })

    # Add excess PnL rows
    if "actual_excess_vs_same_gross_benchmark" in row:
        ex = float(row["actual_excess_vs_same_gross_benchmark"])
        rows.append({
            "field": "actual_excess_pnl_vs_same_gross_benchmark",
            "中文名": "同仓位基准超额 PnL",
            "数值": ex * capital,
            "展示值": money(ex * capital),
            "单位": "CNY",
            "解释": "actual_excess_vs_same_gross_benchmark × capital。",
        })

    if "actual_excess_vs_full_benchmark" in row:
        ex = float(row["actual_excess_vs_full_benchmark"])
        rows.append({
            "field": "actual_excess_pnl_vs_full_benchmark",
            "中文名": "满仓基准超额 PnL",
            "数值": ex * capital,
            "展示值": money(ex * capital),
            "单位": "CNY",
            "解释": "actual_excess_vs_full_benchmark × capital。",
        })

    out_df = pd.DataFrame(rows)
    out = out_dir / "summary_for_boss_cn.csv"
    out_df.to_csv(out, index=False, encoding="utf-8-sig")
    print("[saved]", out)
    return out_df


def build_robustness_cn(robustness_path: Path, out_dir: Path, capital: float):
    if not robustness_path.exists():
        print("[skip] robustness csv not found:", robustness_path)
        return None

    df = pd.read_csv(robustness_path)

    # 如果没有 Sharpe，允许先跳过 Sharpe 列。
    rows = []
    for _, r in df.iterrows():
        ex_same = float(r.get("actual_excess_vs_same_gross_benchmark", np.nan))
        rows.append({
            "版本": r.get("case", ""),
            "实际收益率": pct(r.get("actual_return", np.nan)),
            "满仓基准收益率": pct(r.get("benchmark_return", np.nan)),
            "同仓位基准收益率": pct(r.get("same_gross_benchmark_return", np.nan)),
            "同仓位超额收益率": pct(ex_same),
            "同仓位超额PnL_CNY": money(ex_same * capital) if np.isfinite(ex_same) else "",
            "日频超额Sharpe": num(r.get("daily_excess_sharpe_same_gross", np.nan), 4),
            "日频超额t值": num(r.get("daily_excess_tstat_same_gross", np.nan), 4),
            "换手倍数": num(r.get("turnover_to_capital", np.nan), 4),
            "总成本_CNY": money(r.get("total_cost", np.nan)),
            "手续费_CNY": money(r.get("total_fee", np.nan)),
            "价差成本_CNY": money(r.get("total_spread_cost_est", np.nan)),
            "平均目标仓位": pct(r.get("avg_target_gross", np.nan)),
            "平均实际仓位": pct(r.get("avg_actual_gross", np.nan)),
            "平均持仓数": num(r.get("avg_n_hold", np.nan), 2),
            "交易事件数": num(r.get("num_trade_events", np.nan), 0),
            "小单过滤次数": num(r.get("total_blocked_small", np.nan), 0),
            "最大回撤_CNY": money(r.get("max_drawdown", np.nan)),
        })

    out_df = pd.DataFrame(rows)
    out = out_dir / "robustness_for_boss_cn.csv"
    out_df.to_csv(out, index=False, encoding="utf-8-sig")
    print("[saved]", out)
    return out_df


def build_curves(minute_path: Path, attrib_path: Path, daily_path: Path, out_dir: Path, capital: float):
    minute = pd.read_csv(minute_path)
    attrib = pd.read_csv(attrib_path)

    minute["datetime"] = pd.to_datetime(minute["datetime"])
    attrib["datetime"] = pd.to_datetime(attrib["datetime"])

    keep = [
        "date", "datetime",
        "benchmark_return",
        "same_gross_benchmark_return",
        "no_cost_strategy_return",
        "no_cost_excess_vs_same_gross_benchmark",
        "no_cost_excess_vs_full_benchmark",
    ]
    keep = [c for c in keep if c in attrib.columns]

    x = minute.merge(attrib[keep], on=["date", "datetime"], how="left")
    x = x.sort_values(["datetime", "date"]).reset_index(drop=True)

    # Actual cumulative return from executed accounting equity.
    if "accounting_equity" in x.columns:
        x["actualret"] = x["accounting_equity"] / capital - 1.0
    else:
        x["actualret"] = x["total_pnl"].cumsum() / capital

    # Benchmark curve: main curve uses same-gross benchmark.
    x["benchmarkret"] = (1.0 + x["same_gross_benchmark_return"].fillna(0.0)).cumprod() - 1.0
    x["full_benchmarkret"] = (1.0 + x["benchmark_return"].fillna(0.0)).cumprod() - 1.0

    # Alpha curve = actual cumulative return - benchmark cumulative return.
    x["alpharet"] = x["actualret"] - x["benchmarkret"]

    # No-cost reference curves.
    if "no_cost_strategy_return" in x.columns:
        x["no_cost_actualret"] = (1.0 + x["no_cost_strategy_return"].fillna(0.0)).cumprod() - 1.0
        x["no_cost_alpharet"] = x["no_cost_actualret"] - x["benchmarkret"]

    curve_cols = [
        "date", "datetime",
        "actualret", "benchmarkret", "alpharet",
        "full_benchmarkret",
        "turnover", "fee", "spread_cost_est", "total_cost",
        "target_gross", "actual_gross_weight", "n_hold",
    ]
    curve_cols = [c for c in curve_cols if c in x.columns]

    curve = x[curve_cols].copy()

    # Add percent columns for boss readability.
    for c in ["actualret", "benchmarkret", "alpharet", "full_benchmarkret"]:
        if c in curve.columns:
            curve[c + "_pct"] = curve[c] * 100.0

    out_curve = out_dir / "curve_rebalance_actualret_benchmarkret_alpharet.csv"
    curve.to_csv(out_curve, index=False, encoding="utf-8-sig")
    print("[saved]", out_curve)

    # Daily curve.
    daily = pd.read_csv(daily_path)
    daily["date"] = daily["date"].astype(int)

    daily["actualret"] = daily["daily_net_pnl"].cumsum() / capital
    daily["benchmarkret"] = (1.0 + daily["same_gross_benchmark_return"].fillna(0.0)).cumprod() - 1.0
    daily["full_benchmarkret"] = (1.0 + daily["benchmark_return"].fillna(0.0)).cumprod() - 1.0
    daily["alpharet"] = daily["actualret"] - daily["benchmarkret"]

    if "actual_excess_vs_same_gross_benchmark" in daily.columns:
        ex = daily["actual_excess_vs_same_gross_benchmark"].astype(float)
        daily["daily_excess_return"] = ex
        daily["daily_excess_cumsum"] = ex.cumsum()

        if ex.std(ddof=1) > 0:
            sharpe = ex.mean() / ex.std(ddof=1) * np.sqrt(252)
        else:
            sharpe = np.nan
        daily["daily_excess_sharpe_same_gross"] = sharpe

    daily_out_cols = [
        "date",
        "actual_strategy_return",
        "same_gross_benchmark_return",
        "actual_excess_vs_same_gross_benchmark",
        "actualret", "benchmarkret", "alpharet",
        "full_benchmarkret",
        "daily_excess_return", "daily_excess_cumsum",
        "daily_excess_sharpe_same_gross",
    ]
    daily_out_cols = [c for c in daily_out_cols if c in daily.columns]

    out_daily = out_dir / "curve_daily_actualret_benchmarkret_alpharet.csv"
    daily[daily_out_cols].to_csv(out_daily, index=False, encoding="utf-8-sig")
    print("[saved]", out_daily)

    # Plot rebalance-level curve.
    plt.figure(figsize=(12, 6))
    plt.plot(curve["datetime"], curve["actualret"] * 100.0, label="actualret")
    plt.plot(curve["datetime"], curve["benchmarkret"] * 100.0, label="benchmarkret_same_gross")
    plt.plot(curve["datetime"], curve["alpharet"] * 100.0, label="alpharet")
    plt.axhline(0.0, linewidth=0.8)
    plt.title("Pure-CS Curve: actualret vs benchmarkret vs alpharet")
    plt.xlabel("time")
    plt.ylabel("cumulative return (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_png = out_dir / "curve_rebalance_actualret_benchmarkret_alpharet.png"
    plt.savefig(out_png, dpi=180)
    plt.close()
    print("[saved]", out_png)

    # Plot daily curve.
    plt.figure(figsize=(12, 6))
    plt.plot(pd.to_datetime(daily["date"].astype(str)), daily["actualret"] * 100.0, marker="o", label="actualret")
    plt.plot(pd.to_datetime(daily["date"].astype(str)), daily["benchmarkret"] * 100.0, marker="o", label="benchmarkret_same_gross")
    plt.plot(pd.to_datetime(daily["date"].astype(str)), daily["alpharet"] * 100.0, marker="o", label="alpharet")
    plt.axhline(0.0, linewidth=0.8)
    plt.title("Pure-CS Daily Curve")
    plt.xlabel("date")
    plt.ylabel("cumulative return (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_daily_png = out_dir / "curve_daily_actualret_benchmarkret_alpharet.png"
    plt.savefig(out_daily_png, dpi=180)
    plt.close()
    print("[saved]", out_daily_png)

    return curve, daily


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--capital", type=float, default=200_000_000.0)
    ap.add_argument("--robustness-csv", default="")
    args = ap.parse_args()

    case_dir = Path(args.case_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = find_one(case_dir, "pure_cs_taker_summary_*.csv")
    minute_path = find_one(case_dir, "pure_cs_taker_minute_*.csv")
    daily_path = find_one(case_dir, "pure_cs_taker_daily_*.csv")
    attrib_path = find_one(case_dir, "pure_cs_taker_attribution_grid_*.csv")

    print("[summary]", summary_path)
    print("[minute ]", minute_path)
    print("[daily  ]", daily_path)
    print("[attrib ]", attrib_path)

    build_field_dictionary(out_dir)
    build_summary_cn(summary_path, out_dir, args.capital)

    if args.robustness_csv:
        build_robustness_cn(Path(args.robustness_csv), out_dir, args.capital)

    build_curves(minute_path, attrib_path, daily_path, out_dir, args.capital)

    print("\n[done] report dir:", out_dir)


if __name__ == "__main__":
    main()
PY
