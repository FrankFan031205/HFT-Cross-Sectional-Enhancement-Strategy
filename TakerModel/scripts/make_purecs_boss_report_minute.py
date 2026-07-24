# -*- coding: utf-8 -*-
"""
Minute-level boss report for Pure-CS strategy.

Definitions:
- actualret: minute-level cumulative strategy return. Strategy target weights are forward-filled from rebalance timestamps to minute market grid, marked to market by minute mid-price returns, and execution cost is deducted at rebalance timestamps.
- benchmarkret: minute-level FULL EW benchmark cumulative return. It does NOT use same-gross benchmark.
- alpharet: actualret - benchmarkret.

Outputs:
  field_dictionary_cn.csv
  curve_minute_actualret_benchmarkret_alpharet.csv
  curve_minute_actualret_benchmarkret_alpharet_cn.csv
  curve_minute_actualret_benchmarkret_alpharet.png
  curve_daily_actualret_benchmarkret_alpharet.csv
  curve_daily_actualret_benchmarkret_alpharet_cn.csv
  curve_daily_actualret_benchmarkret_alpharet.png
"""

import argparse
from pathlib import Path
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_any(path):
    path = Path(path)
    suf = "".join(path.suffixes).lower()
    if suf.endswith(".parquet"):
        return pd.read_parquet(path)
    if suf.endswith(".csv") or suf.endswith(".csv.gz"):
        return pd.read_csv(path, low_memory=False)
    raise ValueError(f"unsupported file: {path}")


def read_glob(path_glob):
    files = sorted(glob.glob(path_glob))
    if not files:
        raise FileNotFoundError(f"no files matched: {path_glob}")
    print("[market files]", len(files))
    return pd.concat([read_any(f) for f in files], ignore_index=True)


def pick_col(df, candidates, required=True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"cannot find any of {candidates}; columns={list(df.columns)}")
    return None


def compound(x):
    x = pd.Series(x).replace([np.inf, -np.inf], np.nan).dropna()
    if x.empty:
        return np.nan
    return float((1.0 + x).prod() - 1.0)


def find_one(root, pattern):
    files = sorted(Path(root).glob(pattern))
    if not files:
        raise FileNotFoundError(f"cannot find {pattern} under {root}")
    return files[0]


def normalize_market(mkt):
    date_col = pick_col(mkt, ["date", "execution_date"])
    dt_col = pick_col(mkt, ["datetime", "execution_datetime"])
    sid_col = pick_col(mkt, ["securityid", "SecurityID", "sid", "symbol"])
    mid_col = pick_col(mkt, ["mid_price", "price", "tmid"])
    bench_col = pick_col(mkt, ["benchmark_weight", "ew_benchmark_weight", "bench_weight"], required=False)

    out = pd.DataFrame({
        "date": mkt[date_col].astype(int),
        "datetime": pd.to_datetime(mkt[dt_col]),
        "securityid": mkt[sid_col].astype(str).str.zfill(6),
        "mid_price": pd.to_numeric(mkt[mid_col], errors="coerce"),
    })

    if bench_col is None:
        out["benchmark_weight"] = np.nan
    else:
        out["benchmark_weight"] = pd.to_numeric(mkt[bench_col], errors="coerce")

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=["date", "datetime", "securityid", "mid_price"])
    out = out[out["mid_price"] > 0].copy()

    # Full EW benchmark if benchmark_weight is missing.
    if out["benchmark_weight"].isna().all():
        out["benchmark_weight"] = 1.0 / out.groupby(["date", "datetime"])["securityid"].transform("count")
    else:
        out["benchmark_weight"] = out["benchmark_weight"].fillna(0.0)

    return out.sort_values(["date", "securityid", "datetime"]).reset_index(drop=True)


def normalize_targets(pos):
    date_col = pick_col(pos, ["date", "execution_date"])
    dt_col = pick_col(pos, ["datetime", "execution_datetime"])
    sid_col = pick_col(pos, ["securityid", "SecurityID", "sid", "symbol"])
    w_col = pick_col(pos, ["target_weight", "effective_target_weight", "weight", "w", "opt_weight", "optimized_weight"])

    out = pd.DataFrame({
        "date": pos[date_col].astype(int),
        "datetime": pd.to_datetime(pos[dt_col]),
        "securityid": pos[sid_col].astype(str).str.zfill(6),
        "target_weight_rebalance": pd.to_numeric(pos[w_col], errors="coerce"),
    })

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=["date", "datetime", "securityid", "target_weight_rebalance"])
    out["target_weight_rebalance"] = out["target_weight_rebalance"].clip(lower=0.0)

    return out.sort_values(["date", "securityid", "datetime"]).reset_index(drop=True)


def load_cost_by_rebalance(case_dir, capital):
    minute_path = find_one(case_dir, "pure_cs_taker_minute_*.csv")
    x = pd.read_csv(minute_path, low_memory=False)
    x["date"] = x["date"].astype(int)
    x["datetime"] = pd.to_datetime(x["datetime"])

    if "total_cost" in x.columns:
        x["execution_cost"] = x["total_cost"].astype(float)
    else:
        fee = x["fee"].astype(float) if "fee" in x.columns else 0.0
        spread = x["spread_cost_est"].astype(float) if "spread_cost_est" in x.columns else 0.0
        x["execution_cost"] = fee + spread

    x["execution_cost_return"] = x["execution_cost"] / float(capital)

    keep = ["date", "datetime", "execution_cost", "execution_cost_return"]
    for c in ["turnover", "fee", "spread_cost_est", "total_cost", "target_gross", "actual_gross_weight", "n_hold"]:
        if c in x.columns:
            keep.append(c)

    return x[keep].copy()


def build_minute_curve(market, targets, costs, capital):
    # Merge rebalance weights into minute grid.
    df = market.merge(targets, on=["date", "datetime", "securityid"], how="left")
    df = df.sort_values(["date", "securityid", "datetime"]).reset_index(drop=True)

    # Forward-fill weights within each day and stock.
    df["target_weight"] = (
        df.groupby(["date", "securityid"])["target_weight_rebalance"]
        .ffill()
        .fillna(0.0)
    )

    # Intraday minute forward return.
    df["next_mid"] = df.groupby(["date", "securityid"])["mid_price"].shift(-1)
    df["fwd_ret_1m"] = df["next_mid"] / df["mid_price"] - 1.0
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["fwd_ret_1m"])

    df["strategy_ret_contrib"] = df["target_weight"] * df["fwd_ret_1m"]
    df["benchmark_ret_contrib"] = df["benchmark_weight"] * df["fwd_ret_1m"]

    per_min = (
        df.groupby(["date", "datetime"], as_index=False)
        .agg(
            strategy_ret_no_cost=("strategy_ret_contrib", "sum"),
            benchmark_ret=("benchmark_ret_contrib", "sum"),
            target_gross=("target_weight", "sum"),
            benchmark_gross=("benchmark_weight", "sum"),
            n_names=("securityid", "nunique"),
            n_target_names=("target_weight", lambda x: int((x > 0).sum())),
        )
    )

    # Deduct TakerModel estimated execution cost only at rebalance timestamps.
    per_min = per_min.merge(
        costs,
        on=["date", "datetime"],
        how="left",
        suffixes=("", "_exec"),
    )

    # Defensive fix: if duplicate names became _x/_y in any pandas version,
    # restore the strategy-side columns used by later daily aggregation.
    rename_back = {}
    for c in ["target_gross", "benchmark_gross", "n_names", "n_target_names"]:
        if c not in per_min.columns and f"{c}_x" in per_min.columns:
            rename_back[f"{c}_x"] = c
    if rename_back:
        per_min = per_min.rename(columns=rename_back)

    per_min["execution_cost"] = per_min["execution_cost"].fillna(0.0)
    per_min["execution_cost_return"] = per_min["execution_cost_return"].fillna(0.0)

    per_min["actual_ret"] = per_min["strategy_ret_no_cost"] - per_min["execution_cost_return"]

    # Full benchmark, not same-gross benchmark.
    per_min["actualret"] = (1.0 + per_min["actual_ret"]).cumprod() - 1.0
    per_min["benchmarkret"] = (1.0 + per_min["benchmark_ret"]).cumprod() - 1.0
    per_min["alpharet"] = per_min["actualret"] - per_min["benchmarkret"]

    # Attribution:
    # full alpha = selection alpha + exposure effect.
    # selection alpha compares strategy with benchmark scaled to the strategy gross.
    per_min["scaled_benchmark_ret"] = np.where(
        per_min["benchmark_gross"].abs() > 1e-12,
        per_min["benchmark_ret"] * per_min["target_gross"] / per_min["benchmark_gross"],
        np.nan,
    )
    per_min["selection_alpha_ret_1m"] = per_min["actual_ret"] - per_min["scaled_benchmark_ret"]
    per_min["exposure_effect_ret_1m"] = per_min["scaled_benchmark_ret"] - per_min["benchmark_ret"]
    per_min["alpha_ret_1m"] = per_min["actual_ret"] - per_min["benchmark_ret"]

    per_min["selection_alpharet"] = (1.0 + per_min["selection_alpha_ret_1m"].fillna(0.0)).cumprod() - 1.0
    per_min["exposure_effect_cum"] = (1.0 + per_min["exposure_effect_ret_1m"].fillna(0.0)).cumprod() - 1.0
    per_min["alpharet_compound"] = (1.0 + per_min["alpha_ret_1m"].fillna(0.0)).cumprod() - 1.0

    for c in [
        "actual_ret", "benchmark_ret", "scaled_benchmark_ret",
        "alpha_ret_1m", "selection_alpha_ret_1m", "exposure_effect_ret_1m",
        "actualret", "benchmarkret", "alpharet",
        "selection_alpharet", "exposure_effect_cum", "alpharet_compound"
    ]:
        per_min[c + "_pct"] = per_min[c] * 100.0

    return per_min


def build_daily_curve(min_curve):
    daily = (
        min_curve.groupby("date", as_index=False)
        .agg(
            actual_strategy_return=("actual_ret", compound),
            benchmark_return=("benchmark_ret", compound),
            daily_cost=("execution_cost", "sum"),
            avg_target_gross=("target_gross", "mean"),
            avg_benchmark_gross=("benchmark_gross", "mean"),
            avg_n_target_names=("n_target_names", "mean"),
            n_minutes=("datetime", "nunique"),
        )
    )

    daily["actual_excess_vs_benchmark"] = daily["actual_strategy_return"] - daily["benchmark_return"]
    daily["actualret"] = (1.0 + daily["actual_strategy_return"]).cumprod() - 1.0
    daily["benchmarkret"] = (1.0 + daily["benchmark_return"]).cumprod() - 1.0
    daily["alpharet"] = daily["actualret"] - daily["benchmarkret"]

    ex = daily["actual_excess_vs_benchmark"].replace([np.inf, -np.inf], np.nan).dropna()
    if len(ex) >= 2 and ex.std(ddof=1) > 0:
        daily["daily_excess_sharpe_vs_benchmark"] = ex.mean() / ex.std(ddof=1) * np.sqrt(252)
        daily["daily_excess_tstat_vs_benchmark"] = ex.mean() / ex.std(ddof=1) * np.sqrt(len(ex))
    else:
        daily["daily_excess_sharpe_vs_benchmark"] = np.nan
        daily["daily_excess_tstat_vs_benchmark"] = np.nan

    return daily


def write_cn_versions(min_curve, daily_curve, out_dir):
    min_cn = min_curve.rename(columns={
        "date": "日期",
        "datetime": "时间",
        "actual_ret": "策略分钟收益",
        "benchmark_ret": "基准分钟收益_满仓等权",
        "alpha_ret_1m": "分钟超额收益",
        "actualret": "策略累计收益",
        "benchmarkret": "基准累计收益_满仓等权",
        "alpharet": "累计超额收益_策略减基准",
        "alpharet_compound": "复利累计超额收益",
        "execution_cost": "执行成本_CNY",
        "execution_cost_return": "执行成本收益率",
        "target_gross": "策略目标仓位",
        "benchmark_gross": "基准仓位",
        "n_names": "股票池数量",
        "n_target_names": "目标持仓数量",
    })

    daily_cn = daily_curve.rename(columns={
        "date": "日期",
        "actual_strategy_return": "策略日收益",
        "benchmark_return": "基准日收益_满仓等权",
        "actual_excess_vs_benchmark": "日超额收益",
        "actualret": "策略累计收益",
        "benchmarkret": "基准累计收益_满仓等权",
        "alpharet": "累计超额收益",
        "daily_cost": "日执行成本_CNY",
        "avg_target_gross": "平均策略目标仓位",
        "avg_benchmark_gross": "平均基准仓位",
        "avg_n_target_names": "平均持仓数量",
        "n_minutes": "分钟数",
        "daily_excess_sharpe_vs_benchmark": "日频超额Sharpe",
        "daily_excess_tstat_vs_benchmark": "日频超额t值",
    })

    min_cn.to_csv(out_dir / "curve_minute_actualret_benchmarkret_alpharet_cn.csv", index=False, encoding="utf-8-sig")
    daily_cn.to_csv(out_dir / "curve_daily_actualret_benchmarkret_alpharet_cn.csv", index=False, encoding="utf-8-sig")


def write_field_dictionary(out_dir):
    rows = [
        ("actualret", "策略累计收益", "分钟级策略实际累计收益；用 target weight 在分钟行情上 mark-to-market，并在 rebalance 时扣除估算交易成本。"),
        ("benchmarkret", "基准累计收益", "分钟级 full ZZ2000 等权 benchmark 累计收益；不做 same-gross 缩放。"),
        ("alpharet", "累计超额收益", "actualret - benchmarkret。"),
        ("actual_ret", "策略分钟收益", "当前分钟策略收益，已在 rebalance 时间扣除执行成本。"),
        ("benchmark_ret", "基准分钟收益", "当前分钟 full ZZ2000 等权 benchmark 收益。"),
        ("alpha_ret_1m", "分钟超额收益", "actual_ret - benchmark_ret。"),
        ("alpharet_compound", "复利累计超额收益", "对分钟超额收益 alpha_ret_1m 做复利累乘。"),
        ("target_gross", "策略目标仓位", "当前分钟由最近一次 rebalance target weight 前向填充得到的总 gross。"),
        ("benchmark_gross", "基准仓位", "当前分钟 benchmark 权重之和，一般接近 1。"),
        ("execution_cost", "执行成本", "TakerModel 在 rebalance 时间估算的手续费 + spread cost，单位 CNY。"),
        ("daily_excess_sharpe_vs_benchmark", "日频超额 Sharpe", "用 daily actual_strategy_return - daily benchmark_return 计算的年化 Sharpe。"),
    ]
    pd.DataFrame(rows, columns=["field", "中文名", "含义"]).to_csv(
        out_dir / "field_dictionary_cn.csv", index=False, encoding="utf-8-sig"
    )


def plot_curves(min_curve, daily_curve, out_dir):
    plt.figure(figsize=(13, 6))
    plt.plot(min_curve["datetime"], min_curve["actualret"] * 100.0, label="actualret")
    plt.plot(min_curve["datetime"], min_curve["benchmarkret"] * 100.0, label="benchmarkret")
    plt.plot(min_curve["datetime"], min_curve["alpharet"] * 100.0, label="alpharet")
    plt.axhline(0.0, linewidth=0.8)
    plt.title("Minute-level Curve: actualret vs benchmarkret vs alpharet")
    plt.xlabel("time")
    plt.ylabel("cumulative return (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "curve_minute_actualret_benchmarkret_alpharet.png", dpi=180)
    plt.close()

    plt.figure(figsize=(13, 6))
    dts = pd.to_datetime(daily_curve["date"].astype(str))
    plt.plot(dts, daily_curve["actualret"] * 100.0, marker="o", label="actualret")
    plt.plot(dts, daily_curve["benchmarkret"] * 100.0, marker="o", label="benchmarkret")
    plt.plot(dts, daily_curve["alpharet"] * 100.0, marker="o", label="alpharet")
    plt.axhline(0.0, linewidth=0.8)
    plt.title("Daily Curve: actualret vs benchmarkret vs alpharet")
    plt.xlabel("date")
    plt.ylabel("cumulative return (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "curve_daily_actualret_benchmarkret_alpharet.png", dpi=180)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--positions", required=True)
    ap.add_argument("--market-glob", required=True)
    ap.add_argument("--case-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--capital", type=float, default=200_000_000.0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    market = normalize_market(read_glob(args.market_glob))
    targets = normalize_targets(read_any(args.positions))
    costs = load_cost_by_rebalance(Path(args.case_dir), args.capital)

    print("[market minutes]", market[["date", "datetime"]].drop_duplicates().shape[0])
    print("[target rebalances]", targets[["date", "datetime"]].drop_duplicates().shape[0])
    print("[cost rows]", costs.shape)

    min_curve = build_minute_curve(market, targets, costs, args.capital)
    daily_curve = build_daily_curve(min_curve)

    min_curve.to_csv(out_dir / "curve_minute_actualret_benchmarkret_alpharet.csv", index=False, encoding="utf-8-sig")
    daily_curve.to_csv(out_dir / "curve_daily_actualret_benchmarkret_alpharet.csv", index=False, encoding="utf-8-sig")

    write_cn_versions(min_curve, daily_curve, out_dir)
    write_field_dictionary(out_dir)
    plot_curves(min_curve, daily_curve, out_dir)

    print("\n===== final summary =====")
    print("actualret final     :", min_curve["actualret"].iloc[-1])
    print("benchmarkret final  :", min_curve["benchmarkret"].iloc[-1])
    print("alpharet final      :", min_curve["alpharet"].iloc[-1])
    print("alpha compound final:", min_curve["alpharet_compound"].iloc[-1])
    print("daily excess sharpe :", daily_curve["daily_excess_sharpe_vs_benchmark"].iloc[-1])
    print("\n[saved dir]", out_dir)


if __name__ == "__main__":
    main()
