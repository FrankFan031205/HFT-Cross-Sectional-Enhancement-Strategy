# -*- coding: utf-8 -*-
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


def find_one(root, pattern):
    files = sorted(Path(root).glob(pattern))
    if not files:
        raise FileNotFoundError(f"cannot find {pattern} under {root}")
    return files[0]


def compound_return(x):
    x = pd.Series(x).replace([np.inf, -np.inf], np.nan).dropna()
    if x.empty:
        return np.nan
    return float((1.0 + x).prod() - 1.0)


def normalize_market(mkt):
    date_col = pick_col(mkt, ["date", "execution_date"])
    dt_col = pick_col(mkt, ["datetime", "execution_datetime"])
    sid_col = pick_col(mkt, ["securityid", "SecurityID", "sid", "symbol"])
    mid_col = pick_col(mkt, ["mid_price", "price", "tmid"])
    bid_col = pick_col(mkt, ["bid1", "bid_price", "bid"], required=False)
    ask_col = pick_col(mkt, ["ask1", "ask_price", "ask"], required=False)
    bench_col = pick_col(mkt, ["benchmark_weight", "ew_benchmark_weight", "bench_weight"], required=False)

    out = pd.DataFrame({
        "date": mkt[date_col].astype(int),
        "datetime": pd.to_datetime(mkt[dt_col]),
        "securityid": mkt[sid_col].astype(str).str.zfill(6),
        "mid_price": pd.to_numeric(mkt[mid_col], errors="coerce"),
    })

    if bid_col is not None:
        out["bid_price"] = pd.to_numeric(mkt[bid_col], errors="coerce")
    else:
        out["bid_price"] = out["mid_price"]

    if ask_col is not None:
        out["ask_price"] = pd.to_numeric(mkt[ask_col], errors="coerce")
    else:
        out["ask_price"] = out["mid_price"]

    if bench_col is None:
        out["benchmark_weight"] = np.nan
    else:
        out["benchmark_weight"] = pd.to_numeric(mkt[bench_col], errors="coerce")

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=["date", "datetime", "securityid", "mid_price", "bid_price", "ask_price"])
    out = out[
        (out["mid_price"] > 0)
        & (out["bid_price"] > 0)
        & (out["ask_price"] > 0)
        & (out["ask_price"] >= out["bid_price"])
    ].copy()

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


def load_costs(case_dir, capital):
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
    for c in ["turnover", "fee", "spread_cost_est", "total_cost"]:
        if c in x.columns:
            keep.append(c)

    return x[keep].copy()


def build_nav_curve(market, targets, costs, capital):
    df = market.merge(targets, on=["date", "datetime", "securityid"], how="left")
    df = df.sort_values(["date", "securityid", "datetime"]).reset_index(drop=True)

    # target weight is held until next rebalance.
    # Important: carry position across trading days.
    # We still exclude overnight return below, but the position itself is not reset to zero every morning.
    df["target_weight"] = (
        df.groupby(["securityid"])["target_weight_rebalance"]
        .ffill()
        .fillna(0.0)
    )

    # Opponent level-1 NAV convention for long-only book:
    # use bid_price as liquidation/mark price. This matches mentor's "opponent first-level price" requirement.
    # first minute each day is set to 0, so overnight gap is excluded
    df["mark_price"] = df["bid_price"]
    df["ret_1m"] = df.groupby(["date", "securityid"])["mark_price"].pct_change()
    df["ret_1m"] = df["ret_1m"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # current minute PnL uses previous minute holdings.
    # Carry previous close position into next day open, while ret_1m for first minute is set to 0.
    df["held_weight_prev"] = (
        df.groupby(["securityid"])["target_weight"]
        .shift(1)
        .fillna(0.0)
    )
    df["benchmark_weight_prev"] = (
        df.groupby(["date", "securityid"])["benchmark_weight"]
        .shift(1)
        .fillna(0.0)
    )

    df["strategy_ret_contrib"] = df["held_weight_prev"] * df["ret_1m"]
    df["benchmark_ret_contrib"] = df["benchmark_weight_prev"] * df["ret_1m"]

    per_min = (
        df.groupby(["date", "datetime"], as_index=False)
        .agg(
            actual_ret_no_cost=("strategy_ret_contrib", "sum"),
            benchmark_ret=("benchmark_ret_contrib", "sum"),
            target_gross=("target_weight", "sum"),
            held_gross_prev=("held_weight_prev", "sum"),
            benchmark_gross=("benchmark_weight", "sum"),
            benchmark_gross_prev=("benchmark_weight_prev", "sum"),
            n_names=("securityid", "nunique"),
            n_target_names=("target_weight", lambda x: int((x > 0).sum())),
        )
    )

    per_min = per_min.merge(costs, on=["date", "datetime"], how="left")
    per_min["execution_cost"] = per_min["execution_cost"].fillna(0.0)
    per_min["execution_cost_return"] = per_min["execution_cost_return"].fillna(0.0)

    per_min["actual_ret"] = per_min["actual_ret_no_cost"] - per_min["execution_cost_return"]
    per_min["alpha_ret"] = per_min["actual_ret"] - per_min["benchmark_ret"]

    per_min["strategy_asset"] = float(capital) * (1.0 + per_min["actual_ret"]).cumprod()
    per_min["benchmark_asset"] = float(capital) * (1.0 + per_min["benchmark_ret"]).cumprod()

    per_min["actualret"] = per_min["strategy_asset"] / float(capital) - 1.0
    per_min["benchmarkret"] = per_min["benchmark_asset"] / float(capital) - 1.0
    per_min["alpharet"] = per_min["actualret"] - per_min["benchmarkret"]
    per_min["alpharet_compound"] = (1.0 + per_min["alpha_ret"]).cumprod() - 1.0

    for c in ["actual_ret", "benchmark_ret", "alpha_ret", "actualret", "benchmarkret", "alpharet", "alpharet_compound"]:
        per_min[c + "_pct"] = per_min[c] * 100.0

    return per_min


def build_daily_curve(curve):
    daily = (
        curve.groupby("date", as_index=False)
        .agg(
            actual_strategy_return=("actual_ret", compound_return),
            benchmark_return=("benchmark_ret", compound_return),
            daily_cost=("execution_cost", "sum"),
            avg_target_gross=("target_gross", "mean"),
            avg_held_gross=("held_gross_prev", "mean"),
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


def make_ticks(df):
    first = df.groupby("date", as_index=False)["bar_index"].min()
    n = len(first)
    step = 1 if n <= 12 else 2 if n <= 24 else max(1, n // 12)
    ticks = first.iloc[::step]
    return ticks["bar_index"].tolist(), ticks["date"].astype(str).tolist()


def plot_curves(curve, daily, out_dir):
    curve = curve.copy()
    curve = curve.sort_values(["date", "datetime"]).reset_index(drop=True)
    curve["bar_index"] = np.arange(len(curve))

    tick_pos, tick_lab = make_ticks(curve)

    plt.figure(figsize=(14, 6))
    plt.plot(curve["bar_index"], curve["actualret"] * 100.0, label="actualret")
    plt.plot(curve["bar_index"], curve["benchmarkret"] * 100.0, label="benchmarkret")
    plt.plot(curve["bar_index"], curve["alpharet"] * 100.0, label="alpharet")
    plt.axhline(0.0, linewidth=0.8)
    plt.xticks(tick_pos, tick_lab, rotation=45)
    plt.title("Pure-CS NAV Curve, Compressed Trading Minutes")
    plt.xlabel("trading minute, overnight/weekend gaps removed")
    plt.ylabel("cumulative return (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "nav_curve_minute_compressed_actualret_benchmarkret_alpharet.png", dpi=180)
    plt.close()

    plt.figure(figsize=(14, 5))
    plt.plot(curve["bar_index"], curve["alpharet"] * 100.0, label="alpharet")
    plt.axhline(0.0, linewidth=0.8)
    plt.xticks(tick_pos, tick_lab, rotation=45)
    plt.title("Pure-CS NAV Alpha Curve: actualret - benchmarkret")
    plt.xlabel("trading minute, overnight/weekend gaps removed")
    plt.ylabel("cumulative alpha (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "nav_curve_minute_compressed_alpharet.png", dpi=180)
    plt.close()

    return curve


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
    costs = load_costs(Path(args.case_dir), args.capital)

    print("[market rows]", market.shape)
    print("[market minutes]", market[["date", "datetime"]].drop_duplicates().shape[0])
    print("[target rows]", targets.shape)
    print("[target rebalances]", targets[["date", "datetime"]].drop_duplicates().shape[0])
    print("[cost rows]", costs.shape)

    curve = build_nav_curve(market, targets, costs, args.capital)
    daily = build_daily_curve(curve)
    curve = plot_curves(curve, daily, out_dir)

    curve.to_csv(out_dir / "nav_curve_minute_actualret_benchmarkret_alpharet.csv", index=False, encoding="utf-8-sig")
    daily.to_csv(out_dir / "nav_curve_daily_actualret_benchmarkret_alpharet.csv", index=False, encoding="utf-8-sig")

    print("\n===== final NAV curve summary =====")
    print("n minute bars:", len(curve))
    print("actualret final:", curve["actualret"].iloc[-1])
    print("benchmarkret final:", curve["benchmarkret"].iloc[-1])
    print("alpharet final:", curve["alpharet"].iloc[-1])
    print("daily excess sharpe:", daily["daily_excess_sharpe_vs_benchmark"].iloc[-1])
    print("[saved dir]", out_dir)


if __name__ == "__main__":
    main()
