# -*- coding: utf-8 -*-
"""
Plot NAV for T+1-aware pure-CS optimizer output.

Inputs
------
1. market-glob:
   ZZY minute market parquet/csv files used for NAV curve.
   Example:
   /mnt/data1/fwz/HFT_010-dev_fwz_outputs/TakerPipeline/zzy_minute_market_for_curve_tmid/*.parquet

2. positions:
   T+1-aware optimizer output target_positions.csv.
   It should contain actual_weight_after, which is the executed actual position after T+1/cash/lot constraints.

3. rebalance-summary:
   T+1-aware optimizer output summary_by_rebalance.csv.
   It contains total_cost and turnover_weight at each rebalance timestamp.

Outputs
-------
<tag>_nav_curve.csv
<tag>_nav_curve.png
<tag>_alpha_curve.png
<tag>_plot_summary.csv

Important
---------
- PnL uses previous-minute actual weight, not current target weight.
- NAV curve is compounded from minute returns.
- Overnight return is excluded by using within-day pct_change.
- Trading cost is deducted at rebalance timestamps.
"""

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_any(path: str) -> pd.DataFrame:
    path = Path(path)
    suffix = "".join(path.suffixes).lower()
    if suffix.endswith(".parquet"):
        return pd.read_parquet(path)
    if suffix.endswith(".csv") or suffix.endswith(".csv.gz"):
        return pd.read_csv(path, low_memory=False)
    raise ValueError(f"unsupported file type: {path}")


def read_glob(path_glob: str) -> pd.DataFrame:
    files = sorted(glob.glob(path_glob))
    if not files:
        raise FileNotFoundError(path_glob)

    parts = []
    for f in files:
        print("[read]", f)
        parts.append(read_any(f))

    return pd.concat(parts, ignore_index=True)


def pick_col(df: pd.DataFrame, candidates, required=True, name="column"):
    for c in candidates:
        if c in df.columns:
            return c

    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]

    if required:
        raise KeyError(f"cannot find {name}; candidates={candidates}; columns={list(df.columns)}")
    return None


def normalize_market(mkt: pd.DataFrame) -> pd.DataFrame:
    date_col = pick_col(mkt, ["date", "execution_date"], name="date")
    dt_col = pick_col(mkt, ["datetime", "execution_datetime", "tsminute", "timestamp"], name="datetime")
    sid_col = pick_col(mkt, ["securityid", "SecurityID", "sid", "symbol"], name="symbol")

    mid_col = pick_col(mkt, ["mid_price", "price", "tmid"], required=False, name="mid price")
    bid_col = pick_col(mkt, ["bid_price", "bid1", "bid", "tbid"], required=False, name="bid price")
    ask_col = pick_col(mkt, ["ask_price", "ask1", "ask", "task"], required=False, name="ask price")
    bench_col = pick_col(
        mkt,
        ["benchmark_weight", "bench_weight", "index_weight", "ew_benchmark_weight"],
        required=False,
        name="benchmark weight",
    )

    if mid_col is None and bid_col is None:
        raise KeyError("need at least one of mid_price/price/tmid or bid_price/bid1/bid/tbid")

    out = pd.DataFrame(
        {
            "date": mkt[date_col].astype(int),
            "datetime": pd.to_datetime(mkt[dt_col]),
            "securityid": mkt[sid_col].astype(str).str.zfill(6),
        }
    )

    if mid_col is not None:
        out["mid_price"] = pd.to_numeric(mkt[mid_col], errors="coerce")
    else:
        out["mid_price"] = pd.to_numeric(mkt[bid_col], errors="coerce")

    if bid_col is not None:
        out["bid_price"] = pd.to_numeric(mkt[bid_col], errors="coerce")
    else:
        out["bid_price"] = out["mid_price"]

    if ask_col is not None:
        out["ask_price"] = pd.to_numeric(mkt[ask_col], errors="coerce")
    else:
        out["ask_price"] = out["mid_price"]

    if bench_col is not None:
        out["benchmark_weight"] = pd.to_numeric(mkt[bench_col], errors="coerce")
    else:
        out["benchmark_weight"] = np.nan

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=["date", "datetime", "securityid", "bid_price"])
    out = out[out["bid_price"] > 0].copy()

    if out.empty:
        raise RuntimeError("normalized market is empty")

    if out["benchmark_weight"].isna().all():
        out["benchmark_weight"] = 1.0 / out.groupby(["date", "datetime"])["securityid"].transform("count")
    else:
        out["benchmark_weight"] = out["benchmark_weight"].fillna(0.0)

    out = out.sort_values(["securityid", "datetime"]).drop_duplicates(
        ["date", "datetime", "securityid"],
        keep="last",
    )

    print("[market normalized]", out.shape)
    print("[market dates]", out["date"].min(), "->", out["date"].max(), "n=", out["date"].nunique())
    print("[market minutes]", out[["date", "datetime"]].drop_duplicates().shape[0])
    print("[market symbols]", out["securityid"].nunique())

    return out.reset_index(drop=True)


def normalize_positions(pos: pd.DataFrame) -> pd.DataFrame:
    date_col = pick_col(pos, ["date"], name="position date")
    dt_col = pick_col(pos, ["datetime"], name="position datetime")
    sid_col = pick_col(pos, ["securityid", "SecurityID", "sid", "symbol"], name="position symbol")

    if "actual_weight_after" in pos.columns:
        w_col = "actual_weight_after"
    elif "actual_weight" in pos.columns:
        w_col = "actual_weight"
    elif "target_weight" in pos.columns:
        w_col = "target_weight"
        print("[WARN] using target_weight because actual_weight_after is missing")
    else:
        raise KeyError("positions need actual_weight_after / actual_weight / target_weight")

    out = pd.DataFrame(
        {
            "date": pos[date_col].astype(int),
            "datetime": pd.to_datetime(pos[dt_col]),
            "securityid": pos[sid_col].astype(str).str.zfill(6),
            "actual_weight_after": pd.to_numeric(pos[w_col], errors="coerce").fillna(0.0),
        }
    )

    out = out.sort_values(["securityid", "datetime"]).drop_duplicates(
        ["date", "datetime", "securityid"],
        keep="last",
    )

    print("[positions normalized]", out.shape)
    print("[position dates]", out["date"].min(), "->", out["date"].max(), "n=", out["date"].nunique())
    print("[position rebalances]", out[["date", "datetime"]].drop_duplicates().shape[0])
    print("[position symbols]", out["securityid"].nunique())
    print("[weight col]", w_col)

    return out.reset_index(drop=True)


def normalize_summary(s: pd.DataFrame) -> pd.DataFrame:
    date_col = pick_col(s, ["date"], name="summary date")
    dt_col = pick_col(s, ["datetime"], name="summary datetime")

    out = pd.DataFrame(
        {
            "date": s[date_col].astype(int),
            "datetime": pd.to_datetime(s[dt_col]),
        }
    )

    if "total_cost" in s.columns:
        out["total_cost"] = pd.to_numeric(s["total_cost"], errors="coerce").fillna(0.0)
    elif {"fee", "spread_cost_est"}.issubset(s.columns):
        out["total_cost"] = (
            pd.to_numeric(s["fee"], errors="coerce").fillna(0.0)
            + pd.to_numeric(s["spread_cost_est"], errors="coerce").fillna(0.0)
        )
    else:
        out["total_cost"] = 0.0

    if "turnover_weight" in s.columns:
        out["turnover_weight"] = pd.to_numeric(s["turnover_weight"], errors="coerce").fillna(0.0)
    elif "turnover_notional" in s.columns:
        out["turnover_weight"] = pd.to_numeric(s["turnover_notional"], errors="coerce").fillna(0.0)
    else:
        out["turnover_weight"] = 0.0

    out = out.groupby(["date", "datetime"], as_index=False).sum(numeric_only=True)
    print("[summary normalized]", out.shape)
    return out


def compound_curve(ret: pd.Series) -> pd.Series:
    ret = pd.Series(ret).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return (1.0 + ret).cumprod() - 1.0


def compound_return(ret: pd.Series) -> float:
    ret = pd.Series(ret).replace([np.inf, -np.inf], np.nan).dropna()
    if ret.empty:
        return np.nan
    return float((1.0 + ret).prod() - 1.0)


def make_ticks(curve: pd.DataFrame):
    first = curve.groupby("date", as_index=False)["bar_index"].min()
    n = len(first)
    step = 1 if n <= 12 else 2 if n <= 24 else max(1, n // 12)
    ticks = first.iloc[::step]
    return ticks["bar_index"].tolist(), ticks["date"].astype(str).tolist()


def build_curve(market: pd.DataFrame, positions: pd.DataFrame, summary: pd.DataFrame, capital: float) -> pd.DataFrame:
    df = market.merge(
        positions,
        on=["date", "datetime", "securityid"],
        how="left",
    )

    df = df.sort_values(["securityid", "datetime"]).reset_index(drop=True)

    # Carry executed actual position across minute grid.
    df["actual_weight_after"] = (
        df.groupby("securityid")["actual_weight_after"]
        .ffill()
        .fillna(0.0)
    )

    # Within-day return only. This excludes overnight return by construction.
    df["ret_1m"] = df.groupby(["date", "securityid"])["bid_price"].pct_change()
    df["ret_1m"] = df["ret_1m"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # PnL uses previous-minute holdings to avoid lookahead.
    df["held_weight_prev"] = (
        df.groupby("securityid")["actual_weight_after"]
        .shift(1)
        .fillna(0.0)
    )

    # Benchmark also uses previous-minute benchmark weight.
    df["benchmark_weight_prev"] = (
        df.groupby("securityid")["benchmark_weight"]
        .shift(1)
        .fillna(0.0)
    )

    # First minute of each date has zero return, so these do not affect PnL.
    first_idx = df.groupby(["date", "securityid"]).head(1).index
    df.loc[first_idx, "held_weight_prev"] = 0.0
    df.loc[first_idx, "benchmark_weight_prev"] = 0.0

    df["strategy_ret_contrib"] = df["held_weight_prev"] * df["ret_1m"]
    df["benchmark_ret_contrib"] = df["benchmark_weight_prev"] * df["ret_1m"]

    curve = (
        df.groupby(["date", "datetime"], as_index=False)
        .agg(
            actual_ret_no_cost=("strategy_ret_contrib", "sum"),
            benchmark_ret=("benchmark_ret_contrib", "sum"),
            actual_gross=("actual_weight_after", "sum"),
            held_gross_prev=("held_weight_prev", "sum"),
            benchmark_gross=("benchmark_weight", "sum"),
            n_hold=("actual_weight_after", lambda x: int((x > 1e-12).sum())),
        )
    )

    # Deduct execution costs at rebalance timestamps.
    cost = summary.copy()
    cost["cost_ret"] = cost["total_cost"] / float(capital)

    curve = curve.merge(
        cost[["date", "datetime", "total_cost", "turnover_weight", "cost_ret"]],
        on=["date", "datetime"],
        how="left",
    )

    curve["total_cost"] = curve["total_cost"].fillna(0.0)
    curve["turnover_weight"] = curve["turnover_weight"].fillna(0.0)
    curve["cost_ret"] = curve["cost_ret"].fillna(0.0)

    curve["actual_ret"] = curve["actual_ret_no_cost"] - curve["cost_ret"]
    curve["alpha_ret"] = curve["actual_ret"] - curve["benchmark_ret"]

    # NAV levels are compounded from minute returns.
    curve["actualret"] = compound_curve(curve["actual_ret"])
    curve["benchmarkret"] = compound_curve(curve["benchmark_ret"])
    curve["alpharet"] = curve["actualret"] - curve["benchmarkret"]
    curve["alpharet_compound"] = compound_curve(curve["alpha_ret"])

    curve = curve.sort_values(["date", "datetime"]).reset_index(drop=True)
    curve["bar_index"] = np.arange(len(curve))

    return curve


def plot_curve(curve: pd.DataFrame, out_dir: Path, tag: str):
    tick_pos, tick_lab = make_ticks(curve)

    daily = (
        curve.groupby("date", as_index=False)
        .agg(
            actual_return=("actual_ret", compound_return),
            benchmark_return=("benchmark_ret", compound_return),
        )
    )
    daily["excess_return"] = daily["actual_return"] - daily["benchmark_return"]
    ex = daily["excess_return"].dropna()

    daily_excess_sharpe = np.nan
    if len(ex) >= 2 and ex.std(ddof=1) > 1e-12:
        daily_excess_sharpe = float(ex.mean() / ex.std(ddof=1) * np.sqrt(252))

    actual = float(curve["actualret"].iloc[-1])
    bench = float(curve["benchmarkret"].iloc[-1])
    alpha = float(curve["alpharet"].iloc[-1])

    stats_text = (
        f"Strategy Return: {actual:.2%}\n"
        f"Benchmark Return: {bench:.2%}\n"
        f"Alpha Return: {alpha:.2%}\n"
        f"Daily Excess Sharpe: {daily_excess_sharpe:.2f}"
    )

    fig = plt.figure(figsize=(18, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=[5.0, 1.25], wspace=0.08)

    ax = fig.add_subplot(gs[0, 0])
    box = fig.add_subplot(gs[0, 1])
    box.axis("off")

    ax.plot(curve["bar_index"], curve["actualret"] * 100.0, label="T+1-aware Strategy NAV", linewidth=1.5)
    ax.plot(curve["bar_index"], curve["benchmarkret"] * 100.0, label="Full Benchmark", linewidth=1.5)
    ax.plot(curve["bar_index"], curve["alpharet"] * 100.0, label="Alpha", linewidth=1.8)

    ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_title("T+1-aware Pure-CS NAV", fontsize=16, fontweight="bold")
    ax.set_xlabel("Trading date")
    ax.set_ylabel("Cumulative return (%)")
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lab, rotation=45, ha="right")
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.legend(loc="lower left", frameon=True)

    box.text(
        0.05,
        0.95,
        "Summary\n\n" + stats_text,
        ha="left",
        va="top",
        fontsize=12,
        linespacing=1.5,
        bbox=dict(boxstyle="round,pad=0.6", facecolor="white", edgecolor="gray", alpha=0.95),
    )

    fig.tight_layout()
    nav_png = out_dir / f"{tag}_nav_curve.png"
    fig.savefig(nav_png, dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(curve["bar_index"], curve["alpharet"] * 100.0, label="Alpha", linewidth=1.8)
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_title("T+1-aware Pure-CS Alpha Curve", fontsize=15, fontweight="bold")
    ax.set_xlabel("Trading date")
    ax.set_ylabel("Cumulative alpha (%)")
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lab, rotation=45, ha="right")
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.legend(loc="lower left")
    fig.tight_layout()
    alpha_png = out_dir / f"{tag}_alpha_curve.png"
    fig.savefig(alpha_png, dpi=200)
    plt.close(fig)

    summary = pd.DataFrame(
        [
            {
                "actual_return": actual,
                "benchmark_return": bench,
                "alpha_return": alpha,
                "daily_excess_sharpe": daily_excess_sharpe,
                "avg_actual_gross": float(curve["actual_gross"].mean()),
                "avg_held_gross_prev": float(curve["held_gross_prev"].mean()),
                "avg_n_hold": float(curve["n_hold"].mean()),
                "total_cost": float(curve["total_cost"].sum()),
                "turnover_weight": float(curve["turnover_weight"].sum()),
                "n_minutes": int(len(curve)),
                "n_days": int(curve["date"].nunique()),
            }
        ]
    )

    summary_csv = out_dir / f"{tag}_plot_summary.csv"
    summary.to_csv(summary_csv, index=False)

    print("\n===== NAV summary =====")
    print(summary.T.to_string(header=False))

    print("\n[saved]")
    print(nav_png)
    print(alpha_png)
    print(summary_csv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market-glob", required=True)
    ap.add_argument("--positions", required=True)
    ap.add_argument("--rebalance-summary", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--capital", type=float, default=200_000_000.0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[read market]")
    market_raw = read_glob(args.market_glob)
    market = normalize_market(market_raw)

    print("[read positions]")
    positions_raw = read_any(args.positions)
    positions = normalize_positions(positions_raw)

    print("[read rebalance summary]")
    summary_raw = read_any(args.rebalance_summary)
    rebalance_summary = normalize_summary(summary_raw)

    print("[build curve]")
    curve = build_curve(market, positions, rebalance_summary, args.capital)

    curve_csv = out_dir / f"{args.tag}_nav_curve.csv"
    curve.to_csv(curve_csv, index=False)
    print("[saved]", curve_csv)

    plot_curve(curve, out_dir, args.tag)


if __name__ == "__main__":
    main()
