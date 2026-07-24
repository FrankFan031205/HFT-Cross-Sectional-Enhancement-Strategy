# -*- coding: utf-8 -*-
"""
Plot NAV for T+1-aware pure-CS optimizer from actual executed shares.

Why this script exists
----------------------
The earlier plotting script used actual_weight_after directly. That weight was
computed against the initial capital, so after losses it could overstate the
cash-drag problem. This script instead uses actual_shares_after and calculates
PnL in currency:

    PnL_t = shares_{t-1} * (price_t - price_{t-1}) - trading_cost_t
    return_t = PnL_t / equity_{t-1}

It excludes overnight return by using within-day price changes only.
"""

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_any(path):
    path = Path(path)
    suf = "".join(path.suffixes).lower()
    if suf.endswith(".parquet"):
        return pd.read_parquet(path)
    if suf.endswith(".csv") or suf.endswith(".csv.gz"):
        return pd.read_csv(path, low_memory=False)
    raise ValueError(f"unsupported file type: {path}")


def read_glob(path_glob):
    files = sorted(glob.glob(path_glob))
    if not files:
        raise FileNotFoundError(path_glob)
    parts = []
    for f in files:
        print("[read]", f)
        parts.append(read_any(f))
    return pd.concat(parts, ignore_index=True)


def pick_col(df, candidates, required=True, name="column"):
    for c in candidates:
        if c in df.columns:
            return c
    lower = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    if required:
        raise KeyError(f"cannot find {name}; candidates={candidates}; columns={list(df.columns)}")
    return None


def normalize_market(df):
    date_col = pick_col(df, ["date", "execution_date"], name="date")
    dt_col = pick_col(df, ["datetime", "execution_datetime", "tsminute", "timestamp"], name="datetime")
    sid_col = pick_col(df, ["securityid", "SecurityID", "sid", "symbol"], name="symbol")

    bid_col = pick_col(df, ["bid_price", "bid1", "bid", "tbid"], required=False, name="bid")
    mid_col = pick_col(df, ["mid_price", "price", "tmid"], required=False, name="mid")
    bench_col = pick_col(df, ["benchmark_weight", "bench_weight", "index_weight", "ew_benchmark_weight"], required=False, name="benchmark_weight")

    if bid_col is None and mid_col is None:
        raise KeyError("need bid_price/bid1/tbid or mid_price/price/tmid")

    out = pd.DataFrame({
        "date": df[date_col].astype(int),
        "datetime": pd.to_datetime(df[dt_col]),
        "securityid": df[sid_col].astype(str).str.zfill(6),
    })

    if bid_col is not None:
        out["price"] = pd.to_numeric(df[bid_col], errors="coerce")
    else:
        out["price"] = pd.to_numeric(df[mid_col], errors="coerce")

    if bench_col is not None:
        out["benchmark_weight"] = pd.to_numeric(df[bench_col], errors="coerce")
    else:
        out["benchmark_weight"] = np.nan

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=["date", "datetime", "securityid", "price"])
    out = out[out["price"] > 0].copy()

    if out.empty:
        raise RuntimeError("market data is empty after normalization")

    if out["benchmark_weight"].isna().all():
        out["benchmark_weight"] = 1.0 / out.groupby(["date", "datetime"])["securityid"].transform("count")
    else:
        out["benchmark_weight"] = out["benchmark_weight"].fillna(0.0)

    out = out.sort_values(["securityid", "datetime"]).drop_duplicates(
        ["date", "datetime", "securityid"], keep="last"
    )

    print("[market normalized]", out.shape)
    print("[market dates]", out["date"].min(), "->", out["date"].max(), "n=", out["date"].nunique())
    print("[market minutes]", out[["date", "datetime"]].drop_duplicates().shape[0])
    print("[market symbols]", out["securityid"].nunique())
    return out.reset_index(drop=True)


def normalize_positions(df):
    date_col = pick_col(df, ["date"], name="position date")
    dt_col = pick_col(df, ["datetime"], name="position datetime")
    sid_col = pick_col(df, ["securityid", "SecurityID", "sid", "symbol"], name="position symbol")
    sh_col = pick_col(df, ["actual_shares_after", "actual_shares", "shares", "position_shares"], name="actual shares")

    out = pd.DataFrame({
        "date": df[date_col].astype(int),
        "datetime": pd.to_datetime(df[dt_col]),
        "securityid": df[sid_col].astype(str).str.zfill(6),
        "shares_after": pd.to_numeric(df[sh_col], errors="coerce").fillna(0.0),
    })

    out = out.sort_values(["securityid", "datetime"]).drop_duplicates(
        ["date", "datetime", "securityid"], keep="last"
    )

    print("[positions normalized]", out.shape)
    print("[position dates]", out["date"].min(), "->", out["date"].max(), "n=", out["date"].nunique())
    print("[position rebalances]", out[["date", "datetime"]].drop_duplicates().shape[0])
    print("[position symbols]", out["securityid"].nunique())
    print("[share col]", sh_col)
    return out.reset_index(drop=True)


def normalize_summary(df):
    date_col = pick_col(df, ["date"], name="summary date")
    dt_col = pick_col(df, ["datetime"], name="summary datetime")

    out = pd.DataFrame({
        "date": df[date_col].astype(int),
        "datetime": pd.to_datetime(df[dt_col]),
    })

    if "total_cost" in df.columns:
        out["total_cost"] = pd.to_numeric(df["total_cost"], errors="coerce").fillna(0.0)
    elif {"fee", "spread_cost_est"}.issubset(df.columns):
        out["total_cost"] = (
            pd.to_numeric(df["fee"], errors="coerce").fillna(0.0)
            + pd.to_numeric(df["spread_cost_est"], errors="coerce").fillna(0.0)
        )
    else:
        out["total_cost"] = 0.0

    if "turnover_weight" in df.columns:
        out["turnover_weight"] = pd.to_numeric(df["turnover_weight"], errors="coerce").fillna(0.0)
    elif "turnover_notional" in df.columns:
        # If only notional exists, caller must pass capital for interpretation in plot summary.
        out["turnover_weight"] = pd.to_numeric(df["turnover_notional"], errors="coerce").fillna(0.0)
    else:
        out["turnover_weight"] = 0.0

    out = out.groupby(["date", "datetime"], as_index=False).sum(numeric_only=True)
    print("[rebalance summary normalized]", out.shape)
    return out


def compound_return(x):
    x = pd.Series(x).replace([np.inf, -np.inf], np.nan).dropna()
    if x.empty:
        return np.nan
    return float((1.0 + x).prod() - 1.0)


def build_curve(market, positions, summary, capital):
    df = market.merge(positions, on=["date", "datetime", "securityid"], how="left")
    df = df.sort_values(["securityid", "datetime"]).reset_index(drop=True)

    # Carry actual executed shares after each rebalance.
    df["shares_after"] = df.groupby("securityid")["shares_after"].ffill().fillna(0.0)

    # PnL for current minute uses previous-minute shares.
    df["shares_prev"] = df.groupby("securityid")["shares_after"].shift(1).fillna(0.0)

    # Within-day price changes only: no overnight PnL.
    df["price_prev"] = df.groupby(["date", "securityid"])["price"].shift(1)
    df["dprice"] = (df["price"] - df["price_prev"]).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["ret_1m"] = df.groupby(["date", "securityid"])["price"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)

    df["benchmark_weight_prev"] = df.groupby("securityid")["benchmark_weight"].shift(1).fillna(0.0)

    first_idx = df.groupby(["date", "securityid"]).head(1).index
    df.loc[first_idx, "shares_prev"] = 0.0
    df.loc[first_idx, "benchmark_weight_prev"] = 0.0
    df.loc[first_idx, "dprice"] = 0.0
    df.loc[first_idx, "ret_1m"] = 0.0

    df["pnl_currency"] = df["shares_prev"] * df["dprice"]
    df["benchmark_ret_contrib"] = df["benchmark_weight_prev"] * df["ret_1m"]
    df["gross_prev_currency"] = np.abs(df["shares_prev"] * df["price"])
    df["gross_after_currency"] = np.abs(df["shares_after"] * df["price"])

    per_min = (
        df.groupby(["date", "datetime"], as_index=False)
        .agg(
            pnl_currency=("pnl_currency", "sum"),
            benchmark_ret=("benchmark_ret_contrib", "sum"),
            gross_prev_currency=("gross_prev_currency", "sum"),
            gross_after_currency=("gross_after_currency", "sum"),
            n_hold=("shares_after", lambda x: int((x > 0).sum())),
            benchmark_gross=("benchmark_weight", "sum"),
        )
        .sort_values(["date", "datetime"])
        .reset_index(drop=True)
    )

    per_min = per_min.merge(
        summary[["date", "datetime", "total_cost", "turnover_weight"]],
        on=["date", "datetime"],
        how="left",
    )
    per_min["total_cost"] = per_min["total_cost"].fillna(0.0)
    per_min["turnover_weight"] = per_min["turnover_weight"].fillna(0.0)

    equity = float(capital)
    benchmark_nav = 1.0
    rows = []

    for r in per_min.itertuples(index=False):
        equity_before = equity
        pnl = float(r.pnl_currency)
        cost = float(r.total_cost)
        actual_ret = (pnl - cost) / equity_before if equity_before > 0 else 0.0
        equity = equity_before + pnl - cost

        bench_ret = float(r.benchmark_ret)
        benchmark_nav *= (1.0 + bench_ret)

        actualret = equity / capital - 1.0
        benchmarkret = benchmark_nav - 1.0
        alpharet = actualret - benchmarkret

        gross_prev_to_equity = float(r.gross_prev_currency) / equity_before if equity_before > 0 else 0.0
        gross_after_to_equity = float(r.gross_after_currency) / equity if equity > 0 else 0.0

        rows.append({
            "date": int(r.date),
            "datetime": pd.to_datetime(r.datetime),
            "equity_before": equity_before,
            "equity": equity,
            "pnl_currency": pnl,
            "total_cost": cost,
            "turnover_weight": float(r.turnover_weight),
            "actual_ret": actual_ret,
            "benchmark_ret": bench_ret,
            "alpha_ret": actual_ret - bench_ret,
            "actualret": actualret,
            "benchmarkret": benchmarkret,
            "alpharet": alpharet,
            "gross_prev_to_capital": float(r.gross_prev_currency) / capital,
            "gross_after_to_capital": float(r.gross_after_currency) / capital,
            "gross_prev_to_equity": gross_prev_to_equity,
            "gross_after_to_equity": gross_after_to_equity,
            "n_hold": int(r.n_hold),
            "benchmark_gross": float(r.benchmark_gross),
        })

    curve = pd.DataFrame(rows)
    curve["bar_index"] = np.arange(len(curve))
    return curve


def make_ticks(curve):
    first = curve.groupby("date", as_index=False)["bar_index"].min()
    step = 1 if len(first) <= 12 else 2 if len(first) <= 24 else max(1, len(first) // 12)
    ticks = first.iloc[::step]
    return ticks["bar_index"].tolist(), ticks["date"].astype(str).tolist()


def plot_curve(curve, out_dir, tag):
    tick_pos, tick_lab = make_ticks(curve)

    daily = curve.groupby("date", as_index=False).agg(
        actual_return=("actual_ret", compound_return),
        benchmark_return=("benchmark_ret", compound_return),
    )
    daily["excess_return"] = daily["actual_return"] - daily["benchmark_return"]
    ex = daily["excess_return"].dropna()
    sharpe = np.nan
    if len(ex) >= 2 and ex.std(ddof=1) > 1e-12:
        sharpe = float(ex.mean() / ex.std(ddof=1) * np.sqrt(252))

    actual = float(curve["actualret"].iloc[-1])
    bench = float(curve["benchmarkret"].iloc[-1])
    alpha = float(curve["alpharet"].iloc[-1])

    stats = (
        f"Strategy Return: {actual:.2%}\n"
        f"Benchmark Return: {bench:.2%}\n"
        f"Alpha Return: {alpha:.2%}\n"
        f"Daily Excess Sharpe: {sharpe:.2f}"
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
    ax.set_title("T+1-aware Pure-CS NAV from Actual Shares", fontsize=16, fontweight="bold")
    ax.set_xlabel("Trading date")
    ax.set_ylabel("Cumulative return (%)")
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lab, rotation=45, ha="right")
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.legend(loc="lower left")

    box.text(
        0.05, 0.95,
        "Summary\n\n" + stats,
        ha="left", va="top", fontsize=12, linespacing=1.5,
        bbox=dict(boxstyle="round,pad=0.6", facecolor="white", edgecolor="gray", alpha=0.95),
    )

    fig.tight_layout()
    nav_png = out_dir / f"{tag}_nav_curve_from_shares.png"
    fig.savefig(nav_png, dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(curve["bar_index"], curve["alpharet"] * 100.0, label="Alpha", linewidth=1.8)
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_title("T+1-aware Alpha Curve from Actual Shares", fontsize=15, fontweight="bold")
    ax.set_xlabel("Trading date")
    ax.set_ylabel("Cumulative alpha (%)")
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lab, rotation=45, ha="right")
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.legend(loc="lower left")
    fig.tight_layout()
    alpha_png = out_dir / f"{tag}_alpha_curve_from_shares.png"
    fig.savefig(alpha_png, dpi=200)
    plt.close(fig)

    summary = pd.DataFrame([{
        "actual_return": actual,
        "benchmark_return": bench,
        "alpha_return": alpha,
        "daily_excess_sharpe": sharpe,
        "avg_gross_prev_to_capital": float(curve["gross_prev_to_capital"].mean()),
        "avg_gross_prev_to_equity": float(curve["gross_prev_to_equity"].mean()),
        "avg_gross_after_to_equity": float(curve["gross_after_to_equity"].mean()),
        "avg_n_hold": float(curve["n_hold"].mean()),
        "total_cost": float(curve["total_cost"].sum()),
        "turnover_weight": float(curve["turnover_weight"].sum()),
        "n_minutes": int(len(curve)),
        "n_days": int(curve["date"].nunique()),
    }])

    summary_csv = out_dir / f"{tag}_plot_summary_from_shares.csv"
    summary.to_csv(summary_csv, index=False)

    print("\n===== NAV from shares summary =====")
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
    ap.add_argument("--capital", type=float, default=200000000.0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[read market]")
    market = normalize_market(read_glob(args.market_glob))
    print("[read positions]")
    positions = normalize_positions(read_any(args.positions))
    print("[read summary]")
    summary = normalize_summary(read_any(args.rebalance_summary))
    print("[build curve]")
    curve = build_curve(market, positions, summary, args.capital)

    curve_csv = out_dir / f"{args.tag}_nav_curve_from_shares.csv"
    curve.to_csv(curve_csv, index=False)
    print("[saved]", curve_csv)

    plot_curve(curve, out_dir, args.tag)


if __name__ == "__main__":
    main()
