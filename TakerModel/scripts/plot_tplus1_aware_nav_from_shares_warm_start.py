# -*- coding: utf-8 -*-
"""
Warm-start NAV from actual shares.

Assumption
----------
The portfolio is already fully established at 09:30 on the first day.
We use the first optimizer position snapshot as the initial holdings at 09:30.
Initial stock build cost is set to zero.
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
    raise ValueError(path)


def read_glob(path_glob):
    files = sorted(glob.glob(path_glob))
    if not files:
        raise FileNotFoundError(path_glob)
    return pd.concat([read_any(f) for f in files], ignore_index=True)


def pick_col(df, candidates, required=True, name="column"):
    for c in candidates:
        if c in df.columns:
            return c
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    if required:
        raise KeyError(f"cannot find {name}: {candidates}; columns={list(df.columns)}")
    return None


def normalize_market(df):
    date_col = pick_col(df, ["date"], name="date")
    dt_col = pick_col(df, ["datetime", "tsminute", "timestamp"], name="datetime")
    sid_col = pick_col(df, ["securityid", "SecurityID", "sid", "symbol"], name="symbol")

    bid_col = pick_col(df, ["bid_price", "bid1", "bid", "tbid"], required=False, name="bid")
    mid_col = pick_col(df, ["mid_price", "price", "tmid"], required=False, name="mid")
    bench_col = pick_col(df, ["benchmark_weight", "bench_weight", "index_weight"], required=False, name="benchmark")

    if bid_col is None and mid_col is None:
        raise KeyError("need bid price or mid price")

    out = pd.DataFrame({
        "date": df[date_col].astype(int),
        "datetime": pd.to_datetime(df[dt_col]),
        "securityid": df[sid_col].astype(str).str.zfill(6),
    })

    out["price"] = pd.to_numeric(df[bid_col if bid_col is not None else mid_col], errors="coerce")

    if bench_col is not None:
        out["benchmark_weight"] = pd.to_numeric(df[bench_col], errors="coerce")
    else:
        out["benchmark_weight"] = np.nan

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=["date", "datetime", "securityid", "price"])
    out = out[out["price"] > 0].copy()

    if out["benchmark_weight"].isna().all():
        out["benchmark_weight"] = 1.0 / out.groupby(["date", "datetime"])["securityid"].transform("count")
    else:
        out["benchmark_weight"] = out["benchmark_weight"].fillna(0.0)

    out = out.sort_values(["securityid", "datetime"]).drop_duplicates(
        ["date", "datetime", "securityid"], keep="last"
    ).reset_index(drop=True)

    return out


def normalize_positions(df):
    date_col = pick_col(df, ["date"], name="date")
    dt_col = pick_col(df, ["datetime"], name="datetime")
    sid_col = pick_col(df, ["securityid", "SecurityID", "sid", "symbol"], name="symbol")
    sh_col = pick_col(df, ["actual_shares_after", "actual_shares", "shares"], name="actual shares")

    out = pd.DataFrame({
        "date": df[date_col].astype(int),
        "datetime": pd.to_datetime(df[dt_col]),
        "securityid": df[sid_col].astype(str).str.zfill(6),
        "shares_after": pd.to_numeric(df[sh_col], errors="coerce").fillna(0.0),
    })

    out = out.sort_values(["securityid", "datetime"]).drop_duplicates(
        ["date", "datetime", "securityid"], keep="last"
    ).reset_index(drop=True)

    return out


def normalize_summary(df, zero_initial_cost=True):
    date_col = pick_col(df, ["date"], name="date")
    dt_col = pick_col(df, ["datetime"], name="datetime")

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
    else:
        out["turnover_weight"] = 0.0

    out = out.groupby(["date", "datetime"], as_index=False).sum(numeric_only=True)
    out = out.sort_values(["date", "datetime"]).reset_index(drop=True)

    if zero_initial_cost and len(out) > 0:
        out.loc[out.index[0], ["total_cost", "turnover_weight"]] = 0.0

    return out


def compound_return(x):
    x = pd.Series(x).replace([np.inf, -np.inf], np.nan).dropna()
    if x.empty:
        return np.nan
    return float((1.0 + x).prod() - 1.0)


def build_curve(market, positions, summary, capital):
    first_reb_dt = positions["datetime"].min()
    first_snapshot = positions[positions["datetime"] == first_reb_dt][["securityid", "shares_after"]]
    init_shares = dict(zip(first_snapshot["securityid"], first_snapshot["shares_after"]))

    df = market.merge(positions, on=["date", "datetime", "securityid"], how="left")
    df = df.sort_values(["securityid", "datetime"]).reset_index(drop=True)

    df["shares_after"] = df.groupby("securityid")["shares_after"].ffill()

    pre_mask = df["datetime"] < first_reb_dt
    if pre_mask.any():
        df.loc[pre_mask, "shares_after"] = df.loc[pre_mask, "securityid"].map(init_shares).fillna(0.0)

    df["shares_after"] = df["shares_after"].fillna(0.0)
    df["shares_prev"] = df.groupby("securityid")["shares_after"].shift(1).fillna(0.0)

    df["price_prev"] = df.groupby(["date", "securityid"])["price"].shift(1)
    df["dprice"] = (df["price"] - df["price_prev"]).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["ret_1m"] = df.groupby(["date", "securityid"])["price"].pct_change()
    df["ret_1m"] = df["ret_1m"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["benchmark_weight_prev"] = df.groupby("securityid")["benchmark_weight"].shift(1).fillna(0.0)

    first_idx = df.groupby(["date", "securityid"]).head(1).index
    df.loc[first_idx, "shares_prev"] = 0.0
    df.loc[first_idx, "dprice"] = 0.0
    df.loc[first_idx, "ret_1m"] = 0.0
    df.loc[first_idx, "benchmark_weight_prev"] = 0.0

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

    per_min = per_min.merge(summary[["date", "datetime", "total_cost", "turnover_weight"]], on=["date", "datetime"], how="left")
    per_min["total_cost"] = per_min["total_cost"].fillna(0.0)
    per_min["turnover_weight"] = per_min["turnover_weight"].fillna(0.0)

    equity = float(capital)
    benchmark_nav = 1.0
    rows = []

    for r in per_min.itertuples(index=False):
        equity_before = equity
        pnl = float(r.pnl_currency)
        cost_now = float(r.total_cost)

        actual_ret = (pnl - cost_now) / equity_before if equity_before > 0 else 0.0
        equity = equity_before + pnl - cost_now
        benchmark_nav *= (1.0 + float(r.benchmark_ret))

        gross_prev_to_equity = float(r.gross_prev_currency) / equity_before if equity_before > 0 else 0.0
        gross_after_to_equity = float(r.gross_after_currency) / equity if equity > 0 else 0.0

        rows.append({
            "date": int(r.date),
            "datetime": pd.to_datetime(r.datetime),
            "equity_before": equity_before,
            "equity": equity,
            "pnl_currency": pnl,
            "total_cost": cost_now,
            "turnover_weight": float(r.turnover_weight),
            "actual_ret": actual_ret,
            "benchmark_ret": float(r.benchmark_ret),
            "alpha_ret": actual_ret - float(r.benchmark_ret),
            "actualret": equity / capital - 1.0,
            "benchmarkret": benchmark_nav - 1.0,
            "alpharet": equity / capital - 1.0 - (benchmark_nav - 1.0),
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
    daily = curve.groupby("date", as_index=False).agg(
        actual_return=("actual_ret", compound_return),
        benchmark_return=("benchmark_ret", compound_return),
    )
    daily["excess_return"] = daily["actual_return"] - daily["benchmark_return"]
    ex = daily["excess_return"].dropna()

    sharpe = np.nan
    if len(ex) >= 2 and ex.std(ddof=1) > 1e-12:
        sharpe = float(ex.mean() / ex.std(ddof=1) * np.sqrt(252))

    actual = compound_return(curve["actual_ret"])
    bench = compound_return(curve["benchmark_ret"])
    alpha = actual - bench

    summary = pd.DataFrame([{
        "start_datetime": str(curve["datetime"].iloc[0]),
        "end_datetime": str(curve["datetime"].iloc[-1]),
        "actual_return": actual,
        "benchmark_return": bench,
        "alpha_return": alpha,
        "daily_excess_sharpe": sharpe,
        "avg_gross_prev_to_capital": curve["gross_prev_to_capital"].mean(),
        "avg_gross_prev_to_equity": curve["gross_prev_to_equity"].mean(),
        "avg_gross_after_to_equity": curve["gross_after_to_equity"].mean(),
        "avg_n_hold": curve["n_hold"].mean(),
        "total_cost": curve["total_cost"].sum(),
        "turnover_weight": curve["turnover_weight"].sum(),
        "n_minutes": len(curve),
        "n_days": curve["date"].nunique(),
    }])

    fig = plt.figure(figsize=(18, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=[5.5, 1.3], wspace=0.08)
    ax = fig.add_subplot(gs[0, 0])
    box = fig.add_subplot(gs[0, 1])
    box.axis("off")

    ax.plot(curve["bar_index"], curve["actualret"] * 100, label="stock-only actualret")
    ax.plot(curve["bar_index"], curve["benchmarkret"] * 100, label="benchmarkret")
    ax.plot(curve["bar_index"], curve["alpharet"] * 100, label="stock-only alpharet")
    ax.axhline(0, linestyle="--", linewidth=0.8)
    ax.grid(True, alpha=0.3)
    ax.set_title("T+1-aware Pure-CS Stock-only NAV, Warm Start at 09:30")
    ax.set_ylabel("cumulative return (%)")
    ax.set_xlabel("trading minute index, assuming full initial position at 09:30")

    ticks, labels = make_ticks(curve)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend(loc="lower left")

    row = summary.iloc[0]
    text = (
        "Summary\n\n"
        f"Strategy Return\n{row['actual_return']:.2%}\n\n"
        f"Benchmark Return\n{row['benchmark_return']:.2%}\n\n"
        f"Alpha Return\n{row['alpha_return']:.2%}\n\n"
        f"Daily Excess Sharpe\n{row['daily_excess_sharpe']:.2f}"
    )
    box.text(0.02, 0.95, text, va="top", ha="left", fontsize=12,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="0.6"))

    curve_out = out_dir / f"{tag}_nav_curve_from_shares_warm_start.csv"
    summary_out = out_dir / f"{tag}_plot_summary_from_shares_warm_start.csv"
    png_out = out_dir / f"{tag}_nav_curve_from_shares_warm_start.png"

    curve.to_csv(curve_out, index=False)
    summary.to_csv(summary_out, index=False)
    fig.tight_layout()
    fig.savefig(png_out, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print("===== warm-start stock-only summary =====")
    print(summary.T.to_string(header=False))
    print("\n[saved]")
    print(curve_out)
    print(summary_out)
    print(png_out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market-glob", required=True)
    ap.add_argument("--positions", required=True)
    ap.add_argument("--rebalance-summary", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--capital", type=float, default=200000000.0)
    ap.add_argument("--zero-initial-cost", type=int, default=1)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    market = normalize_market(read_glob(args.market_glob))
    positions = normalize_positions(read_any(args.positions))
    summary = normalize_summary(read_any(args.rebalance_summary), zero_initial_cost=bool(args.zero_initial_cost))
    curve = build_curve(market, positions, summary, args.capital)
    plot_curve(curve, out_dir, args.tag)


if __name__ == "__main__":
    main()
