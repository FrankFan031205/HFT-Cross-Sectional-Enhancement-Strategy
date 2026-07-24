# -*- coding: utf-8 -*-
import argparse
import glob
import math
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
        raise KeyError(f"cannot find {name}; candidates={candidates}; columns={list(df.columns)}")
    return None


def floor_lot(x, lot_size):
    if not np.isfinite(x) or x <= 0:
        return 0
    return int(math.floor(x / lot_size) * lot_size) if lot_size > 1 else int(math.floor(x))


def compound_return(x):
    x = pd.Series(x).replace([np.inf, -np.inf], np.nan).dropna()
    if x.empty:
        return np.nan
    return float((1.0 + x).prod() - 1.0)


def normalize_market(df):
    date_col = pick_col(df, ["date", "execution_date"], name="date")
    dt_col = pick_col(df, ["datetime", "execution_datetime", "tsminute", "timestamp"], name="datetime")
    sid_col = pick_col(df, ["securityid", "SecurityID", "sid", "symbol"], name="symbol")

    bid_col = pick_col(df, ["bid_price", "bid1", "bid", "tbid"], required=False, name="bid")
    mid_col = pick_col(df, ["mid_price", "price", "tmid"], required=False, name="mid")
    bench_col = pick_col(df, ["benchmark_weight", "bench_weight", "index_weight"], name="benchmark_weight")

    if bid_col is None and mid_col is None:
        raise KeyError("need bid or mid price")

    out = pd.DataFrame({
        "date": df[date_col].astype(int),
        "datetime": pd.to_datetime(df[dt_col]),
        "securityid": df[sid_col].astype(str).str.extract(r"(\d+)")[0].str.zfill(6),
    })

    out["price"] = pd.to_numeric(df[bid_col], errors="coerce") if bid_col else pd.to_numeric(df[mid_col], errors="coerce")
    out["benchmark_weight"] = pd.to_numeric(df[bench_col], errors="coerce").fillna(0.0)

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=["date", "datetime", "securityid", "price"])
    out = out[out["price"] > 0].copy()

    g = out.groupby(["date", "datetime"])["benchmark_weight"].transform("sum")
    n = out.groupby(["date", "datetime"])["securityid"].transform("count")
    out["benchmark_weight"] = np.where(g > 1e-12, out["benchmark_weight"] / g, 1.0 / n)

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
        "securityid": df[sid_col].astype(str).str.extract(r"(\d+)")[0].str.zfill(6),
        "shares_after": pd.to_numeric(df[sh_col], errors="coerce").fillna(0.0),
    })

    out = out.sort_values(["securityid", "datetime"]).drop_duplicates(
        ["date", "datetime", "securityid"], keep="last"
    ).reset_index(drop=True)

    return out


def normalize_summary(df):
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

    return out.groupby(["date", "datetime"], as_index=False).sum(numeric_only=True)


def build_curve(market, positions, summary, capital, init_gross, lot_size):
    first_dt = market["datetime"].min()
    first_market = market[market["datetime"] == first_dt].copy()

    first_market["init_weight"] = first_market["benchmark_weight"] * float(init_gross)

    init_shares = {}
    init_notional = 0.0
    for r in first_market.itertuples(index=False):
        sid = r.securityid
        px = float(r.price)
        w = float(r.init_weight)
        sh = floor_lot(float(capital) * w / px, int(lot_size))
        if sh > 0:
            init_shares[sid] = int(sh)
            init_notional += sh * px

    print("===== benchmark warm start =====")
    print("first_dt:", first_dt)
    print("init_gross:", init_gross)
    print("init_names:", len(init_shares))
    print("init_notional:", init_notional)
    print("init_cash:", float(capital) - init_notional)
    print("init_gross_to_capital:", init_notional / float(capital))

    df = market.merge(positions, on=["date", "datetime", "securityid"], how="left")
    df = df.sort_values(["securityid", "datetime"]).reset_index(drop=True)

    first_reb_dt = positions["datetime"].min()

    df["shares_after"] = df.groupby("securityid")["shares_after"].ffill()
    pre_mask = df["datetime"] < first_reb_dt
    df.loc[pre_mask, "shares_after"] = df.loc[pre_mask, "securityid"].map(init_shares).fillna(0.0)
    df["shares_after"] = df["shares_after"].fillna(df["securityid"].map(init_shares)).fillna(0.0)

    df["shares_prev"] = df.groupby("securityid")["shares_after"].shift(1)

    first_idx = df.groupby(["date", "securityid"]).head(1).index
    df.loc[first_idx, "shares_prev"] = df.loc[first_idx, "shares_after"]

    df["price_prev"] = df.groupby(["date", "securityid"])["price"].shift(1)
    df["dprice"] = df["price"] - df["price_prev"]
    df["ret_1m"] = df.groupby(["date", "securityid"])["price"].pct_change()

    df.loc[first_idx, "dprice"] = 0.0
    df.loc[first_idx, "ret_1m"] = 0.0

    df["dprice"] = df["dprice"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["ret_1m"] = df["ret_1m"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    df["benchmark_weight_prev"] = df.groupby("securityid")["benchmark_weight"].shift(1)
    df.loc[first_idx, "benchmark_weight_prev"] = df.loc[first_idx, "benchmark_weight"]
    df["benchmark_weight_prev"] = df["benchmark_weight_prev"].fillna(0.0)

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

    per_min = per_min.merge(summary[["date", "datetime", "total_cost", "turnover_weight"]],
                            on=["date", "datetime"], how="left")
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
        benchmark_nav *= 1.0 + float(r.benchmark_ret)

        rows.append({
            "date": int(r.date),
            "datetime": pd.to_datetime(r.datetime),
            "equity_before": equity_before,
            "equity": equity,
            "pnl_currency": pnl,
            "total_cost": cost,
            "turnover_weight": float(r.turnover_weight),
            "actual_ret": actual_ret,
            "benchmark_ret": float(r.benchmark_ret),
            "alpha_ret": actual_ret - float(r.benchmark_ret),
            "actualret": equity / float(capital) - 1.0,
            "benchmarkret": benchmark_nav - 1.0,
            "alpharet": equity / float(capital) - 1.0 - (benchmark_nav - 1.0),
            "gross_prev_to_capital": float(r.gross_prev_currency) / float(capital),
            "gross_after_to_capital": float(r.gross_after_currency) / float(capital),
            "gross_prev_to_equity": float(r.gross_prev_currency) / equity_before if equity_before > 0 else 0.0,
            "gross_after_to_equity": float(r.gross_after_currency) / equity if equity > 0 else 0.0,
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
        actual_day=("actual_ret", compound_return),
        benchmark_day=("benchmark_ret", compound_return),
    )
    daily["excess_day"] = daily["actual_day"] - daily["benchmark_day"]

    sharpe = np.nan
    if len(daily) > 1 and daily["excess_day"].std(ddof=1) > 1e-12:
        sharpe = float(daily["excess_day"].mean() / daily["excess_day"].std(ddof=1) * np.sqrt(252))

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
        "turnover_weight": curve["turnover_weight"].sum(),
        "total_cost": curve["total_cost"].sum(),
        "n_minutes": len(curve),
        "n_days": curve["date"].nunique(),
    }])

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    curve_path = out_dir / f"{tag}_nav_curve_benchmark_warmstart_noovernight.csv"
    summary_path = out_dir / f"{tag}_summary_benchmark_warmstart_noovernight.csv"
    png_path = out_dir / f"{tag}_nav_benchmark_warmstart_noovernight.png"

    curve.to_csv(curve_path, index=False)
    summary.to_csv(summary_path, index=False)

    ticks, labels = make_ticks(curve)

    fig = plt.figure(figsize=(18, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=[5.5, 1.3], wspace=0.08)
    ax = fig.add_subplot(gs[0, 0])
    box = fig.add_subplot(gs[0, 1])
    box.axis("off")

    ax.plot(curve["bar_index"], curve["actualret"] * 100, label="strategy actualret")
    ax.plot(curve["bar_index"], curve["benchmarkret"] * 100, label="CSI2000 benchmarkret")
    ax.plot(curve["bar_index"], curve["alpharet"] * 100, label="alpharet")
    ax.axhline(0, linestyle="--", linewidth=0.8)
    ax.grid(True, alpha=0.3)
    ax.set_title("T+1-aware Pure-CS, CSI2000 Benchmark Warm Start, No Overnight")
    ax.set_ylabel("cumulative return (%)")
    ax.set_xlabel("trading minute index")
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend(loc="lower left")

    r = summary.iloc[0]
    text = (
        "Summary\n\n"
        f"Strategy Return\n{r['actual_return']:.2%}\n\n"
        f"Benchmark Return\n{r['benchmark_return']:.2%}\n\n"
        f"Alpha Return\n{r['alpha_return']:.2%}\n\n"
        f"Daily Excess Sharpe\n{r['daily_excess_sharpe']:.2f}"
    )
    box.text(0.02, 0.95, text, va="top", ha="left", fontsize=12,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="0.6"))

    fig.tight_layout()
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print("\n===== benchmark warm-start no-overnight summary =====")
    print(summary.T.to_string(header=False))
    print("\n[saved]")
    print(curve_path)
    print(summary_path)
    print(png_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market-glob", required=True)
    ap.add_argument("--positions", required=True)
    ap.add_argument("--rebalance-summary", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--capital", type=float, default=200000000.0)
    ap.add_argument("--init-gross", type=float, default=0.95)
    ap.add_argument("--lot-size", type=int, default=100)
    args = ap.parse_args()

    market = normalize_market(read_glob(args.market_glob))
    positions = normalize_positions(read_any(args.positions))
    summary = normalize_summary(read_any(args.rebalance_summary))

    print("[market]", market.shape, market["datetime"].min(), market["datetime"].max())
    print("[positions]", positions.shape, positions["datetime"].min(), positions["datetime"].max())
    print("[summary]", summary.shape)

    curve = build_curve(market, positions, summary, args.capital, args.init_gross, args.lot_size)
    plot_curve(curve, args.out_dir, args.tag)


if __name__ == "__main__":
    main()
