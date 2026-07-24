#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def parse_dt(s):
    s = s.astype(str).str.strip()
    dt = pd.to_datetime(s, errors="coerce")

    bad = dt.isna()
    if bad.any():
        x = s[bad]
        m = x.str.extract(r"(?P<date>\d{8})[_\sT]?(?P<time>\d{6,9})")
        ok = m["date"].notna() & m["time"].notna()

        if ok.any():
            date_part = m.loc[ok, "date"]
            time_part = m.loc[ok, "time"].astype(str)

            hhmmss = time_part.str.slice(0, 6)
            frac = time_part.str.slice(6)

            base = pd.to_datetime(
                date_part + hhmmss,
                format="%Y%m%d%H%M%S",
                errors="coerce",
            )

            ns = frac.apply(lambda z: int(str(z).ljust(9, "0")[:9]) if str(z) else 0)
            parsed = base + pd.to_timedelta(ns.values, unit="ns")

            dt.loc[x.index[ok]] = parsed.values

    return dt


def normalize_side(s):
    x = s.astype(str).str.upper()
    x = x.replace({
        "0": "BUY",
        "1": "SELL",
        "B": "BUY",
        "S": "SELL",
        "BUY": "BUY",
        "SELL": "SELL",
    })
    return x


def make_checkpoints(dates, freq="30min"):
    rows = []

    for d in sorted(dates):
        day = pd.to_datetime(str(d), format="%Y%m%d")

        morning = pd.date_range(
            day + pd.Timedelta(hours=9, minutes=30),
            day + pd.Timedelta(hours=11, minutes=30),
            freq=freq,
        )

        afternoon = pd.date_range(
            day + pd.Timedelta(hours=13, minutes=0),
            day + pd.Timedelta(hours=15, minutes=0),
            freq=freq,
        )

        for t in list(morning) + list(afternoon):
            rows.append({"checkpoint_time": t})

    return pd.DataFrame(rows)


def read_summary(summary_path):
    if summary_path is None:
        return {}

    p = Path(summary_path)
    if not p.exists():
        return {}

    df = pd.read_csv(p)
    if df.empty:
        return {}

    return df.iloc[0].to_dict()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trades", required=True, help="accepted trades pnl csv")
    parser.add_argument("--summary", default=None, help="optional portfolio summary csv")
    parser.add_argument("--output-dir", default="outputs/metrics/portfolio_30min_curves")
    parser.add_argument("--initial-position-per-symbol", type=float, default=5000.0)
    parser.add_argument("--freq", default="30min")
    parser.add_argument("--time-col", default=None)
    parser.add_argument("--symbol-col", default=None)
    parser.add_argument("--side-col", default=None)
    parser.add_argument("--price-col", default="fill_price")
    parser.add_argument("--qty-col", default="fill_qty")
    parser.add_argument("--fee-col", default="fee")
    parser.add_argument("--mark-col", default=None)
    parser.add_argument("--title", default=None)
    parser.add_argument("--max-xticks", type=int, default=60)
    args = parser.parse_args()

    trades_path = Path(args.trades)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(trades_path, low_memory=False)

    time_col = args.time_col or find_col(df, ["fill_time", "decision_time", "datetime", "timestamp", "time"])
    symbol_col = args.symbol_col or find_col(df, ["securityid", "symbol", "SecurityID"])
    side_col = args.side_col or find_col(df, ["side", "trade_side"])

    if time_col is None:
        raise ValueError("Cannot find time column")
    if symbol_col is None:
        raise ValueError("Cannot find symbol column")
    if side_col is None:
        raise ValueError("Cannot find side column")
    if args.price_col not in df.columns:
        raise ValueError(f"Cannot find price column: {args.price_col}")
    if args.qty_col not in df.columns:
        raise ValueError(f"Cannot find qty column: {args.qty_col}")

    if args.mark_col is not None:
        mark_col = args.mark_col
    else:
        mark_col = find_col(df, ["decision_mid", "future_mid", "mid_price", "quote_fair_price", "fair_price"])

    if mark_col is None:
        raise ValueError("Cannot find mark column. Try --mark-col decision_mid")

    df["_dt"] = parse_dt(df[time_col])
    df["_symbol"] = df[symbol_col].astype(str).str.zfill(6)
    df["_side"] = normalize_side(df[side_col])
    df["_price"] = pd.to_numeric(df[args.price_col], errors="coerce")
    df["_qty"] = pd.to_numeric(df[args.qty_col], errors="coerce")
    df["_mark"] = pd.to_numeric(df[mark_col], errors="coerce")

    if args.fee_col in df.columns:
        df["_fee"] = pd.to_numeric(df[args.fee_col], errors="coerce").fillna(0.0)
    else:
        df["_fee"] = 0.0

    df = df[
        df["_dt"].notna()
        & df["_symbol"].notna()
        & df["_side"].isin(["BUY", "SELL"])
        & df["_price"].notna()
        & df["_qty"].notna()
        & df["_mark"].notna()
    ].copy()

    df = df.sort_values("_dt").reset_index(drop=True)
    df["_date"] = df["_dt"].dt.strftime("%Y%m%d")

    symbols = sorted(df["_symbol"].unique())
    dates = sorted(df["_date"].unique())

    checkpoints = make_checkpoints(dates, freq=args.freq)
    checkpoints = checkpoints.sort_values("checkpoint_time").reset_index(drop=True)

    first_mark = (
        df.sort_values("_dt")
        .groupby("_symbol")["_mark"]
        .first()
        .to_dict()
    )

    position = {s: float(args.initial_position_per_symbol) for s in symbols}
    last_mark = {s: float(first_mark[s]) for s in symbols}

    cash = -sum(position[s] * last_mark[s] for s in symbols)
    initial_inventory_value = sum(position[s] * last_mark[s] for s in symbols)
    initial_equity = cash + initial_inventory_value

    rows = []
    trade_idx = 0
    n = len(df)

    for cp in checkpoints["checkpoint_time"]:
        while trade_idx < n and df.loc[trade_idx, "_dt"] <= cp:
            r = df.loc[trade_idx]

            sym = r["_symbol"]
            side = r["_side"]
            price = float(r["_price"])
            qty = float(r["_qty"])
            fee = float(r["_fee"])
            mark = float(r["_mark"])

            last_mark[sym] = mark

            if side == "BUY":
                cash -= price * qty
                cash -= fee
                position[sym] = position.get(sym, 0.0) + qty
            elif side == "SELL":
                cash += price * qty
                cash -= fee
                position[sym] = position.get(sym, 0.0) - qty

            trade_idx += 1

        inventory_value = sum(position[s] * last_mark[s] for s in symbols)
        gross_exposure = sum(abs(position[s] * last_mark[s]) for s in symbols)
        net_exposure = sum(position[s] * last_mark[s] for s in symbols)
        equity = cash + inventory_value
        pnl = equity - initial_equity

        rows.append({
            "datetime": cp,
            "cash": cash,
            "inventory_value": inventory_value,
            "equity": equity,
            "pnl": pnl,
            "gross_exposure": gross_exposure,
            "net_exposure": net_exposure,
            "num_symbols": len(symbols),
            "num_long_symbols": sum(1 for s in symbols if position[s] > 0),
            "num_short_symbols": sum(1 for s in symbols if position[s] < 0),
            "processed_trades": trade_idx,
        })

    out = pd.DataFrame(rows)
    out["period_pnl"] = out["pnl"].diff()
    out.loc[out.index[0], "period_pnl"] = out.loc[out.index[0], "pnl"]

    out["running_max_pnl"] = out["pnl"].cummax()
    out["drawdown"] = out["pnl"] - out["running_max_pnl"]

    out["trading_index"] = range(len(out))
    out["label"] = pd.to_datetime(out["datetime"]).dt.strftime("%Y-%m-%d %H:%M")

    summary = read_summary(args.summary)

    final_pnl_reconstructed = float(out["pnl"].iloc[-1])
    max_drawdown_reconstructed = float(out["drawdown"].min())
    max_gross_reconstructed = float(out["gross_exposure"].max())

    summary_total_pnl = summary.get("total_pnl", np.nan)
    summary_max_dd = summary.get("max_drawdown", np.nan)
    summary_max_gross = summary.get("max_gross_exposure", np.nan)

    stem = trades_path.stem
    out_csv = out_dir / f"{stem}_30min_portfolio_equity.csv"
    out_curve = out_dir / f"{stem}_30min_portfolio_equity_curve.png"
    out_bar = out_dir / f"{stem}_30min_portfolio_period_pnl_bar.png"
    out_summary = out_dir / f"{stem}_30min_portfolio_summary.csv"

    out.to_csv(out_csv, index=False)

    tick_step = max(1, int(np.ceil(len(out) / max(1, args.max_xticks))))
    tick_idx = list(range(0, len(out), tick_step))
    if len(out) - 1 not in tick_idx:
        tick_idx.append(len(out) - 1)

    plt.figure(figsize=(20, 6))
    plt.plot(out["trading_index"], out["pnl"], marker="o", linewidth=1.4)
    plt.title(args.title or f"30min Portfolio-Level Equity Curve: {stem}")
    plt.xlabel("Trading time")
    plt.ylabel("Portfolio PnL = cash + inventory value - initial equity")
    plt.xticks(
        out.loc[tick_idx, "trading_index"],
        out.loc[tick_idx, "label"],
        rotation=45,
        ha="right",
    )
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_curve, dpi=160)
    plt.close()

    plt.figure(figsize=(20, 6))
    plt.bar(out["trading_index"], out["period_pnl"])
    plt.title(args.title or f"30min Portfolio-Level Period PnL: {stem}")
    plt.xlabel("Trading time")
    plt.ylabel("30min portfolio PnL")
    plt.xticks(
        out.loc[tick_idx, "trading_index"],
        out.loc[tick_idx, "label"],
        rotation=45,
        ha="right",
    )
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_bar, dpi=160)
    plt.close()

    report = pd.DataFrame([{
        "trades_file": str(trades_path),
        "summary_file": str(args.summary) if args.summary else "",
        "time_col": time_col,
        "symbol_col": symbol_col,
        "side_col": side_col,
        "price_col": args.price_col,
        "qty_col": args.qty_col,
        "fee_col": args.fee_col,
        "mark_col": mark_col,
        "initial_position_per_symbol": args.initial_position_per_symbol,
        "num_symbols": len(symbols),
        "num_trades": len(df),
        "num_checkpoints": len(out),
        "first_checkpoint": out["datetime"].iloc[0],
        "last_checkpoint": out["datetime"].iloc[-1],
        "initial_equity": initial_equity,
        "final_pnl_reconstructed": final_pnl_reconstructed,
        "max_drawdown_reconstructed": max_drawdown_reconstructed,
        "max_gross_exposure_reconstructed": max_gross_reconstructed,
        "summary_total_pnl": summary_total_pnl,
        "summary_max_drawdown": summary_max_dd,
        "summary_max_gross_exposure": summary_max_gross,
        "pnl_diff_vs_summary": final_pnl_reconstructed - float(summary_total_pnl) if pd.notna(summary_total_pnl) else np.nan,
    }])

    report.to_csv(out_summary, index=False)

    print("input trades:", trades_path)
    print("summary:", args.summary)
    print("rows:", len(df))
    print("symbols:", len(symbols))
    print("mark_col:", mark_col)
    print("checkpoints:", len(out))
    print("saved csv:", out_csv)
    print("saved curve:", out_curve)
    print("saved bar:", out_bar)
    print("saved summary:", out_summary)
    print("")
    print("===== reconstructed summary =====")
    print(report.T.to_string(header=False))


if __name__ == "__main__":
    main()
