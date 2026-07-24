# -*- coding: utf-8 -*-
"""
Pure-CS T+1 TakerModel.

Evaluate optimizer target weights under A-share T+1 sell constraint.

Main logic:
- target_weight is desired position from optimizer
- actual_shares is what we really hold
- sellable_shares is yesterday's position that can be sold today
- buy today is not sellable until next trading day
- sell is clipped by sellable_shares
- NAV uses actual position, not target position
- buy at ask/task, sell at bid/tbid
- fee_bps is charged on traded notional, e.g. 10bp => --fee-bps 10.0
"""

import argparse
from pathlib import Path
import math
import numpy as np
import pandas as pd


def read_table(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    suf = "".join(path.suffixes).lower()
    if suf.endswith(".parquet"):
        return pd.read_parquet(path)
    if suf.endswith(".csv") or suf.endswith(".csv.gz"):
        return pd.read_csv(path, low_memory=False)
    raise ValueError(f"unsupported input format: {path}")


def pick_col(df, candidates, required=True, name="column"):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"cannot find {name}; candidates={candidates}; columns={list(df.columns)}")
    return None


def fnum(x, default=np.nan):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def floor_lot(x, lot):
    if not np.isfinite(x) or x <= 0:
        return 0
    if lot <= 1:
        return int(math.floor(x))
    return int(math.floor(x / lot) * lot)


def normalize_input(raw, args):
    date_col = args.date_col or pick_col(raw, ["date", "execution_date"], name="date")
    dt_col = args.datetime_col or pick_col(raw, ["datetime", "execution_datetime", "timestamp"], name="datetime")
    sid_col = args.symbol_col or pick_col(raw, ["securityid", "SecurityID", "sid", "symbol"], name="symbol")
    w_col = args.weight_col or pick_col(
        raw,
        ["target_weight", "effective_target_weight", "weight", "w", "opt_weight", "optimized_weight"],
        name="target weight",
    )

    mid_col = args.mid_col or pick_col(raw, ["mid_price", "price", "tmid"], required=False, name="mid price")
    bid_col = args.bid_col or pick_col(raw, ["bid_price", "bid1", "bid", "tbid"], required=False, name="bid")
    ask_col = args.ask_col or pick_col(raw, ["ask_price", "ask1", "ask", "task"], required=False, name="ask")
    bench_col = args.benchmark_col or pick_col(
        raw, ["benchmark_weight", "bench_weight", "index_weight", "ew_benchmark_weight"],
        required=False, name="benchmark weight"
    )

    if mid_col is None and (bid_col is None or ask_col is None):
        raise KeyError("need either mid_price or both bid/ask columns")

    df = pd.DataFrame({
        "date": raw[date_col].astype(int),
        "datetime": pd.to_datetime(raw[dt_col]),
        "securityid": raw[sid_col].astype(str).str.zfill(6),
        "target_weight": pd.to_numeric(raw[w_col], errors="coerce"),
    })

    if mid_col is not None:
        df["mid_price"] = pd.to_numeric(raw[mid_col], errors="coerce")
    else:
        bid = pd.to_numeric(raw[bid_col], errors="coerce")
        ask = pd.to_numeric(raw[ask_col], errors="coerce")
        df["mid_price"] = (bid + ask) / 2.0

    df["bid_price"] = pd.to_numeric(raw[bid_col], errors="coerce") if bid_col else df["mid_price"]
    df["ask_price"] = pd.to_numeric(raw[ask_col], errors="coerce") if ask_col else df["mid_price"]

    if bench_col:
        df["benchmark_weight"] = pd.to_numeric(raw[bench_col], errors="coerce")
    else:
        df["benchmark_weight"] = np.nan

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["date", "datetime", "securityid", "target_weight", "mid_price", "bid_price", "ask_price"])
    df = df[
        (df["mid_price"] > 0)
        & (df["bid_price"] > 0)
        & (df["ask_price"] > 0)
        & (df["ask_price"] >= df["bid_price"])
    ].copy()

    if args.long_only:
        df["target_weight"] = df["target_weight"].clip(lower=0.0)

    if df["benchmark_weight"].isna().all():
        df["benchmark_weight"] = 1.0 / df.groupby(["date", "datetime"])["securityid"].transform("count")
    else:
        df["benchmark_weight"] = df["benchmark_weight"].fillna(0.0)

    df = df.sort_values(["date", "datetime", "securityid"]).reset_index(drop=True)

    print("[column mapping]")
    print("date:", date_col, "datetime:", dt_col, "symbol:", sid_col, "weight:", w_col)
    print("mid:", mid_col, "bid:", bid_col, "ask:", ask_col, "benchmark:", bench_col)

    return df


def rows_map(gt):
    return {r.securityid: r for r in gt.itertuples(index=False)}


def mark_equity(cash, positions, rows, last_price):
    equity = float(cash)
    gross = 0.0
    net = 0.0
    n = 0
    missing = 0
    for sid, sh in positions.items():
        if sh == 0:
            continue
        if sid in rows:
            px = fnum(rows[sid].bid_price)
            if not np.isfinite(px) or px <= 0:
                px = fnum(rows[sid].mid_price)
            if np.isfinite(px) and px > 0:
                last_price[sid] = px
        else:
            px = last_price.get(sid, np.nan)
            missing += 1
        if np.isfinite(px) and px > 0:
            val = sh * px
            equity += val
            gross += abs(val)
            net += val
            n += 1
    return equity, gross, net, n, missing


def benchmark_ret_from_prev(gt, prev_w, prev_px, date, prev_date):
    if prev_date is None or date != prev_date:
        prev_w.clear()
        prev_px.clear()
        for r in gt.itertuples(index=False):
            px = fnum(r.bid_price)
            if not np.isfinite(px) or px <= 0:
                px = fnum(r.mid_price)
            if np.isfinite(px) and px > 0:
                prev_px[r.securityid] = px
                prev_w[r.securityid] = fnum(r.benchmark_weight, 0.0)
        return 0.0, date

    ret = 0.0
    for r in gt.itertuples(index=False):
        sid = r.securityid
        px = fnum(r.bid_price)
        if not np.isfinite(px) or px <= 0:
            px = fnum(r.mid_price)
        old_px = prev_px.get(sid, np.nan)
        old_w = prev_w.get(sid, 0.0)
        if np.isfinite(px) and px > 0 and np.isfinite(old_px) and old_px > 0:
            ret += old_w * (px / old_px - 1.0)

    prev_w.clear()
    prev_px.clear()
    for r in gt.itertuples(index=False):
        px = fnum(r.bid_price)
        if not np.isfinite(px) or px <= 0:
            px = fnum(r.mid_price)
        if np.isfinite(px) and px > 0:
            prev_px[r.securityid] = px
            prev_w[r.securityid] = fnum(r.benchmark_weight, 0.0)

    return ret, date


def run_tplus1(df, args):
    capital = float(args.capital)
    fee_rate = float(args.fee_bps) / 10000.0
    lot = int(args.lot_size)
    min_notional = float(args.min_trade_notional)

    cash = capital
    positions = {}
    sellable = {}
    bought_today = {}
    last_price = {}

    prev_equity = capital
    bench_nav = 1.0
    prev_bench_w = {}
    prev_bench_px = {}
    prev_bench_date = None
    current_date = None

    minute_rows, trade_rows, pos_rows = [], [], []

    for (date, dt), gt in df.groupby(["date", "datetime"], sort=True):
        gt = gt.sort_values("securityid").copy()
        rows = rows_map(gt)

        new_day = current_date is None or date != current_date
        if new_day:
            current_date = date
            sellable = {sid: int(sh) for sid, sh in positions.items() if sh > 0}
            bought_today = {}

        equity_pre, _, _, _, _ = mark_equity(cash, positions, rows, last_price)
        if new_day and not args.include_overnight_pnl:
            prev_equity = equity_pre

        bret, prev_bench_date = benchmark_ret_from_prev(gt, prev_bench_w, prev_bench_px, date, prev_bench_date)
        bench_nav *= (1.0 + bret)

        base_value = equity_pre if args.target_base == "equity" else capital

        target_shares = {}
        target_weight = {}
        for r in gt.itertuples(index=False):
            sid = r.securityid
            w = fnum(r.target_weight, 0.0)
            px = fnum(r.bid_price)
            if not np.isfinite(px) or px <= 0:
                px = fnum(r.mid_price)
            target_weight[sid] = w
            target_shares[sid] = floor_lot(base_value * w / px, lot) if w > 0 and px > 0 else 0

        all_sids = set(target_shares) | set(positions)

        turnover = fee = spread_cost = 0.0
        n_buy = n_sell = n_trade = 0
        blocked_tplus1 = clipped_tplus1 = blocked_cash = clipped_cash = blocked_small = blocked_no_price = 0

        # Sells first.
        for sid in sorted(all_sids):
            cur = int(positions.get(sid, 0))
            tgt = int(target_shares.get(sid, 0))
            if tgt >= cur:
                continue

            desired = cur - tgt
            avail = int(sellable.get(sid, 0))
            qty = floor_lot(min(desired, avail), lot)

            if qty <= 0:
                blocked_tplus1 += 1
                continue
            if qty < desired:
                clipped_tplus1 += 1
            if sid not in rows:
                blocked_no_price += 1
                continue

            bid = fnum(rows[sid].bid_price)
            mid = fnum(rows[sid].mid_price)
            if not np.isfinite(bid) or bid <= 0:
                blocked_no_price += 1
                continue
            if not np.isfinite(mid) or mid <= 0:
                mid = bid

            notional = qty * bid
            if notional < min_notional and not args.allow_small_exit:
                blocked_small += 1
                continue

            trade_fee = notional * fee_rate
            cash += notional - trade_fee

            new_sh = cur - qty
            if new_sh > 0:
                positions[sid] = new_sh
            else:
                positions.pop(sid, None)

            sellable[sid] = max(0, int(sellable.get(sid, 0)) - qty)
            if sellable.get(sid, 0) == 0:
                sellable.pop(sid, None)

            turnover += notional
            fee += trade_fee
            spread_cost += max(mid - bid, 0.0) * qty
            n_sell += 1
            n_trade += 1

            trade_rows.append({
                "date": date, "datetime": dt, "securityid": sid, "side": "SELL",
                "shares": qty, "exec_price": bid, "mid_price": mid,
                "notional": notional, "fee": trade_fee,
                "spread_cost_est": max(mid - bid, 0.0) * qty,
                "target_shares": tgt, "prev_shares": cur,
                "new_shares": positions.get(sid, 0), "sellable_before": avail,
            })

        # Buys.
        for sid in sorted(all_sids):
            cur = int(positions.get(sid, 0))
            tgt = int(target_shares.get(sid, 0))
            if tgt <= cur:
                continue

            desired = tgt - cur
            qty = floor_lot(desired, lot)
            if qty <= 0:
                continue
            if sid not in rows:
                blocked_no_price += 1
                continue

            ask = fnum(rows[sid].ask_price)
            mid = fnum(rows[sid].mid_price)
            if not np.isfinite(ask) or ask <= 0:
                blocked_no_price += 1
                continue
            if not np.isfinite(mid) or mid <= 0:
                mid = ask

            notional = qty * ask
            if notional < min_notional:
                blocked_small += 1
                continue

            cash_need = notional * (1.0 + fee_rate)
            if cash_need > cash:
                max_qty = floor_lot(cash / (ask * (1.0 + fee_rate)), lot)
                max_qty = min(max_qty, qty)
                if max_qty <= 0:
                    blocked_cash += 1
                    continue
                qty = max_qty
                notional = qty * ask
                cash_need = notional * (1.0 + fee_rate)
                clipped_cash += 1

            trade_fee = notional * fee_rate
            cash -= notional + trade_fee
            positions[sid] = cur + qty
            bought_today[sid] = int(bought_today.get(sid, 0)) + qty

            turnover += notional
            fee += trade_fee
            spread_cost += max(ask - mid, 0.0) * qty
            n_buy += 1
            n_trade += 1

            trade_rows.append({
                "date": date, "datetime": dt, "securityid": sid, "side": "BUY",
                "shares": qty, "exec_price": ask, "mid_price": mid,
                "notional": notional, "fee": trade_fee,
                "spread_cost_est": max(ask - mid, 0.0) * qty,
                "target_shares": tgt, "prev_shares": cur,
                "new_shares": positions.get(sid, 0), "sellable_before": sellable.get(sid, 0),
            })

        equity_post, gross, net, n_hold, missing = mark_equity(cash, positions, rows, last_price)

        actual_ret = equity_post / prev_equity - 1.0 if prev_equity != 0 else 0.0
        prev_equity = equity_post

        actualret = equity_post / capital - 1.0
        benchmarkret = bench_nav - 1.0
        alpharet = actualret - benchmarkret

        def side_gross(shares_dict):
            total = 0.0
            for sid, sh in shares_dict.items():
                if sh == 0:
                    continue
                if sid in rows:
                    px = fnum(rows[sid].bid_price)
                    if not np.isfinite(px) or px <= 0:
                        px = fnum(rows[sid].mid_price)
                else:
                    px = last_price.get(sid, np.nan)
                if np.isfinite(px) and px > 0:
                    total += sh * px
            return total / capital

        minute_rows.append({
            "date": date,
            "datetime": dt,
            "cash": cash,
            "equity": equity_post,
            "actual_ret": actual_ret,
            "actualret": actualret,
            "benchmark_ret": bret,
            "benchmarkret": benchmarkret,
            "alpharet": alpharet,
            "turnover": turnover,
            "turnover_to_capital": turnover / capital,
            "fee": fee,
            "spread_cost_est": spread_cost,
            "total_cost": fee + spread_cost,
            "target_gross": float(gt["target_weight"].abs().sum()),
            "actual_gross": gross / capital,
            "actual_net": net / capital,
            "benchmark_gross": float(gt["benchmark_weight"].sum()),
            "n_hold": n_hold,
            "n_buy": n_buy,
            "n_sell": n_sell,
            "n_trade": n_trade,
            "blocked_tplus1": blocked_tplus1,
            "clipped_tplus1": clipped_tplus1,
            "blocked_cash": blocked_cash,
            "clipped_cash": clipped_cash,
            "blocked_small": blocked_small,
            "blocked_no_price": blocked_no_price,
            "sellable_gross": side_gross(sellable),
            "bought_today_gross": side_gross(bought_today),
            "missing_price_positions": missing,
        })

        for sid, sh in positions.items():
            if sh == 0:
                continue
            if sid in rows:
                px = fnum(rows[sid].bid_price)
                if not np.isfinite(px) or px <= 0:
                    px = fnum(rows[sid].mid_price)
            else:
                px = last_price.get(sid, np.nan)
            if np.isfinite(px) and px > 0:
                pos_rows.append({
                    "date": date, "datetime": dt, "securityid": sid,
                    "actual_shares": sh,
                    "sellable_shares": int(sellable.get(sid, 0)),
                    "bought_today_shares": int(bought_today.get(sid, 0)),
                    "mark_price": px,
                    "actual_weight": sh * px / capital,
                    "target_weight": target_weight.get(sid, 0.0),
                })

    return pd.DataFrame(minute_rows), pd.DataFrame(trade_rows), pd.DataFrame(pos_rows)


def summarize(minute, args):
    capital = float(args.capital)

    daily = minute.groupby("date", as_index=False).agg(
        num_rebalances=("datetime", "nunique"),
        daily_pnl=("actual_ret", lambda x: np.nan),
        daily_turnover=("turnover", "sum"),
        daily_fee=("fee", "sum"),
        daily_spread_cost=("spread_cost_est", "sum"),
        daily_total_cost=("total_cost", "sum"),
        avg_target_gross=("target_gross", "mean"),
        avg_actual_gross=("actual_gross", "mean"),
        avg_sellable_gross=("sellable_gross", "mean"),
        avg_bought_today_gross=("bought_today_gross", "mean"),
        avg_n_hold=("n_hold", "mean"),
        blocked_tplus1=("blocked_tplus1", "sum"),
        clipped_tplus1=("clipped_tplus1", "sum"),
        blocked_cash=("blocked_cash", "sum"),
        blocked_small=("blocked_small", "sum"),
        n_trade=("n_trade", "sum"),
    )

    equity = minute["equity"]
    drawdown = equity - equity.cummax()
    max_dd = float(drawdown.min()) if len(drawdown) else np.nan

    summary = pd.DataFrame([{
        "start_date": int(minute["date"].min()) if len(minute) else np.nan,
        "end_date": int(minute["date"].max()) if len(minute) else np.nan,
        "capital": capital,
        "num_rebalances": int(len(minute)),
        "num_days": int(minute["date"].nunique()) if len(minute) else 0,
        "actual_return": float(minute["actualret"].iloc[-1]) if len(minute) else np.nan,
        "benchmark_return": float(minute["benchmarkret"].iloc[-1]) if len(minute) else np.nan,
        "alpha_return": float(minute["alpharet"].iloc[-1]) if len(minute) else np.nan,
        "final_equity": float(minute["equity"].iloc[-1]) if len(minute) else np.nan,
        "total_pnl": float(minute["equity"].iloc[-1] - capital) if len(minute) else np.nan,
        "max_drawdown": max_dd,
        "total_turnover": float(minute["turnover"].sum()) if len(minute) else 0.0,
        "turnover_to_capital": float(minute["turnover"].sum() / capital) if len(minute) else 0.0,
        "total_fee": float(minute["fee"].sum()) if len(minute) else 0.0,
        "total_spread_cost_est": float(minute["spread_cost_est"].sum()) if len(minute) else 0.0,
        "total_cost": float(minute["total_cost"].sum()) if len(minute) else 0.0,
        "avg_target_gross": float(minute["target_gross"].mean()) if len(minute) else np.nan,
        "avg_actual_gross": float(minute["actual_gross"].mean()) if len(minute) else np.nan,
        "avg_actual_net": float(minute["actual_net"].mean()) if len(minute) else np.nan,
        "avg_sellable_gross": float(minute["sellable_gross"].mean()) if len(minute) else np.nan,
        "avg_bought_today_gross": float(minute["bought_today_gross"].mean()) if len(minute) else np.nan,
        "avg_n_hold": float(minute["n_hold"].mean()) if len(minute) else np.nan,
        "num_trade_events": int(minute["n_trade"].sum()) if len(minute) else 0,
        "num_buy_events": int(minute["n_buy"].sum()) if len(minute) else 0,
        "num_sell_events": int(minute["n_sell"].sum()) if len(minute) else 0,
        "total_blocked_tplus1": int(minute["blocked_tplus1"].sum()) if len(minute) else 0,
        "total_clipped_tplus1": int(minute["clipped_tplus1"].sum()) if len(minute) else 0,
        "total_blocked_cash": int(minute["blocked_cash"].sum()) if len(minute) else 0,
        "total_blocked_small": int(minute["blocked_small"].sum()) if len(minute) else 0,
        "fee_bps": float(args.fee_bps),
        "lot_size": int(args.lot_size),
        "min_trade_notional": float(args.min_trade_notional),
        "target_base": args.target_base,
        "include_overnight_pnl": int(args.include_overnight_pnl),
    }])

    return daily, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--positions", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--tag", required=True)

    ap.add_argument("--capital", type=float, default=200_000_000.0)
    ap.add_argument("--fee-bps", type=float, default=10.0)
    ap.add_argument("--lot-size", type=int, default=100)
    ap.add_argument("--min-trade-notional", type=float, default=5000.0)
    ap.add_argument("--allow-small-exit", type=int, default=1)
    ap.add_argument("--long-only", type=int, default=1)
    ap.add_argument("--include-overnight-pnl", type=int, default=0)
    ap.add_argument("--target-base", choices=["capital", "equity"], default="capital")

    ap.add_argument("--date-col", default=None)
    ap.add_argument("--datetime-col", default=None)
    ap.add_argument("--symbol-col", default=None)
    ap.add_argument("--weight-col", default=None)
    ap.add_argument("--benchmark-col", default=None)
    ap.add_argument("--mid-col", default=None)
    ap.add_argument("--bid-col", default=None)
    ap.add_argument("--ask-col", default=None)

    args = ap.parse_args()
    args.allow_small_exit = bool(args.allow_small_exit)
    args.long_only = bool(args.long_only)
    args.include_overnight_pnl = bool(args.include_overnight_pnl)

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    raw = read_table(args.positions)
    df = normalize_input(raw, args)

    print("[normalized shape]", df.shape)
    print("[dates]", df["date"].min(), "->", df["date"].max(), "n=", df["date"].nunique())
    print("[rebalances]", df[["date", "datetime"]].drop_duplicates().shape[0])
    print("[symbols]", df["securityid"].nunique())

    minute, trades, positions = run_tplus1(df, args)
    daily, summary = summarize(minute, args)

    minute_path = outdir / f"tplus1_minute_{args.tag}.csv"
    daily_path = outdir / f"tplus1_daily_{args.tag}.csv"
    summary_path = outdir / f"tplus1_summary_{args.tag}.csv"
    trades_path = outdir / f"tplus1_trades_{args.tag}.csv"
    positions_path = outdir / f"tplus1_positions_{args.tag}.csv"

    minute.to_csv(minute_path, index=False)
    daily.to_csv(daily_path, index=False)
    summary.to_csv(summary_path, index=False)
    trades.to_csv(trades_path, index=False)
    positions.to_csv(positions_path, index=False)

    print("\n===== T+1 summary =====")
    print(summary.T.to_string(header=False))

    print("\n[saved minute   ]", minute_path)
    print("[saved daily    ]", daily_path)
    print("[saved summary  ]", summary_path)
    print("[saved trades   ]", trades_path)
    print("[saved positions]", positions_path)


if __name__ == "__main__":
    main()
