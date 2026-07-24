# -*- coding: utf-8 -*-
"""
Pure-CS TakerModel v1

Purpose
-------
Evaluate pure cross-sectional optimizer target weights with a clean execution layer.

This model is different from v8:
  - no market timing
  - no risk-off / derisk
  - no ENTRY / HOLD / EXIT state machine
  - no alpha momentum / local buy gate / cooldown
  - no sequential optimizer state

It only does execution accounting:
  - target_weight -> target_shares
  - current_position -> delta_shares
  - opponent price execution: buy at ask, sell at bid
  - lot size rounding
  - min_trade_notional filter
  - optional volume clipping
  - optional limit up/down blocking
  - fee and spread cost accounting
  - benchmark-relative attribution

Input
-----
A target_positions file from pure_cs_cvxpy optimizer, typically containing:
  date, datetime, securityid
  target_weight / weight / opt_weight / effective_target_weight
  benchmark_weight
  mid_price, bid1/ask1 or bid_price/ask_price
  bid_volume1/ask_volume1 optional
  limit_up/limit_down optional

Output
------
  pure_cs_taker_minute_<tag>.csv
  pure_cs_taker_daily_<tag>.csv
  pure_cs_taker_summary_<tag>.csv
  pure_cs_taker_trades_<tag>.csv
"""

import argparse
from pathlib import Path
import math
import numpy as np
import pandas as pd


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    suf = "".join(path.suffixes).lower()
    if suf.endswith(".parquet"):
        return pd.read_parquet(path)
    if suf.endswith(".csv") or suf.endswith(".csv.gz"):
        return pd.read_csv(path, low_memory=False)
    raise ValueError(f"unsupported input format: {path}")


def pick_col(df: pd.DataFrame, candidates, required=True, name="column"):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"cannot find {name}; candidates={candidates}; columns={list(df.columns)}")
    return None


def safe_float(x, default=np.nan):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def floor_to_lot(shares: float, lot_size: int) -> int:
    if not np.isfinite(shares):
        return 0
    if lot_size <= 1:
        return int(math.floor(max(shares, 0.0)))
    return int(math.floor(max(shares, 0.0) / lot_size) * lot_size)


def round_delta_to_lot(delta: float, lot_size: int) -> int:
    if not np.isfinite(delta):
        return 0
    sign = 1 if delta > 0 else -1
    x = abs(delta)
    if lot_size <= 1:
        return int(sign * math.floor(x))
    return int(sign * math.floor(x / lot_size) * lot_size)


def compound_return(x):
    s = pd.Series(x).replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return np.nan
    return float((1.0 + s).prod() - 1.0)


def normalize_input(df: pd.DataFrame, args) -> pd.DataFrame:
    date_col = args.date_col or pick_col(df, ["execution_date", "date"], name="date")
    dt_col = args.datetime_col or pick_col(df, ["execution_datetime", "datetime", "timestamp"], name="datetime")
    symbol_col = args.symbol_col or pick_col(df, ["securityid", "SecurityID", "sid", "symbol"], name="symbol")

    weight_col = args.weight_col or pick_col(
        df,
        ["target_weight", "effective_target_weight", "weight", "w", "opt_weight", "optimized_weight"],
        name="target weight",
    )

    benchmark_col = args.benchmark_col or pick_col(
        df,
        ["benchmark_weight", "bench_weight", "index_weight", "ew_benchmark_weight"],
        required=False,
        name="benchmark weight",
    )

    mid_col = args.mid_col or pick_col(df, ["mid_price", "price", "tmid"], name="mid price")
    bid_col = args.bid_col or pick_col(df, ["bid_price", "bid1", "bid"], name="bid price")
    ask_col = args.ask_col or pick_col(df, ["ask_price", "ask1", "ask"], name="ask price")

    bid_vol_col = args.bid_volume_col or pick_col(
        df,
        ["bid_volume1", "bid_volume", "tbvol", "bid_vol"],
        required=False,
        name="bid volume",
    )
    ask_vol_col = args.ask_volume_col or pick_col(
        df,
        ["ask_volume1", "ask_volume", "tavol", "ask_vol"],
        required=False,
        name="ask volume",
    )

    limit_up_col = args.limit_up_col or pick_col(
        df,
        ["limit_up", "limit_up_price", "limitUpPrice"],
        required=False,
        name="limit up",
    )
    limit_down_col = args.limit_down_col or pick_col(
        df,
        ["limit_down", "limit_down_price", "limitDownPrice"],
        required=False,
        name="limit down",
    )

    out = pd.DataFrame({
        "date": df[date_col],
        "datetime": df[dt_col],
        "securityid": df[symbol_col].astype(str).str.zfill(6),
        "target_weight": pd.to_numeric(df[weight_col], errors="coerce"),
        "mid_price": pd.to_numeric(df[mid_col], errors="coerce"),
        "bid_price": pd.to_numeric(df[bid_col], errors="coerce"),
        "ask_price": pd.to_numeric(df[ask_col], errors="coerce"),
    })

    if benchmark_col is not None:
        out["benchmark_weight"] = pd.to_numeric(df[benchmark_col], errors="coerce")
    else:
        out["benchmark_weight"] = np.nan

    if bid_vol_col is not None:
        out["bid_volume"] = pd.to_numeric(df[bid_vol_col], errors="coerce")
    else:
        out["bid_volume"] = np.nan

    if ask_vol_col is not None:
        out["ask_volume"] = pd.to_numeric(df[ask_vol_col], errors="coerce")
    else:
        out["ask_volume"] = np.nan

    if limit_up_col is not None:
        out["limit_up"] = pd.to_numeric(df[limit_up_col], errors="coerce")
    else:
        out["limit_up"] = np.nan

    if limit_down_col is not None:
        out["limit_down"] = pd.to_numeric(df[limit_down_col], errors="coerce")
    else:
        out["limit_down"] = np.nan

    # Optional signal columns are only for diagnostics.
    for c in ["signal", "score", "pred_ret_h20", "pred_ret", "alpha", "label_y_raw"]:
        if c in df.columns:
            out[c] = pd.to_numeric(df[c], errors="coerce")

    out["date"] = out["date"].astype(int)
    out["datetime"] = pd.to_datetime(out["datetime"])

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=["date", "datetime", "securityid", "target_weight", "mid_price", "bid_price", "ask_price"])

    out = out[
        (out["mid_price"] > 0)
        & (out["bid_price"] > 0)
        & (out["ask_price"] > 0)
        & (out["ask_price"] >= out["bid_price"])
    ].copy()

    if args.long_only:
        out["target_weight"] = out["target_weight"].clip(lower=0.0)

    # If benchmark missing, use current cross-sectional EW benchmark.
    if out["benchmark_weight"].isna().all():
        out["benchmark_weight"] = 1.0 / out.groupby(["date", "datetime"])["securityid"].transform("count")
    else:
        out["benchmark_weight"] = out["benchmark_weight"].fillna(0.0)

    out = out.sort_values(["datetime", "securityid"]).reset_index(drop=True)

    print("[columns]")
    print("  date_col       =", date_col)
    print("  datetime_col   =", dt_col)
    print("  symbol_col     =", symbol_col)
    print("  weight_col     =", weight_col)
    print("  benchmark_col  =", benchmark_col)
    print("  mid_col        =", mid_col)
    print("  bid_col        =", bid_col)
    print("  ask_col        =", ask_col)
    print("  bid_vol_col    =", bid_vol_col)
    print("  ask_vol_col    =", ask_vol_col)
    print("  limit_up_col   =", limit_up_col)
    print("  limit_down_col =", limit_down_col)

    return out


def compute_no_cost_attribution(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x = x.sort_values(["date", "securityid", "datetime"])

    # Intraday forward return only. No overnight return is included.
    x["next_mid"] = x.groupby(["date", "securityid"])["mid_price"].shift(-1)
    x["fwd_ret"] = x["next_mid"] / x["mid_price"] - 1.0
    x = x.replace([np.inf, -np.inf], np.nan)

    y = x.dropna(subset=["fwd_ret"]).copy()

    y["target_ret_contrib"] = y["target_weight"] * y["fwd_ret"]
    y["benchmark_ret_contrib"] = y["benchmark_weight"] * y["fwd_ret"]

    per_t = (
        y.groupby(["date", "datetime"], as_index=False)
        .agg(
            target_gross=("target_weight", "sum"),
            benchmark_gross=("benchmark_weight", "sum"),
            n_names=("securityid", "nunique"),
            no_cost_strategy_return=("target_ret_contrib", "sum"),
            benchmark_return=("benchmark_ret_contrib", "sum"),
        )
    )

    per_t["same_gross_benchmark_return"] = np.where(
        per_t["benchmark_gross"].abs() > 1e-12,
        per_t["benchmark_return"] * per_t["target_gross"] / per_t["benchmark_gross"],
        np.nan,
    )
    per_t["no_cost_excess_vs_full_benchmark"] = (
        per_t["no_cost_strategy_return"] - per_t["benchmark_return"]
    )
    per_t["no_cost_excess_vs_same_gross_benchmark"] = (
        per_t["no_cost_strategy_return"] - per_t["same_gross_benchmark_return"]
    )

    return per_t


def build_row_dict(gt: pd.DataFrame):
    d = {}
    for r in gt.itertuples(index=False):
        d[r.securityid] = r
    return d


def run_execution(df: pd.DataFrame, args):
    capital = float(args.capital)
    fee_rate = float(args.fee_bps) / 10000.0
    lot_size = int(args.lot_size)

    cash = capital
    positions = {}   # sid -> shares
    last_mid = {}    # sid -> last mid price

    prev_accounting_equity = capital
    cumulative_pnl = 0.0
    current_date = None

    minute_rows = []
    trade_rows = []

    prev_target_weight = {}

    grouped = list(df.groupby(["date", "datetime"], sort=True))

    for (date, dt), gt in grouped:
        gt = gt.sort_values("securityid").copy()

        reset_accounting_after_current_mark = False

        if current_date is None:
            current_date = date
        elif date != current_date:
            # Pure intraday accounting by default:
            # when include_overnight_pnl=0, do NOT count the overnight gap from
            # previous day's last mid to current day's first mid.
            #
            # Important: the old implementation reset prev_accounting_equity using
            # previous last_mid, then immediately marked positions at current mid,
            # which still counted the overnight gap. The correct approach is:
            # first mark existing positions at current mid, then set
            # prev_accounting_equity = equity_pre and mark_pnl = 0 for this first
            # timestamp of the new date.
            if not args.include_overnight_pnl:
                reset_accounting_after_current_mark = True
            current_date = date

        rows = build_row_dict(gt)

        # Mark existing positions at current mid when available, otherwise last mid.
        equity_pre = cash
        missing_price_positions = 0
        for sid, sh in positions.items():
            if sh == 0:
                continue
            if sid in rows:
                px = safe_float(rows[sid].mid_price)
                last_mid[sid] = px
            else:
                px = last_mid.get(sid, np.nan)
                missing_price_positions += 1
            if np.isfinite(px):
                equity_pre += sh * px

        if reset_accounting_after_current_mark:
            prev_accounting_equity = equity_pre
            mark_pnl = 0.0
        else:
            mark_pnl = equity_pre - prev_accounting_equity

        # Build target shares from pure-CS target weights.
        target_shares = {}
        target_weights = {}
        for r in gt.itertuples(index=False):
            sid = r.securityid
            mid = safe_float(r.mid_price)
            w = safe_float(r.target_weight, 0.0)
            target_weights[sid] = w
            if mid > 0 and w > 0:
                raw_shares = capital * w / mid
                target_shares[sid] = floor_to_lot(raw_shares, lot_size)
            else:
                target_shares[sid] = 0

        # Pure turnover proxy in weight space, for diagnostics only.
        all_weight_names = set(prev_target_weight) | set(target_weights)
        target_turnover_weight = sum(
            abs(target_weights.get(sid, 0.0) - prev_target_weight.get(sid, 0.0))
            for sid in all_weight_names
        )
        prev_target_weight = target_weights

        # Execute delta shares.
        traded_notional = 0.0
        fee = 0.0
        spread_cost = 0.0

        n_buy = 0
        n_sell = 0
        n_trade = 0

        blocked_small = 0
        blocked_volume = 0
        blocked_limit = 0
        blocked_no_price = 0
        clipped_volume = 0

        # Only symbols appearing in current target cross-section are tradeable now.
        for sid, tgt_sh in target_shares.items():
            r = rows[sid]
            cur_sh = int(positions.get(sid, 0))
            desired_delta = int(tgt_sh - cur_sh)
            delta = round_delta_to_lot(desired_delta, lot_size)

            if delta == 0:
                continue

            mid = safe_float(r.mid_price)
            bid = safe_float(r.bid_price)
            ask = safe_float(r.ask_price)

            if not (mid > 0 and bid > 0 and ask > 0 and ask >= bid):
                blocked_no_price += 1
                continue

            if delta > 0:
                exec_px = ask

                # limit-up block
                lup = safe_float(getattr(r, "limit_up", np.nan))
                if args.use_limit_up_down and np.isfinite(lup) and lup > 0 and ask >= lup * (1.0 - 1e-8):
                    blocked_limit += 1
                    continue

                # volume clipping
                if args.use_volume_clip:
                    avol = safe_float(getattr(r, "ask_volume", np.nan))
                    if np.isfinite(avol) and avol > 0:
                        max_delta = floor_to_lot(avol * float(args.participation), lot_size)
                        if max_delta <= 0:
                            blocked_volume += 1
                            continue
                        if delta > max_delta:
                            delta = max_delta
                            clipped_volume += 1

                trade_notional = delta * exec_px

                if trade_notional < float(args.min_trade_notional):
                    blocked_small += 1
                    continue

                trade_fee = trade_notional * fee_rate
                cash -= trade_notional + trade_fee
                positions[sid] = cur_sh + delta

                traded_notional += trade_notional
                fee += trade_fee
                spread_cost += max(ask - mid, 0.0) * delta

                n_buy += 1
                n_trade += 1

                trade_rows.append({
                    "date": date,
                    "datetime": dt,
                    "securityid": sid,
                    "side": "BUY",
                    "shares": delta,
                    "exec_price": exec_px,
                    "mid_price": mid,
                    "target_shares": tgt_sh,
                    "prev_shares": cur_sh,
                    "new_shares": positions[sid],
                    "notional": trade_notional,
                    "fee": trade_fee,
                    "spread_cost_est": max(ask - mid, 0.0) * delta,
                })

            else:
                sell_sh = -delta
                exec_px = bid

                # no short
                sell_sh = min(sell_sh, cur_sh)
                sell_sh = floor_to_lot(sell_sh, lot_size)
                if sell_sh <= 0:
                    continue

                # limit-down block
                ldn = safe_float(getattr(r, "limit_down", np.nan))
                if args.use_limit_up_down and np.isfinite(ldn) and ldn > 0 and bid <= ldn * (1.0 + 1e-8):
                    blocked_limit += 1
                    continue

                if args.use_volume_clip:
                    bvol = safe_float(getattr(r, "bid_volume", np.nan))
                    if np.isfinite(bvol) and bvol > 0:
                        max_sell = floor_to_lot(bvol * float(args.participation), lot_size)
                        if max_sell <= 0:
                            blocked_volume += 1
                            continue
                        if sell_sh > max_sell:
                            sell_sh = max_sell
                            clipped_volume += 1

                trade_notional = sell_sh * exec_px

                # allow small exits optionally
                is_exit = tgt_sh == 0
                if trade_notional < float(args.min_trade_notional) and not (args.allow_small_exit and is_exit):
                    blocked_small += 1
                    continue

                trade_fee = trade_notional * fee_rate
                cash += trade_notional - trade_fee
                positions[sid] = cur_sh - sell_sh

                if positions[sid] == 0:
                    positions.pop(sid, None)

                traded_notional += trade_notional
                fee += trade_fee
                spread_cost += max(mid - bid, 0.0) * sell_sh

                n_sell += 1
                n_trade += 1

                trade_rows.append({
                    "date": date,
                    "datetime": dt,
                    "securityid": sid,
                    "side": "SELL",
                    "shares": sell_sh,
                    "exec_price": exec_px,
                    "mid_price": mid,
                    "target_shares": tgt_sh,
                    "prev_shares": cur_sh,
                    "new_shares": positions.get(sid, 0),
                    "notional": trade_notional,
                    "fee": trade_fee,
                    "spread_cost_est": max(mid - bid, 0.0) * sell_sh,
                })

        # Update last mids from current cross-section.
        for r in gt.itertuples(index=False):
            last_mid[r.securityid] = safe_float(r.mid_price)

        # Mark portfolio after trade at current mid.
        equity_post_mark = cash
        actual_gross_notional = 0.0
        actual_net_notional = 0.0
        n_hold = 0

        for sid, sh in positions.items():
            if sh == 0:
                continue
            px = rows[sid].mid_price if sid in rows else last_mid.get(sid, np.nan)
            px = safe_float(px)
            if np.isfinite(px) and px > 0:
                notional = sh * px
                equity_post_mark += notional
                actual_gross_notional += abs(notional)
                actual_net_notional += notional
                n_hold += 1

        trading_pnl = equity_post_mark - equity_pre
        total_pnl = equity_post_mark - prev_accounting_equity

        cumulative_pnl += total_pnl
        accounting_equity = capital + cumulative_pnl
        prev_accounting_equity = equity_post_mark

        target_gross = float(gt["target_weight"].sum())
        benchmark_gross = float(gt["benchmark_weight"].sum())

        minute_rows.append({
            "date": date,
            "datetime": dt,
            "n_universe": int(gt["securityid"].nunique()),
            "target_gross": target_gross,
            "benchmark_gross": benchmark_gross,
            "target_turnover_weight": target_turnover_weight,

            "actual_gross_weight": actual_gross_notional / capital,
            "actual_net_weight": actual_net_notional / capital,
            "n_hold": n_hold,

            "cash": cash,
            "equity_mark": equity_post_mark,
            "accounting_equity": accounting_equity,

            "mark_pnl": mark_pnl,
            "trading_pnl": trading_pnl,
            "total_pnl": total_pnl,
            "cum_pnl": cumulative_pnl,
            "return_on_capital_t": total_pnl / capital,

            "turnover": traded_notional,
            "turnover_to_capital": traded_notional / capital,
            "fee": fee,
            "spread_cost_est": spread_cost,
            "total_cost": fee + spread_cost,

            "n_trade": n_trade,
            "n_buy": n_buy,
            "n_sell": n_sell,

            "blocked_small": blocked_small,
            "blocked_volume": blocked_volume,
            "blocked_limit": blocked_limit,
            "blocked_no_price": blocked_no_price,
            "clipped_volume": clipped_volume,
            "missing_price_positions": missing_price_positions,
        })

    minute = pd.DataFrame(minute_rows)
    trades = pd.DataFrame(trade_rows)

    return minute, trades


def summarize(minute: pd.DataFrame, attrib: pd.DataFrame, trades: pd.DataFrame, args) -> tuple:
    capital = float(args.capital)

    x = minute.merge(
        attrib,
        on=["date", "datetime"],
        how="left",
        suffixes=("", "_attrib"),
    )

    # Defensive compatibility: if an older pandas/default merge already produced _x/_y names,
    # restore the execution/minute-side columns expected below.
    rename_back = {}
    for c in ["target_gross", "benchmark_gross", "n_names"]:
        if c not in x.columns and f"{c}_x" in x.columns:
            rename_back[f"{c}_x"] = c
    if rename_back:
        x = x.rename(columns=rename_back)

    # Strategy actual return from execution accounting.
    daily = (
        x.groupby("date", as_index=False)
        .agg(
            num_rebalances=("datetime", "nunique"),
            daily_net_pnl=("total_pnl", "sum"),
            daily_turnover=("turnover", "sum"),
            daily_fee=("fee", "sum"),
            daily_spread_cost=("spread_cost_est", "sum"),
            daily_total_cost=("total_cost", "sum"),
            daily_n_trade=("n_trade", "sum"),
            avg_target_gross=("target_gross", "mean"),
            avg_actual_gross=("actual_gross_weight", "mean"),
            avg_n_hold=("n_hold", "mean"),
            no_cost_strategy_return=("no_cost_strategy_return", compound_return),
            benchmark_return=("benchmark_return", compound_return),
            same_gross_benchmark_return=("same_gross_benchmark_return", compound_return),
        )
    )

    daily["actual_strategy_return"] = daily["daily_net_pnl"] / capital
    daily["no_cost_excess_vs_full_benchmark"] = (
        daily["no_cost_strategy_return"] - daily["benchmark_return"]
    )
    daily["no_cost_excess_vs_same_gross_benchmark"] = (
        daily["no_cost_strategy_return"] - daily["same_gross_benchmark_return"]
    )
    daily["actual_excess_vs_full_benchmark"] = (
        daily["actual_strategy_return"] - daily["benchmark_return"]
    )
    daily["actual_excess_vs_same_gross_benchmark"] = (
        daily["actual_strategy_return"] - daily["same_gross_benchmark_return"]
    )

    total_net_pnl = float(minute["total_pnl"].sum()) if not minute.empty else 0.0
    total_turnover = float(minute["turnover"].sum()) if not minute.empty else 0.0
    total_fee = float(minute["fee"].sum()) if not minute.empty else 0.0
    total_spread = float(minute["spread_cost_est"].sum()) if not minute.empty else 0.0

    actual_return = total_net_pnl / capital

    no_cost_strategy_return = compound_return(attrib["no_cost_strategy_return"]) if not attrib.empty else np.nan
    benchmark_return = compound_return(attrib["benchmark_return"]) if not attrib.empty else np.nan
    same_gross_benchmark_return = compound_return(attrib["same_gross_benchmark_return"]) if not attrib.empty else np.nan

    equity = minute["accounting_equity"] if "accounting_equity" in minute.columns else pd.Series(dtype=float)
    if not equity.empty:
        running_max = equity.cummax()
        drawdown = equity - running_max
        max_drawdown = float(drawdown.min())
    else:
        max_drawdown = np.nan

    summary = pd.DataFrame([{
        "start_date": int(minute["date"].min()) if not minute.empty else np.nan,
        "end_date": int(minute["date"].max()) if not minute.empty else np.nan,
        "capital": capital,
        "num_rebalances": int(len(minute)),
        "num_days": int(minute["date"].nunique()) if not minute.empty else 0,

        "actual_return": actual_return,
        "total_net_pnl": total_net_pnl,
        "final_accounting_equity": capital + total_net_pnl,
        "max_drawdown": max_drawdown,

        "no_cost_strategy_return": no_cost_strategy_return,
        "benchmark_return": benchmark_return,
        "same_gross_benchmark_return": same_gross_benchmark_return,

        "actual_excess_vs_full_benchmark": actual_return - benchmark_return if np.isfinite(benchmark_return) else np.nan,
        "actual_excess_vs_same_gross_benchmark": actual_return - same_gross_benchmark_return if np.isfinite(same_gross_benchmark_return) else np.nan,
        "no_cost_excess_vs_full_benchmark": no_cost_strategy_return - benchmark_return if np.isfinite(no_cost_strategy_return) and np.isfinite(benchmark_return) else np.nan,
        "no_cost_excess_vs_same_gross_benchmark": no_cost_strategy_return - same_gross_benchmark_return if np.isfinite(no_cost_strategy_return) and np.isfinite(same_gross_benchmark_return) else np.nan,

        "total_turnover": total_turnover,
        "turnover_to_capital": total_turnover / capital,
        "total_fee": total_fee,
        "total_spread_cost_est": total_spread,
        "total_cost": total_fee + total_spread,

        "avg_target_gross": float(minute["target_gross"].mean()) if not minute.empty else np.nan,
        "avg_actual_gross": float(minute["actual_gross_weight"].mean()) if not minute.empty else np.nan,
        "avg_actual_net": float(minute["actual_net_weight"].mean()) if not minute.empty else np.nan,
        "avg_n_hold": float(minute["n_hold"].mean()) if not minute.empty else np.nan,

        "num_trade_events": int(minute["n_trade"].sum()) if not minute.empty else 0,
        "num_buy_events": int(minute["n_buy"].sum()) if not minute.empty else 0,
        "num_sell_events": int(minute["n_sell"].sum()) if not minute.empty else 0,

        "total_blocked_small": int(minute["blocked_small"].sum()) if not minute.empty else 0,
        "total_blocked_volume": int(minute["blocked_volume"].sum()) if not minute.empty else 0,
        "total_blocked_limit": int(minute["blocked_limit"].sum()) if not minute.empty else 0,
        "total_clipped_volume": int(minute["clipped_volume"].sum()) if not minute.empty else 0,

        "fee_bps": float(args.fee_bps),
        "min_trade_notional": float(args.min_trade_notional),
        "lot_size": int(args.lot_size),
        "use_volume_clip": int(args.use_volume_clip),
        "participation": float(args.participation),
        "use_limit_up_down": int(args.use_limit_up_down),
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
    ap.add_argument("--min-trade-notional", type=float, default=20000.0)

    ap.add_argument("--use-volume-clip", type=int, default=1)
    ap.add_argument("--participation", type=float, default=0.10)
    ap.add_argument("--use-limit-up-down", type=int, default=1)
    ap.add_argument("--allow-small-exit", type=int, default=1)
    ap.add_argument("--long-only", type=int, default=1)

    ap.add_argument("--include-overnight-pnl", type=int, default=0)

    # Optional manual column mapping.
    ap.add_argument("--date-col", default=None)
    ap.add_argument("--datetime-col", default=None)
    ap.add_argument("--symbol-col", default=None)
    ap.add_argument("--weight-col", default=None)
    ap.add_argument("--benchmark-col", default=None)
    ap.add_argument("--mid-col", default=None)
    ap.add_argument("--bid-col", default=None)
    ap.add_argument("--ask-col", default=None)
    ap.add_argument("--bid-volume-col", default=None)
    ap.add_argument("--ask-volume-col", default=None)
    ap.add_argument("--limit-up-col", default=None)
    ap.add_argument("--limit-down-col", default=None)

    args = ap.parse_args()

    args.use_volume_clip = bool(args.use_volume_clip)
    args.use_limit_up_down = bool(args.use_limit_up_down)
    args.allow_small_exit = bool(args.allow_small_exit)
    args.long_only = bool(args.long_only)
    args.include_overnight_pnl = bool(args.include_overnight_pnl)

    positions_path = Path(args.positions)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("[read]", positions_path)
    raw = read_table(positions_path)
    print("[raw shape]", raw.shape)

    df = normalize_input(raw, args)
    print("[normalized shape]", df.shape)
    print("[date]", df["date"].min(), "->", df["date"].max(), "n=", df["date"].nunique())
    print("[rebalances]", df[["date", "datetime"]].drop_duplicates().shape[0])
    print("[symbols]", df["securityid"].nunique())
    print("[target gross describe]")
    print(df.groupby(["date", "datetime"])["target_weight"].sum().describe())

    if args.use_volume_clip and (df["bid_volume"].isna().all() or df["ask_volume"].isna().all()):
        print("[WARN] use_volume_clip=1 but bid/ask volume columns are missing or all NaN; volume clipping will be effectively skipped.")

    attrib = compute_no_cost_attribution(df)

    print("[run execution]")
    minute, trades = run_execution(df, args)

    daily, summary = summarize(minute, attrib, trades, args)

    minute_path = outdir / f"pure_cs_taker_minute_{args.tag}.csv"
    daily_path = outdir / f"pure_cs_taker_daily_{args.tag}.csv"
    summary_path = outdir / f"pure_cs_taker_summary_{args.tag}.csv"
    trades_path = outdir / f"pure_cs_taker_trades_{args.tag}.csv"
    attrib_path = outdir / f"pure_cs_taker_attribution_grid_{args.tag}.csv"

    minute.to_csv(minute_path, index=False)
    daily.to_csv(daily_path, index=False)
    summary.to_csv(summary_path, index=False)
    trades.to_csv(trades_path, index=False)
    attrib.to_csv(attrib_path, index=False)

    print("\n===== Pure-CS TakerModel Summary =====")
    print(summary.T.to_string(header=False))

    print("\n[saved minute ]", minute_path)
    print("[saved daily  ]", daily_path)
    print("[saved summary]", summary_path)
    print("[saved trades ]", trades_path)
    print("[saved attrib ]", attrib_path)


if __name__ == "__main__":
    main()
