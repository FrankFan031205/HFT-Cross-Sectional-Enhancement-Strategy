# -*- coding: utf-8 -*-
"""
T+1-aware Pure-CS CVXPY Optimizer.

This script is intended to replace a naive "ideal target then T+1 clip" pipeline.

Main idea
---------
At each rebalance timestamp, solve a cross-sectional portfolio optimization problem
with T+1 feasibility included in the optimizer itself.

State variables maintained by this script:
  - actual_shares: current real holdings
  - sellable_shares: shares that can be sold today
  - bought_today_shares: shares bought today and locked by T+1 until next trading day
  - cash

Optimization variables:
  - w_i: desired target weight for each stock at current rebalance

T+1-aware constraints:
  - w_i >= locked_weight_i
    where locked_weight_i = non-sellable shares * current price / capital
  - sell_i <= sellable_weight_i
    where sell_i >= actual_weight_i - w_i

Other portfolio constraints:
  - long only
  - gross min/max
  - single-name cap
  - active L1 relative to benchmark
  - per-rebalance turnover budget
  - daily turnover budget

Execution:
  - sell first, then buy
  - buy at ask/task
  - sell at bid/tbid
  - fee charged on turnover, e.g. 10bp
  - lot size and min trade notional applied
  - T+1 state updated after execution

Outputs:
  - target_positions.csv
  - summary_by_rebalance.csv
  - trades.csv
  - executed_positions.csv

Notes:
  - This is an optimizer + stateful execution simulator. The optimizer output target is
    already T+1 feasible up to lot/cash/min-notional rounding.
  - NAV reporting can be done separately, but summary includes current actual gross,
    target gross, turnover, and T+1 diagnostics.
"""

import argparse
import glob
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    import polars as pl
except Exception:
    pl = None

try:
    import cvxpy as cp
except Exception as e:
    raise RuntimeError("cvxpy is required for this script") from e


def read_any(path: Path) -> pd.DataFrame:
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
        parts.append(read_any(Path(f)))
    return pd.concat(parts, ignore_index=True)


def pick_col(df: pd.DataFrame, candidates: List[str], required=True, name="column"):
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


def normalize_market_input(df: pd.DataFrame, signal_col: str) -> pd.DataFrame:
    date_col = pick_col(df, ["date", "execution_date"], name="date")
    dt_col = pick_col(df, ["datetime", "execution_datetime", "timestamp"], name="datetime")
    sid_col = pick_col(df, ["securityid", "SecurityID", "sid", "symbol"], name="symbol")

    sig_col = pick_col(df, [signal_col, "pred_ret_h20", "pred_ret_h30", "pred_z", "signal", "score"], name="signal")
    bench_col = pick_col(df, ["benchmark_weight", "bench_weight", "index_weight", "ew_benchmark_weight"], required=False, name="benchmark")

    mid_col = pick_col(df, ["mid_price", "price", "tmid"], required=False, name="mid price")
    bid_col = pick_col(df, ["bid_price", "bid1", "bid", "tbid", "exec_sell_price"], required=False, name="bid price")
    ask_col = pick_col(df, ["ask_price", "ask1", "ask", "task", "exec_buy_price"], required=False, name="ask price")

    if mid_col is None and (bid_col is None or ask_col is None):
        raise KeyError("need either mid price column or bid+ask columns")

    out = pd.DataFrame({
        "date": df[date_col].astype(int),
        "datetime": pd.to_datetime(df[dt_col]),
        "securityid": df[sid_col].astype(str).str.zfill(6),
        "signal": pd.to_numeric(df[sig_col], errors="coerce"),
    })

    if mid_col is not None:
        out["mid_price"] = pd.to_numeric(df[mid_col], errors="coerce")
    else:
        bid_tmp = pd.to_numeric(df[bid_col], errors="coerce")
        ask_tmp = pd.to_numeric(df[ask_col], errors="coerce")
        out["mid_price"] = (bid_tmp + ask_tmp) / 2.0

    if bid_col is not None:
        out["bid_price"] = pd.to_numeric(df[bid_col], errors="coerce")
    else:
        out["bid_price"] = out["mid_price"]

    if ask_col is not None:
        out["ask_price"] = pd.to_numeric(df[ask_col], errors="coerce")
    else:
        out["ask_price"] = out["mid_price"]

    if bench_col is not None:
        out["benchmark_weight"] = pd.to_numeric(df[bench_col], errors="coerce")
    else:
        out["benchmark_weight"] = np.nan

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=["date", "datetime", "securityid", "mid_price", "bid_price", "ask_price"])
    out = out[
        (out["mid_price"] > 0)
        & (out["bid_price"] > 0)
        & (out["ask_price"] > 0)
        & (out["ask_price"] >= out["bid_price"])
    ].copy()

    out["signal"] = out["signal"].fillna(0.0)

    if out["benchmark_weight"].isna().all():
        out["benchmark_weight"] = 1.0 / out.groupby(["date", "datetime"])["securityid"].transform("count")
    else:
        out["benchmark_weight"] = out["benchmark_weight"].fillna(0.0)

    out = out.sort_values(["date", "datetime", "securityid"]).reset_index(drop=True)

    return out


def load_time_grid(time_grid_file: str) -> pd.DataFrame:
    tg = read_any(Path(time_grid_file))
    date_col = pick_col(tg, ["date", "execution_date"], name="time grid date")
    dt_col = pick_col(tg, ["datetime", "execution_datetime"], name="time grid datetime")
    out = tg[[date_col, dt_col]].copy()
    out.columns = ["date", "datetime"]
    out["date"] = out["date"].astype(int)
    out["datetime"] = pd.to_datetime(out["datetime"])
    out = out.drop_duplicates(["date", "datetime"]).sort_values(["date", "datetime"]).reset_index(drop=True)
    return out


def select_rebalance_times(df: pd.DataFrame, args) -> pd.DataFrame:
    if args.time_grid_file:
        tg = load_time_grid(args.time_grid_file)
        available = df[["date", "datetime"]].drop_duplicates()
        merged = tg.merge(available, on=["date", "datetime"], how="inner")
        if merged.empty:
            raise RuntimeError("time_grid_file has no overlap with input market datetimes")
        return merged.sort_values(["date", "datetime"]).reset_index(drop=True)

    times = df[["date", "datetime"]].drop_duplicates().sort_values(["date", "datetime"]).reset_index(drop=True)
    start_time = pd.to_datetime(args.start_time).time() if args.start_time else None
    end_time = pd.to_datetime(args.end_time).time() if args.end_time else None

    out = []
    for d, g in times.groupby("date", sort=True):
        gg = g.copy()
        if start_time is not None:
            gg = gg[gg["datetime"].dt.time >= start_time]
        if end_time is not None:
            gg = gg[gg["datetime"].dt.time <= end_time]
        gg = gg.reset_index(drop=True)
        if gg.empty:
            continue
        out.append(gg.iloc[:: int(args.rebalance_minutes)])
    if not out:
        raise RuntimeError("no rebalance times selected")
    return pd.concat(out, ignore_index=True).sort_values(["date", "datetime"]).reset_index(drop=True)


def zscore_signal(x: np.ndarray, clip: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    mu = x.mean()
    sd = x.std()
    if sd > 1e-12:
        x = (x - mu) / sd
    else:
        x = x * 0.0
    if clip > 0:
        x = np.clip(x, -clip, clip)
    return x


def floor_lot(shares: float, lot_size: int) -> int:
    if not np.isfinite(shares):
        return 0
    shares = max(float(shares), 0.0)
    if lot_size <= 1:
        return int(math.floor(shares))
    return int(math.floor(shares / lot_size) * lot_size)


def solve_one_rebalance(cur: pd.DataFrame, state: dict, args, daily_turnover_used: float):
    n = len(cur)
    sids = cur["securityid"].tolist()

    signal = zscore_signal(cur["signal"].values, args.signal_clip) * float(args.alpha_scale)

    bench = cur["benchmark_weight"].values.astype(float)
    if bench.sum() > 1e-12:
        bench = bench / bench.sum()
    else:
        bench = np.ones(n) / n

    px = cur["bid_price"].values.astype(float)

    actual_sh = np.array([state["positions"].get(sid, 0) for sid in sids], dtype=float)
    sellable_sh = np.array([state["sellable"].get(sid, 0) for sid in sids], dtype=float)
    locked_sh = np.maximum(actual_sh - sellable_sh, 0.0)

    actual_w = actual_sh * px / args.capital
    sellable_w = sellable_sh * px / args.capital
    locked_w = locked_sh * px / args.capital

    locked_sum = float(locked_w.sum())

    dynamic_gross_max = max(float(args.gross_max), locked_sum + 1e-6)
    dynamic_gross_min = min(float(args.gross_min), dynamic_gross_max)

    remaining_daily_turnover = max(0.0, float(args.daily_turnover_budget) - daily_turnover_used)
    turnover_limit = min(float(args.turnover_limit), remaining_daily_turnover)
    if state["is_initial_build"]:
        turnover_limit = max(turnover_limit, float(args.initial_build_limit))

    fee_rate = float(args.fee_bps) / 10000.0
    cash_weight = float(state.get("cash", 0.0)) / float(args.capital)
    cash_buffer = float(getattr(args, "cash_buffer", 0.002))
    cash_available_weight = max(0.0, cash_weight - cash_buffer)

    max_affordable_gross = float(actual_w.sum()) + cash_available_weight / max(1e-12, (1.0 + fee_rate))
    dynamic_gross_min = min(dynamic_gross_min, dynamic_gross_max, max_affordable_gross + 1e-8)

    active_l1_param = cp.Parameter(nonneg=True, value=float(args.active_l1_limit))

    w = cp.Variable(n)
    buy = cp.Variable(n)
    sell = cp.Variable(n)
    active_abs = cp.Variable(n)

    constraints = [
        w >= locked_w,
        w >= 0,
        w <= float(args.single_name_cap),
        cp.sum(w) >= dynamic_gross_min,
        cp.sum(w) <= dynamic_gross_max,

        buy >= w - actual_w,
        buy >= 0,
        sell >= actual_w - w,
        sell >= 0,
        sell <= sellable_w + 1e-12,

        active_abs >= w - bench,
        active_abs >= -(w - bench),
        active_abs >= 0,
        cp.sum(active_abs) <= active_l1_param,

        cp.sum(buy + sell) <= turnover_limit + 1e-12,

        (1.0 + fee_rate) * cp.sum(buy) <= cash_available_weight + (1.0 - fee_rate) * cp.sum(sell) + 1e-12,
    ]

    obj = (
        signal @ w
        - float(args.lambda_turnover) * cp.sum(buy + sell)
        - float(args.lambda_active) * cp.sum(active_abs)
        - float(args.lambda_ridge) * cp.sum_squares(w - bench)
    )

    prob = cp.Problem(cp.Maximize(obj), constraints)

    status = "not_solved"
    solver_used = ""
    objective = np.nan
    w_val = None

    solvers = [x.strip().upper() for x in args.solvers.split(",") if x.strip()]

    active_candidates = [float(args.active_l1_limit)]
    for x in str(getattr(args, "active_l1_relax_list", "")).split(","):
        x = x.strip()
        if not x:
            continue
        val = float(x)
        if val > active_candidates[-1] + 1e-12:
            active_candidates.append(val)

    active_l1_used = active_candidates[0]

    for active_lim in active_candidates:
        active_l1_param.value = float(active_lim)
        active_l1_used = float(active_lim)

        for solver in solvers:
            try:
                if solver == "CLARABEL":
                    prob.solve(solver=cp.CLARABEL, verbose=False)
                elif solver == "OSQP":
                    prob.solve(solver=cp.OSQP, verbose=False, eps_abs=1e-6, eps_rel=1e-6, max_iter=20000)
                elif solver == "SCS":
                    prob.solve(solver=cp.SCS, verbose=False, eps=1e-5, max_iters=10000)
                else:
                    continue

                status = str(prob.status)
                solver_used = solver

                if status in ("optimal", "optimal_inaccurate") and w.value is not None:
                    w_val = np.asarray(w.value, dtype=float)
                    objective = float(prob.value)
                    if active_lim > float(args.active_l1_limit) + 1e-12:
                        status = f"{status}_relaxed_active_l1_{active_lim:.2f}"
                    break
            except Exception as e:
                status = f"{solver}_error:{type(e).__name__}"

        if w_val is not None:
            break

    if w_val is None:
        # Fallback: keep actual portfolio, but respect locked minimum.
        w_val = np.maximum(actual_w, locked_w)
        gross = w_val.sum()
        if gross < dynamic_gross_min:
            add = dynamic_gross_min - gross
            free_cap = np.maximum(float(args.single_name_cap) - w_val, 0.0)
            if free_cap.sum() > 1e-12:
                w_val += min(add, free_cap.sum()) * free_cap / free_cap.sum()
        if w_val.sum() > dynamic_gross_max and w_val.sum() > 1e-12:
            # Reduce only sellable part proportionally, locked cannot be reduced.
            reducible = np.maximum(w_val - locked_w, 0.0)
            excess = w_val.sum() - dynamic_gross_max
            if reducible.sum() > 1e-12:
                w_val -= min(excess, reducible.sum()) * reducible / reducible.sum()
        status = "fallback_keep_actual"

    w_val = np.nan_to_num(w_val, nan=0.0, posinf=0.0, neginf=0.0)
    w_val = np.maximum(w_val, locked_w)
    w_val = np.clip(w_val, 0.0, float(args.single_name_cap))

    return {
        "sids": sids,
        "weights": w_val,
        "benchmark": bench,
        "signal_scaled": signal,
        "actual_w_before": actual_w,
        "sellable_w_before": sellable_w,
        "locked_w_before": locked_w,
        "turnover_limit": turnover_limit,
        "dynamic_gross_min": dynamic_gross_min,
        "dynamic_gross_max": dynamic_gross_max,
        "locked_sum": locked_sum,
        "cash_weight": cash_weight,
        "cash_available_weight": cash_available_weight,
        "max_affordable_gross": max_affordable_gross,
        "active_l1_limit_used": active_l1_used if "active_l1_used" in locals() else float(args.active_l1_limit),
        "status": status,
        "solver": solver_used,
        "objective": objective,
    }


def execute_to_target(cur: pd.DataFrame, solution: dict, state: dict, args):
    sid_to_row = {r.securityid: r for r in cur.itertuples(index=False)}
    target_w = dict(zip(solution["sids"], solution["weights"]))
    bench_w = dict(zip(solution["sids"], solution["benchmark"]))
    signal_scaled = dict(zip(solution["sids"], solution["signal_scaled"]))
    locked_before = dict(zip(solution["sids"], solution["locked_w_before"]))
    sellable_before = dict(zip(solution["sids"], solution["sellable_w_before"]))
    actual_before = dict(zip(solution["sids"], solution["actual_w_before"]))

    trades = []
    out_rows = []

    turnover_notional = 0.0
    turnover_weight = 0.0
    fee = 0.0
    spread_cost = 0.0

    blocked_tplus1 = 0
    clipped_tplus1 = 0
    blocked_cash = 0
    clipped_cash = 0
    blocked_small = 0
    blocked_no_price = 0
    n_buy = 0
    n_sell = 0

    # Build target shares.
    target_shares = {}
    for sid, w in target_w.items():
        r = sid_to_row[sid]
        px = float(r.bid_price)
        if px <= 0:
            px = float(r.mid_price)
        target_shares[sid] = floor_lot(args.capital * float(w) / px, args.lot_size)

    all_sids = sorted(set(state["positions"]) | set(target_shares))

    # Sells first.
    for sid in all_sids:
        cur_sh = int(state["positions"].get(sid, 0))
        tgt_sh = int(target_shares.get(sid, 0))
        if tgt_sh >= cur_sh:
            continue

        if sid not in sid_to_row:
            blocked_no_price += 1
            continue

        desired_sell = cur_sh - tgt_sh
        available = int(state["sellable"].get(sid, 0))
        sell_qty = min(desired_sell, available)
        sell_qty = floor_lot(sell_qty, args.lot_size)

        if sell_qty <= 0:
            blocked_tplus1 += 1
            continue

        if sell_qty < desired_sell:
            clipped_tplus1 += 1

        r = sid_to_row[sid]
        bid = float(r.bid_price)
        mid = float(r.mid_price)

        notional = sell_qty * bid
        if notional < args.min_trade_notional and not args.allow_small_exit:
            blocked_small += 1
            continue

        trade_fee = notional * args.fee_bps / 10000.0

        state["cash"] += notional - trade_fee
        new_sh = cur_sh - sell_qty
        if new_sh > 0:
            state["positions"][sid] = new_sh
        else:
            state["positions"].pop(sid, None)

        state["sellable"][sid] = max(0, int(state["sellable"].get(sid, 0)) - sell_qty)
        if state["sellable"].get(sid, 0) == 0:
            state["sellable"].pop(sid, None)

        turnover_notional += notional
        turnover_weight += notional / args.capital
        fee += trade_fee
        spread_cost += max(mid - bid, 0.0) * sell_qty
        n_sell += 1

        trades.append({
            "side": "SELL",
            "securityid": sid,
            "shares": sell_qty,
            "exec_price": bid,
            "mid_price": mid,
            "notional": notional,
            "fee": trade_fee,
            "spread_cost_est": max(mid - bid, 0.0) * sell_qty,
            "prev_shares": cur_sh,
            "target_shares": tgt_sh,
            "new_shares": state["positions"].get(sid, 0),
            "sellable_before_shares": available,
        })

    # Buys.
    for sid in all_sids:
        cur_sh = int(state["positions"].get(sid, 0))
        tgt_sh = int(target_shares.get(sid, 0))
        if tgt_sh <= cur_sh:
            continue

        if sid not in sid_to_row:
            blocked_no_price += 1
            continue

        buy_qty = floor_lot(tgt_sh - cur_sh, args.lot_size)
        if buy_qty <= 0:
            continue

        r = sid_to_row[sid]
        ask = float(r.ask_price)
        mid = float(r.mid_price)

        notional = buy_qty * ask
        if notional < args.min_trade_notional:
            blocked_small += 1
            continue

        cash_need = notional * (1.0 + args.fee_bps / 10000.0)
        if cash_need > state["cash"]:
            max_qty = floor_lot(state["cash"] / (ask * (1.0 + args.fee_bps / 10000.0)), args.lot_size)
            max_qty = min(max_qty, buy_qty)
            if max_qty <= 0:
                blocked_cash += 1
                continue
            buy_qty = max_qty
            notional = buy_qty * ask
            cash_need = notional * (1.0 + args.fee_bps / 10000.0)
            clipped_cash += 1

        trade_fee = notional * args.fee_bps / 10000.0

        state["cash"] -= notional + trade_fee
        state["positions"][sid] = cur_sh + buy_qty
        state["bought_today"][sid] = int(state["bought_today"].get(sid, 0)) + buy_qty

        turnover_notional += notional
        turnover_weight += notional / args.capital
        fee += trade_fee
        spread_cost += max(ask - mid, 0.0) * buy_qty
        n_buy += 1

        trades.append({
            "side": "BUY",
            "securityid": sid,
            "shares": buy_qty,
            "exec_price": ask,
            "mid_price": mid,
            "notional": notional,
            "fee": trade_fee,
            "spread_cost_est": max(ask - mid, 0.0) * buy_qty,
            "prev_shares": cur_sh,
            "target_shares": tgt_sh,
            "new_shares": state["positions"].get(sid, 0),
            "sellable_before_shares": state["sellable"].get(sid, 0),
        })

    # Output rows for all current universe symbols.
    for sid in solution["sids"]:
        r = sid_to_row[sid]
        sh = int(state["positions"].get(sid, 0))
        px = float(r.bid_price)
        actual_w_after = sh * px / args.capital

        out_rows.append({
            "date": int(r.date),
            "datetime": pd.to_datetime(r.datetime),
            "securityid": sid,
            "target_weight": float(target_w.get(sid, 0.0)),
            "benchmark_weight": float(bench_w.get(sid, 0.0)),
            "signal_scaled": float(signal_scaled.get(sid, 0.0)),
            "actual_weight_before": float(actual_before.get(sid, 0.0)),
            "sellable_weight_before": float(sellable_before.get(sid, 0.0)),
            "locked_weight_before": float(locked_before.get(sid, 0.0)),
            "actual_weight_after": actual_w_after,
            "target_shares": int(target_shares.get(sid, 0)),
            "actual_shares_after": sh,
            "sellable_shares_after": int(state["sellable"].get(sid, 0)),
            "bought_today_shares_after": int(state["bought_today"].get(sid, 0)),
            "mid_price": float(r.mid_price),
            "bid_price": float(r.bid_price),
            "ask_price": float(r.ask_price),
        })

    diag = {
        "turnover_notional": turnover_notional,
        "turnover_weight": turnover_weight,
        "fee": fee,
        "spread_cost_est": spread_cost,
        "total_cost": fee + spread_cost,
        "n_buy": n_buy,
        "n_sell": n_sell,
        "n_trade": n_buy + n_sell,
        "blocked_tplus1": blocked_tplus1,
        "clipped_tplus1": clipped_tplus1,
        "blocked_cash": blocked_cash,
        "clipped_cash": clipped_cash,
        "blocked_small": blocked_small,
        "blocked_no_price": blocked_no_price,
    }

    return out_rows, trades, diag


def mark_state(cur: pd.DataFrame, state: dict, args):
    sid_to_row = {r.securityid: r for r in cur.itertuples(index=False)}
    equity = float(state["cash"])
    gross = 0.0
    net = 0.0
    n_hold = 0

    sellable_gross = 0.0
    locked_gross = 0.0

    for sid, sh in state["positions"].items():
        if sh <= 0:
            continue
        if sid not in sid_to_row:
            continue
        px = float(sid_to_row[sid].bid_price)
        notional = sh * px
        equity += notional
        gross += abs(notional)
        net += notional
        n_hold += 1

        sellable_sh = state["sellable"].get(sid, 0)
        locked_sh = max(sh - sellable_sh, 0)
        sellable_gross += sellable_sh * px
        locked_gross += locked_sh * px

    return {
        "equity": equity,
        "actual_gross": gross / args.capital,
        "actual_net": net / args.capital,
        "n_hold": n_hold,
        "sellable_gross": sellable_gross / args.capital,
        "locked_gross": locked_gross / args.capital,
        "cash_weight": state["cash"] / args.capital,
    }


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--input-glob", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--signal-col", default="pred_ret_h20")
    ap.add_argument("--time-grid-file", default="")

    ap.add_argument("--capital", type=float, default=200_000_000)
    ap.add_argument("--fee-bps", type=float, default=10.0)
    ap.add_argument("--lot-size", type=int, default=100)
    ap.add_argument("--min-trade-notional", type=float, default=5000.0)
    ap.add_argument("--allow-small-exit", type=int, default=1)

    ap.add_argument("--rebalance-minutes", type=int, default=20)
    ap.add_argument("--start-time", default="")
    ap.add_argument("--end-time", default="")

    ap.add_argument("--gross-target", type=float, default=0.95)
    ap.add_argument("--gross-min", type=float, default=0.92)
    ap.add_argument("--gross-max", type=float, default=0.98)
    ap.add_argument("--single-name-cap", type=float, default=0.008)
    ap.add_argument("--active-l1-limit", type=float, default=0.25)

    ap.add_argument("--turnover-limit", type=float, default=0.05)
    ap.add_argument("--daily-turnover-budget", type=float, default=1.20)
    ap.add_argument("--initial-build-limit", type=float, default=1.00)

    ap.add_argument("--alpha-scale", type=float, default=0.0005)
    ap.add_argument("--signal-clip", type=float, default=5.0)
    ap.add_argument("--lambda-turnover", type=float, default=0.0010)
    ap.add_argument("--lambda-active", type=float, default=0.0001)
    ap.add_argument("--lambda-ridge", type=float, default=0.00001)
    ap.add_argument("--solvers", default="CLARABEL,SCS,OSQP")
    ap.add_argument("--cash-buffer", type=float, default=0.002)
    ap.add_argument("--active-l1-relax-list", default="0.35,0.50,0.75,1.00")

    ap.add_argument("--warm-start-benchmark", type=int, default=0)
    ap.add_argument("--warm-start-gross", type=float, default=-1.0)
    ap.add_argument("--warm-start-price-col", default="bid_price")
    args = ap.parse_args()
    args.allow_small_exit = bool(args.allow_small_exit)

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("[read input]", args.input_glob)
    raw = read_glob(args.input_glob)
    df = normalize_market_input(raw, args.signal_col)
    print("[input normalized]", df.shape)
    print("[date]", df["date"].min(), "->", df["date"].max(), "n=", df["date"].nunique())
    print("[all minutes]", df[["date", "datetime"]].drop_duplicates().shape[0])
    print("[symbols]", df["securityid"].nunique())

    rb_times = select_rebalance_times(df, args)
    print("[rebalance times]", rb_times.shape)
    print(rb_times.groupby("date").size().describe())

    # Filter to rebalance times.
    df = df.merge(rb_times, on=["date", "datetime"], how="inner")
    df = df.sort_values(["date", "datetime", "securityid"]).reset_index(drop=True)

    state = {
        "cash": float(args.capital),
        "positions": {},
        "sellable": {},
        "bought_today": {},
        "is_initial_build": True,
    }


    # warm-start benchmark portfolio
    # Assume at OOS start we already hold the benchmark portfolio.
    # Initial holdings are fully sellable and initial build cost is ignored.
    if int(getattr(args, "warm_start_benchmark", 0)) == 1:
        first_key = (
            df[["date", "datetime"]]
            .drop_duplicates()
            .sort_values(["date", "datetime"])
            .iloc[0]
        )
        first_date = int(first_key["date"])
        first_dt = pd.to_datetime(first_key["datetime"])

        first_cur = df[
            (df["date"].astype(int) == first_date)
            & (pd.to_datetime(df["datetime"]) == first_dt)
        ].copy()

        if "benchmark_weight" not in first_cur.columns:
            raise RuntimeError("warm_start_benchmark requires benchmark_weight column")

        px_col = str(getattr(args, "warm_start_price_col", "bid_price"))
        if px_col not in first_cur.columns:
            px_col = "bid_price" if "bid_price" in first_cur.columns else "mid_price"

        warm_gross = float(getattr(args, "warm_start_gross", -1.0))
        if warm_gross <= 0:
            warm_gross = float(args.gross_target)

        bench = pd.to_numeric(first_cur["benchmark_weight"], errors="coerce").fillna(0.0).clip(lower=0.0)
        bench_sum = float(bench.sum())
        if bench_sum <= 1e-12:
            raise RuntimeError("benchmark_weight sum is zero at warm start")

        first_cur["warm_weight"] = bench / bench_sum * warm_gross

        init_positions = {}
        init_notional = 0.0

        for r in first_cur.itertuples(index=False):
            sid = str(getattr(r, "securityid")).zfill(6)
            w = float(getattr(r, "warm_weight"))
            px = float(getattr(r, px_col))
            if px <= 0 or w <= 0:
                continue
            sh = floor_lot(float(args.capital) * w / px, int(args.lot_size))
            if sh <= 0:
                continue
            init_positions[sid] = int(sh)
            init_notional += sh * px

        state["positions"] = init_positions.copy()
        state["sellable"] = init_positions.copy()
        state["bought_today"] = {}
        state["cash"] = float(args.capital) - init_notional
        state["is_initial_build"] = False

        print("\n===== WARM START BENCHMARK =====")
        print("first_date:", first_date)
        print("first_dt:", first_dt)
        print("warm_gross:", warm_gross)
        print("price_col:", px_col)
        print("init_names:", len(init_positions))
        print("init_notional:", init_notional)
        print("init_cash:", state["cash"])
        print("init_gross_to_capital:", init_notional / float(args.capital))
        print("================================\n")


    # ===== WARM START BENCHMARK PORTFOLIO PATCH =====
    # Assume OOS starts with an already established benchmark portfolio.
    # Initial holdings are sellable, and initial build cost is ignored.
    if int(getattr(args, "warm_start_benchmark", 0)) == 1:
        first_key = (
            df[["date", "datetime"]]
            .drop_duplicates()
            .sort_values(["date", "datetime"])
            .iloc[0]
        )
        first_date = int(first_key["date"])
        first_dt = pd.to_datetime(first_key["datetime"])

        first_cur = df[
            (df["date"].astype(int) == first_date)
            & (pd.to_datetime(df["datetime"]) == first_dt)
        ].copy()

        if "benchmark_weight" not in first_cur.columns:
            raise RuntimeError("warm_start_benchmark requires benchmark_weight column in input data")

        px_col = str(getattr(args, "warm_start_price_col", "bid_price"))
        if px_col not in first_cur.columns:
            px_col = "bid_price" if "bid_price" in first_cur.columns else "mid_price"
        if px_col not in first_cur.columns:
            raise RuntimeError("cannot find warm-start price column")

        warm_gross = float(getattr(args, "warm_start_gross", -1.0))
        if warm_gross <= 0:
            warm_gross = float(args.gross_target)

        bench = pd.to_numeric(first_cur["benchmark_weight"], errors="coerce").fillna(0.0).clip(lower=0.0)
        bench_sum = float(bench.sum())
        if bench_sum <= 1e-12:
            raise RuntimeError("benchmark_weight sum is zero at warm start")

        first_cur["warm_weight"] = bench / bench_sum * warm_gross

        init_positions = {}
        init_notional = 0.0

        for r in first_cur.itertuples(index=False):
            sid = str(getattr(r, "securityid")).zfill(6)
            w = float(getattr(r, "warm_weight"))
            px = float(getattr(r, px_col))
            if px <= 0 or w <= 0:
                continue
            sh = floor_lot(float(args.capital) * w / px, int(args.lot_size))
            if sh <= 0:
                continue
            init_positions[sid] = int(sh)
            init_notional += sh * px

        state["positions"] = init_positions.copy()
        state["sellable"] = init_positions.copy()
        state["bought_today"] = {}
        state["cash"] = float(args.capital) - init_notional
        state["is_initial_build"] = False

        print("\n===== WARM START BENCHMARK =====")
        print("first_date:", first_date)
        print("first_dt:", first_dt)
        print("warm_gross:", warm_gross)
        print("price_col:", px_col)
        print("init_names:", len(init_positions))
        print("init_notional:", init_notional)
        print("init_cash:", state["cash"])
        print("init_gross_to_capital:", init_notional / float(args.capital))
        print("================================\n")
    # ===== END WARM START BENCHMARK PORTFOLIO PATCH =====

    target_rows = []
    summary_rows = []
    trade_rows = []
    executed_rows = []

    current_date = None
    daily_turnover_used = 0.0

    for (date, dt), cur in df.groupby(["date", "datetime"], sort=True):
        cur = cur.sort_values("securityid").reset_index(drop=True)

        if current_date is None or date != current_date:
            current_date = date
            daily_turnover_used = 0.0
            state["sellable"] = {sid: int(sh) for sid, sh in state["positions"].items() if sh > 0}
            state["bought_today"] = {}

        sol = solve_one_rebalance(cur, state, args, daily_turnover_used)
        out_rows, trades, diag = execute_to_target(cur, sol, state, args)
        daily_turnover_used += diag["turnover_weight"]
        state["is_initial_build"] = False

        mark = mark_state(cur, state, args)

        for tr in trades:
            tr["date"] = int(date)
            tr["datetime"] = pd.to_datetime(dt)
            trade_rows.append(tr)

        target_rows.extend(out_rows)

        for sid, sh in state["positions"].items():
            if sh <= 0:
                continue
            row = cur[cur["securityid"] == sid]
            if row.empty:
                continue
            r = row.iloc[0]
            px = float(r["bid_price"])
            executed_rows.append({
                "date": int(date),
                "datetime": pd.to_datetime(dt),
                "securityid": sid,
                "actual_shares": int(sh),
                "sellable_shares": int(state["sellable"].get(sid, 0)),
                "bought_today_shares": int(state["bought_today"].get(sid, 0)),
                "mark_price": px,
                "actual_weight": sh * px / args.capital,
            })

        target_gross = float(np.sum(sol["weights"]))
        active_l1 = float(np.sum(np.abs(sol["weights"] - sol["benchmark"])))
        locked_sum = float(sol["locked_sum"])
        actual_before = float(np.sum(sol["actual_w_before"]))
        sellable_before = float(np.sum(sol["sellable_w_before"]))

        summary_rows.append({
            "date": int(date),
            "datetime": pd.to_datetime(dt),
            "status": sol["status"],
            "solver": sol["solver"],
            "objective": sol["objective"],
            "n_names": len(sol["sids"]),
            "target_gross": target_gross,
            "actual_gross_after": mark["actual_gross"],
            "actual_net_after": mark["actual_net"],
            "actual_gross_before_weight_sum": actual_before,
            "sellable_weight_before": sellable_before,
            "locked_weight_before": locked_sum,
            "sellable_gross_after": mark["sellable_gross"],
            "locked_gross_after": mark["locked_gross"],
            "cash_weight_after": mark["cash_weight"],
            "active_l1": active_l1,
            "turnover_weight": diag["turnover_weight"],
            "daily_turnover_used": daily_turnover_used,
            "turnover_notional": diag["turnover_notional"],
            "fee": diag["fee"],
            "spread_cost_est": diag["spread_cost_est"],
            "total_cost": diag["total_cost"],
            "n_buy": diag["n_buy"],
            "n_sell": diag["n_sell"],
            "n_trade": diag["n_trade"],
            "blocked_tplus1": diag["blocked_tplus1"],
            "clipped_tplus1": diag["clipped_tplus1"],
            "blocked_cash": diag["blocked_cash"],
            "clipped_cash": diag["clipped_cash"],
            "blocked_small": diag["blocked_small"],
            "blocked_no_price": diag["blocked_no_price"],
            "dynamic_gross_min": sol["dynamic_gross_min"],
            "dynamic_gross_max": sol["dynamic_gross_max"],
            "turnover_limit": sol["turnover_limit"],
        })

        if len(summary_rows) % 20 == 0:
            print(f"[progress] {len(summary_rows)} / {len(rb_times)} {date} {dt} "
                  f"status={sol['status']} target_gross={target_gross:.4f} actual_gross={mark['actual_gross']:.4f}")

    target_df = pd.DataFrame(target_rows)
    summary_df = pd.DataFrame(summary_rows)
    trades_df = pd.DataFrame(trade_rows)
    executed_df = pd.DataFrame(executed_rows)

    target_path = outdir / "target_positions.csv"
    summary_path = outdir / "summary_by_rebalance.csv"
    trades_path = outdir / "trades.csv"
    executed_path = outdir / "executed_positions.csv"

    target_df.to_csv(target_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    trades_df.to_csv(trades_path, index=False)
    executed_df.to_csv(executed_path, index=False)

    print("\n===== final summary =====")
    print("target_path:", target_path)
    print("summary_path:", summary_path)
    print("trades_path:", trades_path)
    print("executed_path:", executed_path)
    print("num_rebalances:", len(summary_df))
    print("avg_target_gross:", summary_df["target_gross"].mean())
    print("avg_actual_gross_after:", summary_df["actual_gross_after"].mean())
    print("avg_locked_weight_before:", summary_df["locked_weight_before"].mean())
    print("avg_sellable_weight_before:", summary_df["sellable_weight_before"].mean())
    print("total_turnover_weight:", summary_df["turnover_weight"].sum())
    print("total_cost:", summary_df["total_cost"].sum())
    print("blocked_tplus1:", summary_df["blocked_tplus1"].sum())
    print("clipped_tplus1:", summary_df["clipped_tplus1"].sum())
    print("blocked_cash:", summary_df["blocked_cash"].sum())
    print("blocked_small:", summary_df["blocked_small"].sum())
    print("\nstatus counts:")
    print(summary_df["status"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
