# -*- coding: utf-8 -*-
"""
v15 two-clock target-execution optimizer.

Target clock:     h20 -> theoretical target position every 20 minutes.
Execution clock:  h10 -> execution urgency every 10 minutes.

Main idea:
- h20 rank-gated value-conviction score generates w_theory.
- h10 rank-gated value-conviction score only controls how fast we move toward w_theory.
- Execution can only trade toward h20 theoretical target, not reverse away from it.

This script is self-contained: it prepares signals from h10/h20 input dirs, runs optimizer,
tracks actual shares/cash/T+1, writes nav curve, summary, rebalance diagnostics and PNG.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import cvxpy as cp
except Exception as e:  # pragma: no cover
    raise RuntimeError("cvxpy is required for v15 optimizer") from e


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def safe_float(x, default=np.nan) -> float:
    try:
        y = float(x)
        if not np.isfinite(y):
            return default
        return y
    except Exception:
        return default


def parse_list(s: str, typ=float) -> List:
    if s is None or str(s).strip() == "":
        return []
    return [typ(x.strip()) for x in str(s).split(",") if x.strip()]


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def pick_col(cols: Iterable[str], candidates: List[str]) -> Optional[str]:
    s = set(cols)
    for c in candidates:
        if c in s:
            return c
    return None


def normalize_sid(s: pd.Series) -> pd.Series:
    # ZZY sid is internal id. Do NOT pad as securityid.
    return s.astype(str)


def compound_return(x: pd.Series) -> float:
    x = pd.Series(x).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return float((1.0 + x).prod() - 1.0)


def annualized_sharpe(x: pd.Series) -> float:
    x = pd.Series(x).replace([np.inf, -np.inf], np.nan).dropna()
    if len(x) < 2:
        return np.nan
    sd = x.std(ddof=1)
    if not np.isfinite(sd) or sd == 0:
        return np.nan
    return float(x.mean() / sd * np.sqrt(252.0))


# -----------------------------------------------------------------------------
# Cross-sectional score: rank gate * value conviction
# -----------------------------------------------------------------------------


def _cs_robust_z_one(s: pd.Series, clip_z: float) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    med = x.median(skipna=True)
    mad = (x - med).abs().median(skipna=True)
    if not np.isfinite(mad) or mad <= 1e-12:
        mu = x.mean(skipna=True)
        sd = x.std(skipna=True)
        if not np.isfinite(sd) or sd <= 1e-12:
            return pd.Series(0.0, index=s.index)
        z = (x - mu) / sd
    else:
        z = (x - med) / (1.4826 * mad)
    return z.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-clip_z, clip_z)


def _cs_rank_score_one(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    pct = x.rank(method="average", pct=True)
    r = 2.0 * pct - 1.0
    return r.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-1.0, 1.0)


def add_rank_value_score(
    df: pd.DataFrame,
    pred_col: str,
    out_col: str,
    *,
    r0: float = 0.25,
    z0: float = 2.0,
    clip_z: float = 5.0,
    group_col: str = "datetime",
) -> pd.DataFrame:
    """Add rank-gated value-conviction score for pred_col.

    For each timestamp:
        value_z = robust_z(pred)
        r = 2 * percentile_rank(pred) - 1
        rank_gate = max(|r| - r0, 0) / (1 - r0)
        value_conviction = tanh(|value_z| / z0)
        score = sign(r) * rank_gate * value_conviction

    This is deliberately NOT rank/value weighted average.
    """
    pieces = []
    for _, g in df.groupby(group_col, sort=False):
        idx = g.index
        val = _cs_robust_z_one(g[pred_col], clip_z=clip_z)
        r = _cs_rank_score_one(g[pred_col])
        rank_gate = ((r.abs() - r0).clip(lower=0.0) / max(1e-12, (1.0 - r0))).clip(0.0, 1.0)
        value_conv = np.tanh(val.abs() / max(1e-12, z0))
        score = np.sign(r) * rank_gate * value_conv
        # keep score bounded; do not zscore again, because bounded score is intended.
        pieces.append(pd.Series(score, index=idx))
    df[out_col] = pd.concat(pieces).sort_index().astype(float)
    return df


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------


@dataclass
class Config:
    h10_dir: Path
    h20_dir: Path
    output_dir: Path
    tag: str
    capital: float
    fee_bps: float
    lot_size: int
    min_trade_notional: float
    allow_small_exit: int
    gross_target: float
    gross_min: float
    gross_max: float
    single_name_cap: float
    active_l1_limit: float
    active_l1_relax_list: List[float]
    target_step_minutes: int
    execution_step_minutes: int
    turnover_limit: float
    daily_turnover_budget: float
    cash_buffer: float
    lambda_target_alpha: float
    lambda_target_active: float
    lambda_target_ridge: float
    lambda_dist: float
    lambda_turnover: float
    lambda_active_exec: float
    lambda_ridge_exec: float
    turncost_beta: float
    urgency_base: float
    urgency_amp: float
    urgency_min: float
    urgency_max: float
    score_r0: float
    score_z0: float
    score_clip_z: float
    max_names: int
    target_score_mode: str
    solvers: List[str]


def read_day_inputs(h20_file: Path, h10_dir: Path) -> pd.DataFrame:
    base = pd.read_parquet(h20_file).copy()
    h10_file = h10_dir / h20_file.name
    if not h10_file.exists():
        raise FileNotFoundError(f"h10 file not found: {h10_file}")
    h10 = pd.read_parquet(h10_file)

    for c in ["date", "ts", "sid"]:
        if c not in base.columns:
            raise KeyError(f"{h20_file} missing required column {c}")
        if c not in h10.columns:
            raise KeyError(f"{h10_file} missing required column {c}")

    base["date"] = base["date"].astype(int)
    base["ts"] = base["ts"].astype(int)
    base["sid"] = normalize_sid(base["sid"])
    if "datetime" not in base.columns:
        raise KeyError(f"{h20_file} missing datetime")
    base["datetime"] = pd.to_datetime(base["datetime"])

    h10["date"] = h10["date"].astype(int)
    h10["ts"] = h10["ts"].astype(int)
    h10["sid"] = normalize_sid(h10["sid"])

    h20_sig = pick_col(base.columns, ["pred_ret_h20", "pred_z", "pred", "signal", "alpha"])
    if h20_sig is None:
        raise KeyError(f"cannot find h20 signal column in {h20_file}; cols={list(base.columns)}")
    h10_sig = pick_col(h10.columns, ["pred_ret_h10", "pred_z", "pred_ret_h20", "pred", "signal", "alpha"])
    if h10_sig is None:
        raise KeyError(f"cannot find h10 signal column in {h10_file}; cols={list(h10.columns)}")

    h10_small = h10[["date", "ts", "sid", h10_sig]].rename(columns={h10_sig: "pred_ret_h10"})
    if h20_sig != "pred_ret_h20":
        base = base.rename(columns={h20_sig: "pred_ret_h20"})

    df = base.merge(h10_small, on=["date", "ts", "sid"], how="left")
    df["pred_ret_h20"] = pd.to_numeric(df["pred_ret_h20"], errors="coerce").fillna(0.0)
    df["pred_ret_h10"] = pd.to_numeric(df["pred_ret_h10"], errors="coerce").fillna(0.0)

    # Price columns standardization
    price_col = pick_col(df.columns, ["mid_price", "mid", "tmid", "last_price", "lastprice"])
    bid_col = pick_col(df.columns, ["bid1", "bid_price", "bidprice1"])
    ask_col = pick_col(df.columns, ["ask1", "ask_price", "askprice1"])
    if price_col is None:
        if bid_col is not None and ask_col is not None:
            df["mid_price"] = (pd.to_numeric(df[bid_col], errors="coerce") + pd.to_numeric(df[ask_col], errors="coerce")) / 2.0
        else:
            raise KeyError(f"cannot find price columns in {h20_file}")
    elif price_col != "mid_price":
        df["mid_price"] = pd.to_numeric(df[price_col], errors="coerce")

    if bid_col is None:
        df["bid1"] = df["mid_price"]
    elif bid_col != "bid1":
        df["bid1"] = pd.to_numeric(df[bid_col], errors="coerce")
    if ask_col is None:
        df["ask1"] = df["mid_price"]
    elif ask_col != "ask1":
        df["ask1"] = pd.to_numeric(df[ask_col], errors="coerce")

    if "benchmark_weight" not in df.columns:
        df["benchmark_weight"] = 0.0
    df["benchmark_weight"] = pd.to_numeric(df["benchmark_weight"], errors="coerce").fillna(0.0).clip(lower=0.0)

    # Volume columns optional
    for c in ["bid_volume1", "ask_volume1", "turnover_amount"]:
        if c not in df.columns:
            df[c] = np.nan
        else:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "securityid" not in df.columns:
        df["securityid"] = df["sid"].astype(str)
    else:
        df["securityid"] = df["securityid"].astype(str).str.zfill(6)

    # Limit columns optional: treat numeric price limits if available.
    if "limit_up" not in df.columns:
        df["limit_up"] = np.nan
    if "limit_down" not in df.columns:
        df["limit_down"] = np.nan

    df = df.sort_values(["datetime", "securityid"]).reset_index(drop=True)

    # Add scores.
    df = add_rank_value_score(
        df, "pred_ret_h20", "s20_position",
        r0=GLOBAL_CFG.score_r0, z0=GLOBAL_CFG.score_z0, clip_z=GLOBAL_CFG.score_clip_z,
    )
    df = add_rank_value_score(
        df, "pred_ret_h10", "s10_timing",
        r0=GLOBAL_CFG.score_r0, z0=GLOBAL_CFG.score_z0, clip_z=GLOBAL_CFG.score_clip_z,
    )

    # v15b sanity mode:
    # Use existing pred_ret_h20 as target score. If h20_dir is mix_406000,
    # this means target score = simple 0.4 h10 + 0.6 h20, close to old baseline.
    mode = getattr(GLOBAL_CFG, "target_score_mode", "rank_gate")
    if mode == "raw_h20":
        df["s20_position"] = pd.to_numeric(df["pred_ret_h20"], errors="coerce").fillna(0.0).clip(
            -GLOBAL_CFG.score_clip_z, GLOBAL_CFG.score_clip_z
        )
    elif mode == "csz_h20":
        pieces = []
        for _, gg in df.groupby("datetime", sort=False):
            pieces.append(_cs_robust_z_one(gg["pred_ret_h20"], clip_z=GLOBAL_CFG.score_clip_z))
        df["s20_position"] = pd.concat(pieces).sort_index().astype(float)
    elif mode == "rank_gate":
        pass
    else:
        raise ValueError(f"unknown target_score_mode={mode}")

    return df


# This global is set in main to avoid passing config deeply in read_day_inputs.
GLOBAL_CFG: Config


# -----------------------------------------------------------------------------
# Optimizers
# -----------------------------------------------------------------------------


def solve_cvx_problem(prob: cp.Problem, solvers: List[str]) -> Tuple[bool, str]:
    last_status = "not_solved"
    for solver in solvers:
        try:
            prob.solve(solver=solver, verbose=False)
            last_status = str(prob.status)
            if prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                return True, last_status
        except Exception as e:
            last_status = f"{solver}_error:{type(e).__name__}"
            continue
    return False, last_status


def solve_target_optimizer(g: pd.DataFrame, cfg: Config) -> Tuple[pd.Series, str]:
    n = len(g)
    if n == 0:
        return pd.Series(dtype=float), "empty"

    sid = g["securityid"].astype(str).to_numpy()
    alpha = pd.to_numeric(g["s20_position"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    b = pd.to_numeric(g["benchmark_weight"], errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(dtype=float)
    if b.sum() > 0:
        b = b / b.sum() * cfg.gross_target
    else:
        b = np.ones(n) / n * cfg.gross_target

    # Restrict max names if requested: keep all benchmark names + score extremes.
    # In most runs keep max_names <=0, i.e. all names.
    w = cp.Variable(n)
    constraints = [
        w >= 0,
        w <= cfg.single_name_cap,
        cp.sum(w) >= cfg.gross_min,
        cp.sum(w) <= cfg.gross_max,
        cp.norm1(w - b) <= cfg.active_l1_limit,
    ]

    obj = cp.Maximize(
        cfg.lambda_target_alpha * alpha @ (w - b)
        - cfg.lambda_target_active * cp.sum_squares(w - b)
        - cfg.lambda_target_ridge * cp.sum_squares(w)
    )
    prob = cp.Problem(obj, constraints)
    ok, status = solve_cvx_problem(prob, cfg.solvers)

    if not ok or w.value is None:
        # fallback: benchmark gross portfolio.
        out = pd.Series(b, index=sid)
        return out, f"target_fallback_{status}"

    val = np.asarray(w.value, dtype=float)
    val = np.nan_to_num(val, nan=0.0, posinf=0.0, neginf=0.0).clip(min=0.0)
    out = pd.Series(val, index=sid)
    return out, status


@dataclass
class PortfolioState:
    actual_shares: Dict[str, int]
    sellable_shares: Dict[str, int]
    cash: float
    initialized: bool = False
    current_date: Optional[int] = None


def initialize_benchmark_state(g: pd.DataFrame, cfg: Config) -> PortfolioState:
    b = pd.to_numeric(g["benchmark_weight"], errors="coerce").fillna(0.0).clip(lower=0.0)
    if b.sum() <= 0:
        b = pd.Series(1.0 / len(g), index=g.index)
    else:
        b = b / b.sum()

    shares: Dict[str, int] = {}
    used = 0.0
    for idx, row in g.iterrows():
        sid = str(row["securityid"])
        px = safe_float(row["mid_price"], np.nan)
        if not np.isfinite(px) or px <= 0:
            continue
        target_notional = cfg.capital * cfg.gross_target * float(b.loc[idx])
        sh = int(math.floor(target_notional / px / cfg.lot_size) * cfg.lot_size)
        if sh > 0:
            shares[sid] = sh
            used += sh * px
    cash = cfg.capital - used
    return PortfolioState(actual_shares=shares.copy(), sellable_shares=shares.copy(), cash=float(cash), initialized=True)


def mark_equity(state: PortfolioState, g: pd.DataFrame) -> Tuple[float, Dict[str, float]]:
    price_map = dict(zip(g["securityid"].astype(str), pd.to_numeric(g["mid_price"], errors="coerce")))
    mv = 0.0
    w_price: Dict[str, float] = {}
    for sid, sh in state.actual_shares.items():
        px = safe_float(price_map.get(sid, np.nan), np.nan)
        if np.isfinite(px) and px > 0 and sh != 0:
            mv += sh * px
            w_price[sid] = px
    equity = float(state.cash + mv)
    return equity, w_price


def get_current_weight_arrays(state: PortfolioState, g: pd.DataFrame, equity: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    sec = g["securityid"].astype(str).to_numpy()
    mid = pd.to_numeric(g["mid_price"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    actual = np.zeros(len(g), dtype=float)
    sellable = np.zeros(len(g), dtype=float)
    for i, sid in enumerate(sec):
        sh = state.actual_shares.get(sid, 0)
        sell_sh = state.sellable_shares.get(sid, 0)
        actual[i] = sh * mid[i] / max(equity, 1e-12)
        sellable[i] = sell_sh * mid[i] / max(equity, 1e-12)
    locked = np.maximum(actual - sellable, 0.0)
    return actual, sellable, locked


def is_limit_up(row) -> bool:
    lu = safe_float(getattr(row, "limit_up", np.nan), np.nan)
    ask = safe_float(getattr(row, "ask1", np.nan), np.nan)
    if np.isfinite(lu) and lu > 0 and np.isfinite(ask) and ask >= lu * 0.999:
        return True
    return False


def is_limit_down(row) -> bool:
    ld = safe_float(getattr(row, "limit_down", np.nan), np.nan)
    bid = safe_float(getattr(row, "bid1", np.nan), np.nan)
    if np.isfinite(ld) and ld > 0 and np.isfinite(bid) and bid <= ld * 1.001:
        return True
    return False


def solve_execution_optimizer(
    g: pd.DataFrame,
    state: PortfolioState,
    w_theory_map: Dict[str, float],
    cfg: Config,
    active_limit: float,
) -> Tuple[np.ndarray, np.ndarray, str, Dict[str, float]]:
    n = len(g)
    sec = g["securityid"].astype(str).to_numpy()

    equity, _ = mark_equity(state, g)
    if equity <= 0:
        return np.zeros(n), np.zeros(n), "bad_equity", {}

    w_cur, w_sellable, w_locked = get_current_weight_arrays(state, g, equity)
    w_theory = np.array([float(w_theory_map.get(s, 0.0)) for s in sec], dtype=float)

    b = pd.to_numeric(g["benchmark_weight"], errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(dtype=float)
    if b.sum() > 0:
        b = b / b.sum() * cfg.gross_target
    else:
        b = np.ones(n) / n * cfg.gross_target

    s10 = pd.to_numeric(g["s10_timing"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    gap = w_theory - w_cur
    direction = np.sign(gap)
    alignment = direction * s10
    urgency = np.clip(
        cfg.urgency_base + cfg.urgency_amp * np.tanh(alignment),
        cfg.urgency_min,
        cfg.urgency_max,
    )

    buy_gap_cap = np.maximum(gap, 0.0) * urgency
    sell_gap_cap = np.maximum(-gap, 0.0) * urgency

    # Limit up/down protection.
    for i, row in enumerate(g.itertuples(index=False)):
        if is_limit_up(row):
            buy_gap_cap[i] = 0.0
        if is_limit_down(row):
            sell_gap_cap[i] = 0.0

    buy = cp.Variable(n, nonneg=True)
    sell = cp.Variable(n, nonneg=True)
    w_next = w_cur + buy - sell

    turn_cost = cfg.lambda_turnover * np.clip(1.0 - cfg.turncost_beta * np.tanh(alignment), 0.35, 1.75)
    cash_weight = state.cash / equity

    constraints = [
        buy <= buy_gap_cap,
        sell <= sell_gap_cap,
        sell <= w_sellable,
        w_next >= w_locked,
        w_next >= 0,
        w_next <= cfg.single_name_cap,
        cp.sum(w_next) >= cfg.gross_min,
        cp.sum(w_next) <= cfg.gross_max,
        cp.norm1(w_next - b) <= active_limit,
        cp.sum(buy + sell) <= cfg.turnover_limit,
        (1.0 + cfg.fee_bps / 10000.0) * cp.sum(buy)
        <= cash_weight + (1.0 - cfg.fee_bps / 10000.0) * cp.sum(sell) - cfg.cash_buffer,
    ]

    obj = cp.Minimize(
        cfg.lambda_dist * cp.sum_squares(w_next - w_theory)
        + turn_cost @ (buy + sell)
        + cfg.lambda_active_exec * cp.sum_squares(w_next - b)
        + cfg.lambda_ridge_exec * cp.sum_squares(w_next)
    )
    prob = cp.Problem(obj, constraints)
    ok, status = solve_cvx_problem(prob, cfg.solvers)

    if not ok or buy.value is None or sell.value is None:
        return np.zeros(n), np.zeros(n), f"exec_fallback_{status}", {
            "equity": equity,
            "active_limit_used": active_limit,
            "gross_before": float(w_cur.sum()),
            "gross_theory": float(w_theory.sum()),
            "urgency_mean": float(np.mean(urgency)),
        }

    buy_w = np.nan_to_num(np.asarray(buy.value, dtype=float), nan=0.0, posinf=0.0, neginf=0.0).clip(min=0.0)
    sell_w = np.nan_to_num(np.asarray(sell.value, dtype=float), nan=0.0, posinf=0.0, neginf=0.0).clip(min=0.0)
    diag = {
        "equity": equity,
        "active_limit_used": active_limit,
        "gross_before": float(w_cur.sum()),
        "gross_theory": float(w_theory.sum()),
        "gross_after_cont": float((w_cur + buy_w - sell_w).sum()),
        "urgency_mean": float(np.mean(urgency)),
        "urgency_p10": float(np.quantile(urgency, 0.10)),
        "urgency_p90": float(np.quantile(urgency, 0.90)),
        "alignment_mean": float(np.mean(alignment)),
        "alignment_p10": float(np.quantile(alignment, 0.10)),
        "alignment_p90": float(np.quantile(alignment, 0.90)),
    }
    return buy_w, sell_w, status, diag


# -----------------------------------------------------------------------------
# Execution state update and NAV
# -----------------------------------------------------------------------------


def execute_trades(
    g: pd.DataFrame,
    state: PortfolioState,
    buy_w: np.ndarray,
    sell_w: np.ndarray,
    cfg: Config,
) -> Dict[str, float]:
    equity, _ = mark_equity(state, g)
    fee_rate = cfg.fee_bps / 10000.0

    sec = g["securityid"].astype(str).to_numpy()
    bid = pd.to_numeric(g["bid1"], errors="coerce").fillna(pd.to_numeric(g["mid_price"], errors="coerce")).to_numpy(dtype=float)
    ask = pd.to_numeric(g["ask1"], errors="coerce").fillna(pd.to_numeric(g["mid_price"], errors="coerce")).to_numpy(dtype=float)

    # Sell first.
    sell_notional = 0.0
    sell_fee = 0.0
    sell_names = 0
    for i, sid in enumerate(sec):
        px = bid[i]
        if not np.isfinite(px) or px <= 0:
            continue
        desired_notional = max(0.0, float(sell_w[i]) * equity)
        if desired_notional < cfg.min_trade_notional and not cfg.allow_small_exit:
            continue
        desired_shares = int(math.floor(desired_notional / px / cfg.lot_size) * cfg.lot_size)
        avail = int(state.sellable_shares.get(sid, 0))
        sh = max(0, min(desired_shares, avail))
        if sh <= 0:
            continue
        notional = sh * px
        if notional < cfg.min_trade_notional and not cfg.allow_small_exit:
            continue
        fee = notional * fee_rate
        state.actual_shares[sid] = int(state.actual_shares.get(sid, 0) - sh)
        state.sellable_shares[sid] = int(state.sellable_shares.get(sid, 0) - sh)
        if state.actual_shares.get(sid, 0) <= 0:
            state.actual_shares.pop(sid, None)
        if state.sellable_shares.get(sid, 0) <= 0:
            state.sellable_shares.pop(sid, None)
        state.cash += notional - fee
        sell_notional += notional
        sell_fee += fee
        sell_names += 1

    # Buy second.
    buy_notional = 0.0
    buy_fee = 0.0
    buy_names = 0
    for i, sid in enumerate(sec):
        px = ask[i]
        if not np.isfinite(px) or px <= 0:
            continue
        desired_notional = max(0.0, float(buy_w[i]) * equity)
        if desired_notional < cfg.min_trade_notional:
            continue
        max_affordable_shares = int(math.floor((state.cash / (1.0 + fee_rate)) / px / cfg.lot_size) * cfg.lot_size)
        desired_shares = int(math.floor(desired_notional / px / cfg.lot_size) * cfg.lot_size)
        sh = max(0, min(desired_shares, max_affordable_shares))
        if sh <= 0:
            continue
        notional = sh * px
        if notional < cfg.min_trade_notional:
            continue
        fee = notional * fee_rate
        state.actual_shares[sid] = int(state.actual_shares.get(sid, 0) + sh)
        # T+1: bought today is not sellable, so do not add to sellable_shares.
        state.cash -= notional + fee
        buy_notional += notional
        buy_fee += fee
        buy_names += 1

    total_notional = buy_notional + sell_notional
    total_fee = buy_fee + sell_fee
    return {
        "buy_notional": buy_notional,
        "sell_notional": sell_notional,
        "turnover_notional": total_notional,
        "fee": total_fee,
        "buy_names": buy_names,
        "sell_names": sell_names,
    }


def benchmark_step_return(g: pd.DataFrame, prev_price: Dict[str, float]) -> float:
    weights = pd.to_numeric(g["benchmark_weight"], errors="coerce").fillna(0.0).clip(lower=0.0)
    if weights.sum() <= 0:
        return 0.0
    weights = weights / weights.sum()
    mids = pd.to_numeric(g["mid_price"], errors="coerce")
    ret_sum = 0.0
    w_sum = 0.0
    for sid, w, px in zip(g["securityid"].astype(str), weights, mids):
        p0 = prev_price.get(sid, np.nan)
        if np.isfinite(p0) and p0 > 0 and np.isfinite(px) and px > 0:
            ret_sum += float(w) * (float(px) / float(p0) - 1.0)
            w_sum += float(w)
    if w_sum <= 0:
        return 0.0
    return ret_sum / w_sum


# -----------------------------------------------------------------------------
# Main backtest loop
# -----------------------------------------------------------------------------


def should_exec(dt: pd.Timestamp, first_dt: pd.Timestamp, step_minutes: int) -> bool:
    diff_min = int(round((dt - first_dt).total_seconds() / 60.0))
    return diff_min >= 0 and diff_min % step_minutes == 0


def run_backtest(cfg: Config) -> None:
    ensure_dir(cfg.output_dir)
    ensure_dir(cfg.output_dir / "logs")

    h20_files = sorted(cfg.h20_dir.glob("*.parquet"))
    if not h20_files:
        raise FileNotFoundError(cfg.h20_dir)

    # Load all days one by one to avoid enormous memory? We keep per-day loop.
    state: Optional[PortfolioState] = None
    last_theory: Dict[str, float] = {}
    nav_rows = []
    reb_rows = []
    target_rows = []
    position_rows = []

    prev_equity: Optional[float] = None
    prev_bench_prices: Dict[str, float] = {}
    prev_date: Optional[int] = None
    cumulative_cost = 0.0
    cumulative_turnover = 0.0
    daily_turnover: Dict[int, float] = {}

    for file_idx, h20_file in enumerate(h20_files, 1):
        print(f"\n===== load day {file_idx}/{len(h20_files)}: {h20_file.name} =====", flush=True)
        day_df = read_day_inputs(h20_file, cfg.h10_dir)
        d0 = int(day_df["date"].iloc[0])

        # Market minutes available for the day.
        minutes = list(day_df["datetime"].drop_duplicates().sort_values())
        if not minutes:
            continue
        first_dt = minutes[0]

        # Reset sellable at day start; no overnight PnL in return curve.
        if state is not None:
            state.sellable_shares = state.actual_shares.copy()
            state.current_date = d0
            prev_equity = None
            prev_bench_prices = {}
        daily_turnover.setdefault(d0, 0.0)

        for dt in minutes:
            g = day_df[day_df["datetime"] == dt].copy().reset_index(drop=True)
            if len(g) == 0:
                continue

            if state is None or not state.initialized:
                state = initialize_benchmark_state(g, cfg)
                state.current_date = d0
                prev_equity = None
                prev_bench_prices = {}
                print(f"[init] {dt} cash={state.cash:.2f} names={len(state.actual_shares)}", flush=True)

            assert state is not None

            # Target update every 20 minutes.
            target_update = should_exec(dt, first_dt, cfg.target_step_minutes)
            execution_update = should_exec(dt, first_dt, cfg.execution_step_minutes)

            target_status = "not_update"
            if target_update:
                w_theory, target_status = solve_target_optimizer(g, cfg)
                last_theory = {str(k): float(v) for k, v in w_theory.items()}
                # store target diagnostics for this timestamp.
                for sid, val in last_theory.items():
                    target_rows.append({
                        "date": d0,
                        "datetime": dt,
                        "securityid": sid,
                        "w_theory": val,
                    })

            trade_info = {
                "buy_notional": 0.0,
                "sell_notional": 0.0,
                "turnover_notional": 0.0,
                "fee": 0.0,
                "buy_names": 0,
                "sell_names": 0,
            }
            exec_status = "not_exec"
            diag = {}

            if execution_update and last_theory:
                # Daily budget check.
                equity_before, _ = mark_equity(state, g)
                budget_left = max(0.0, cfg.daily_turnover_budget - daily_turnover.get(d0, 0.0))
                old_turnover_limit = cfg.turnover_limit
                cfg.turnover_limit = min(cfg.turnover_limit, budget_left)

                if cfg.turnover_limit > 1e-8:
                    # Try active L1 base then relax list.
                    all_active = [cfg.active_l1_limit] + [x for x in cfg.active_l1_relax_list if x > cfg.active_l1_limit]
                    buy_w = sell_w = np.zeros(len(g), dtype=float)
                    for active_limit in all_active:
                        buy_w, sell_w, exec_status, diag = solve_execution_optimizer(
                            g, state, last_theory, cfg, active_limit=active_limit
                        )
                        if not exec_status.startswith("exec_fallback"):
                            break
                    trade_info = execute_trades(g, state, buy_w, sell_w, cfg)
                    equity_after_trade, _ = mark_equity(state, g)
                    turn_w = trade_info["turnover_notional"] / max(equity_before, 1e-12)
                    daily_turnover[d0] = daily_turnover.get(d0, 0.0) + turn_w
                    cumulative_turnover += turn_w
                    cumulative_cost += trade_info["fee"]
                else:
                    exec_status = "skip_daily_turnover_budget"

                cfg.turnover_limit = old_turnover_limit

            # Mark after any trade.
            equity, _ = mark_equity(state, g)
            if prev_equity is None or prev_date != d0:
                actual_ret = 0.0
            else:
                actual_ret = equity / max(prev_equity, 1e-12) - 1.0

            if prev_date != d0:
                bench_ret = 0.0
            else:
                bench_ret = benchmark_step_return(g, prev_bench_prices)

            # Update previous benchmark prices.
            prev_bench_prices = dict(zip(g["securityid"].astype(str), pd.to_numeric(g["mid_price"], errors="coerce")))
            prev_equity = equity
            prev_date = d0

            gross = 0.0
            if equity > 0:
                for row in g.itertuples(index=False):
                    sid = str(row.securityid)
                    sh = state.actual_shares.get(sid, 0)
                    px = safe_float(getattr(row, "mid_price"), np.nan)
                    if sh and np.isfinite(px) and px > 0:
                        gross += sh * px / equity

            # Export actual shares snapshot for canonical evaluator.
            # Include all names in current market universe, including zero shares,
            # so ffill will not keep stale positions after complete exits.
            for _row in g.itertuples(index=False):
                _sid = str(getattr(_row, "securityid"))
                position_rows.append({
                    "date": d0,
                    "datetime": dt,
                    "securityid": _sid,
                    "actual_shares_after": int(state.actual_shares.get(_sid, 0)),
                })

            nav_rows.append({
                "date": d0,
                "datetime": dt,
                "actual_ret": actual_ret,
                "benchmark_ret": bench_ret,
                "equity": equity,
                "cash": state.cash,
                "gross_prev_to_equity": gross,
                "turnover_weight": trade_info["turnover_notional"] / max(equity, 1e-12),
                "total_cost": trade_info["fee"],
                "cum_turnover_weight": cumulative_turnover,
                "cum_total_cost": cumulative_cost,
            })

            if target_update or execution_update:
                reb_rows.append({
                    "date": d0,
                    "datetime": dt,
                    "target_update": int(target_update),
                    "execution_update": int(execution_update),
                    "target_status": target_status,
                    "exec_status": exec_status,
                    "equity": equity,
                    "cash": state.cash,
                    "gross": gross,
                    "daily_turnover_used": daily_turnover.get(d0, 0.0),
                    "total_cost": float(trade_info.get("fee", 0.0)),
                    "turnover_weight": float(trade_info.get("turnover_notional", 0.0)) / max(float(equity), 1e-12),
                    **trade_info,
                    **diag,
                })

        print(f"[day done] {d0} cash={state.cash:.2f} names={len(state.actual_shares)} daily_turn={daily_turnover.get(d0,0.0):.4f}", flush=True)

    nav = pd.DataFrame(nav_rows)
    if nav.empty:
        raise RuntimeError("empty nav")
    nav = nav.sort_values("datetime").reset_index(drop=True)
    nav["strategy_nav"] = (1.0 + nav["actual_ret"].fillna(0.0)).cumprod()
    nav["benchmark_nav"] = (1.0 + nav["benchmark_ret"].fillna(0.0)).cumprod()
    nav["alpha_ret"] = nav["actual_ret"] - nav["benchmark_ret"]
    nav["alpha_nav"] = (1.0 + nav["alpha_ret"].fillna(0.0)).cumprod()
    nav["strategy_cumret"] = nav["strategy_nav"] - 1.0
    nav["benchmark_cumret"] = nav["benchmark_nav"] - 1.0
    nav["alpha_cumret"] = nav["alpha_nav"] - 1.0

    daily = nav.groupby("date", as_index=False).agg(
        actual_day=("actual_ret", compound_return),
        benchmark_day=("benchmark_ret", compound_return),
        alpha_day=("alpha_ret", compound_return),
        turnover=("turnover_weight", "sum"),
        cost=("total_cost", "sum"),
        avg_gross=("gross_prev_to_equity", "mean"),
    )

    strategy_return = compound_return(nav["actual_ret"])
    benchmark_return = compound_return(nav["benchmark_ret"])
    alpha_return = strategy_return - benchmark_return
    daily_excess_sharpe = annualized_sharpe(daily["alpha_day"])

    reb = pd.DataFrame(reb_rows)
    target_df = pd.DataFrame(target_rows)
    position_df = pd.DataFrame(position_rows)
    fallback_count = 0
    if not reb.empty and "exec_status" in reb.columns:
        fallback_count = int(reb["exec_status"].astype(str).str.contains("fallback", case=False, na=False).sum())

    summary = pd.DataFrame([{
        "tag": cfg.tag,
        "start_datetime": str(nav["datetime"].min()),
        "end_datetime": str(nav["datetime"].max()),
        "actual_return": strategy_return,
        "benchmark_return": benchmark_return,
        "alpha_return": alpha_return,
        "daily_excess_sharpe": daily_excess_sharpe,
        "avg_gross_prev_to_equity": float(nav["gross_prev_to_equity"].mean()),
        "turnover_weight": float(nav["turnover_weight"].sum()),
        "total_cost": float(nav["total_cost"].sum()),
        "opt_fallback_count": fallback_count,
        "n_minutes": int(len(nav)),
        "n_days": int(nav["date"].nunique()),
        "target_step_minutes": cfg.target_step_minutes,
        "execution_step_minutes": cfg.execution_step_minutes,
        "turnover_limit_per_exec": cfg.turnover_limit,
        "urgency_base": cfg.urgency_base,
        "urgency_amp": cfg.urgency_amp,
        "score_r0": cfg.score_r0,
        "score_z0": cfg.score_z0,
    }])

    nav_path = cfg.output_dir / f"{cfg.tag}_nav_curve.csv"
    daily_path = cfg.output_dir / f"{cfg.tag}_daily.csv"
    reb_path = cfg.output_dir / "summary_by_rebalance.csv"
    target_path = cfg.output_dir / "w_theory_targets.csv"
    positions_path = cfg.output_dir / "target_positions.csv"
    summary_path = cfg.output_dir / f"{cfg.tag}_summary.csv"

    nav.to_csv(nav_path, index=False)
    daily.to_csv(daily_path, index=False)
    reb.to_csv(reb_path, index=False)
    target_df.to_csv(target_path, index=False)
    position_df.to_csv(positions_path, index=False)
    summary.to_csv(summary_path, index=False)

    make_report_plot(nav, summary.iloc[0], cfg.output_dir / f"{cfg.tag}_nav_report.png", cfg.tag)

    print("\n===== v15 summary =====")
    print(summary.T.to_string(header=False))
    print("\n[saved]")
    print(nav_path)
    print(daily_path)
    print(reb_path)
    print(target_path)
    print(positions_path)
    print(summary_path)
    print(cfg.output_dir / f"{cfg.tag}_nav_report.png")


def make_report_plot(nav: pd.DataFrame, summary_row: pd.Series, out_png: Path, tag: str) -> None:
    x = np.arange(len(nav))
    fig = plt.figure(figsize=(16, 9), dpi=160)
    ax = fig.add_axes([0.07, 0.12, 0.66, 0.78])
    ax.plot(x, nav["strategy_cumret"] * 100.0, label="strategyret", linewidth=1.5)
    ax.plot(x, nav["benchmark_cumret"] * 100.0, label="benchmarkret", linewidth=1.5)
    ax.plot(x, nav["alpha_cumret"] * 100.0, label="alpharet", linewidth=1.7)
    ax.axhline(0.0, linestyle="--", linewidth=0.8, alpha=0.8)

    if "date" in nav.columns:
        first_idx = nav.groupby("date").head(1).index.to_numpy()
        labels = nav.loc[first_idx, "date"].astype(str).tolist()
        step = max(1, len(first_idx) // 10)
        ax.set_xticks(first_idx[::step])
        ax.set_xticklabels(labels[::step], rotation=45, ha="right")

    ax.set_title("v15 Two-clock Target-Execution Optimizer\n" + tag, fontsize=13, weight="bold")
    ax.set_xlabel("trading minute index, benchmark warm-start, no overnight PnL")
    ax.set_ylabel("cumulative return (%)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower left", fontsize=9)

    def pct(x):
        return f"{100.0 * float(x):.2f}%" if np.isfinite(float(x)) else "NA"

    text = "\n".join([
        "Summary",
        "",
        "Strategy Return", pct(summary_row["actual_return"]),
        "",
        "Benchmark Return", pct(summary_row["benchmark_return"]),
        "",
        "Alpha Return", pct(summary_row["alpha_return"]),
        "",
        "Daily Excess Sharpe", f"{float(summary_row['daily_excess_sharpe']):.2f}",
        "",
        "Turnover", f"{float(summary_row['turnover_weight']):.2f}x",
        "",
        "Fallback", str(int(summary_row["opt_fallback_count"])),
    ])
    fig.text(
        0.77, 0.62, text,
        fontsize=10.5,
        va="center",
        ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="0.6"),
    )
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h10-dir", type=Path, required=True)
    ap.add_argument("--h20-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--tag", type=str, default="pure_cs_v15_two_clock")

    ap.add_argument("--capital", type=float, default=200000000.0)
    ap.add_argument("--fee-bps", type=float, default=10.0)
    ap.add_argument("--lot-size", type=int, default=100)
    ap.add_argument("--min-trade-notional", type=float, default=5000.0)
    ap.add_argument("--allow-small-exit", type=int, default=1)

    ap.add_argument("--gross-target", type=float, default=0.95)
    ap.add_argument("--gross-min", type=float, default=0.90)
    ap.add_argument("--gross-max", type=float, default=0.98)
    ap.add_argument("--single-name-cap", type=float, default=0.008)
    ap.add_argument("--active-l1-limit", type=float, default=0.25)
    ap.add_argument("--active-l1-relax-list", type=str, default="0.35,0.50,0.75,1.00")

    ap.add_argument("--target-step-minutes", type=int, default=20)
    ap.add_argument("--execution-step-minutes", type=int, default=10)
    ap.add_argument("--turnover-limit", type=float, default=0.025)
    ap.add_argument("--daily-turnover-budget", type=float, default=1.00)
    ap.add_argument("--cash-buffer", type=float, default=0.001)

    ap.add_argument("--lambda-target-alpha", type=float, default=0.0005)
    ap.add_argument("--lambda-target-active", type=float, default=0.0001)
    ap.add_argument("--lambda-target-ridge", type=float, default=0.00001)

    ap.add_argument("--lambda-dist", type=float, default=5.0)
    ap.add_argument("--lambda-turnover", type=float, default=0.0018)
    ap.add_argument("--lambda-active-exec", type=float, default=0.0001)
    ap.add_argument("--lambda-ridge-exec", type=float, default=0.00001)
    ap.add_argument("--turncost-beta", type=float, default=0.50)

    ap.add_argument("--urgency-base", type=float, default=0.35)
    ap.add_argument("--urgency-amp", type=float, default=0.45)
    ap.add_argument("--urgency-min", type=float, default=0.05)
    ap.add_argument("--urgency-max", type=float, default=0.90)

    ap.add_argument("--score-r0", type=float, default=0.25)
    ap.add_argument("--score-z0", type=float, default=2.0)
    ap.add_argument("--score-clip-z", type=float, default=5.0)
    ap.add_argument("--max-names", type=int, default=0)
    ap.add_argument("--target-score-mode", type=str, default="rank_gate", choices=["rank_gate", "raw_h20", "csz_h20"])
    ap.add_argument("--solvers", type=str, default="CLARABEL,SCS,OSQP")
    return ap


def main() -> None:
    global GLOBAL_CFG
    ap = build_arg_parser()
    args = ap.parse_args()
    cfg = Config(
        h10_dir=args.h10_dir,
        h20_dir=args.h20_dir,
        output_dir=args.output_dir,
        tag=args.tag,
        capital=args.capital,
        fee_bps=args.fee_bps,
        lot_size=args.lot_size,
        min_trade_notional=args.min_trade_notional,
        allow_small_exit=args.allow_small_exit,
        gross_target=args.gross_target,
        gross_min=args.gross_min,
        gross_max=args.gross_max,
        single_name_cap=args.single_name_cap,
        active_l1_limit=args.active_l1_limit,
        active_l1_relax_list=parse_list(args.active_l1_relax_list, float),
        target_step_minutes=args.target_step_minutes,
        execution_step_minutes=args.execution_step_minutes,
        turnover_limit=args.turnover_limit,
        daily_turnover_budget=args.daily_turnover_budget,
        cash_buffer=args.cash_buffer,
        lambda_target_alpha=args.lambda_target_alpha,
        lambda_target_active=args.lambda_target_active,
        lambda_target_ridge=args.lambda_target_ridge,
        lambda_dist=args.lambda_dist,
        lambda_turnover=args.lambda_turnover,
        lambda_active_exec=args.lambda_active_exec,
        lambda_ridge_exec=args.lambda_ridge_exec,
        turncost_beta=args.turncost_beta,
        urgency_base=args.urgency_base,
        urgency_amp=args.urgency_amp,
        urgency_min=args.urgency_min,
        urgency_max=args.urgency_max,
        score_r0=args.score_r0,
        score_z0=args.score_z0,
        score_clip_z=args.score_clip_z,
        max_names=args.max_names,
        target_score_mode=args.target_score_mode,
        solvers=[s.strip() for s in args.solvers.split(",") if s.strip()],
    )
    GLOBAL_CFG = cfg
    print("===== v15 config =====")
    for k, v in cfg.__dict__.items():
        print(f"{k}: {v}")
    run_backtest(cfg)


if __name__ == "__main__":
    main()
