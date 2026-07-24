# -*- coding: utf-8 -*-
"""
v16 soft dual-alpha optimizer.

Core idea:
- h20 is a position alpha: rewards final active position.
- h10 is a short-horizon trading alpha: rewards current-period delta position.
- h10 is a soft objective term, not a hard gate. It can make trades more/less attractive,
  but it does not block trades and does not force target chasing.

Compared with v15 two-clock hard urgency:
- No w_theory hard target.
- No buy <= gap * urgency cap.
- One optimization at each rebalance/execution time:
      maximize h20_score · active_position + h10_score · delta_position
      minus turnover/risk penalties
  subject to A-share T+1, cash, gross, active, single cap, liquidity and limit constraints.

Outputs:
- target_positions.csv: actual shares snapshots for canonical evaluator.
- summary_by_rebalance.csv: cost/turnover/status by rebalance time for canonical evaluator.
- v16 internal nav/summary for diagnostics only; use canonical evaluator for final reporting.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import cvxpy as cp
except Exception as e:  # pragma: no cover
    raise RuntimeError("cvxpy is required for v16 optimizer") from e


# =============================================================================
# Utilities
# =============================================================================


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


def pick_col(cols: Iterable[str], candidates: List[str]) -> Optional[str]:
    s = set(cols)
    for c in candidates:
        if c in s:
            return c
    return None


def normalize_sid(s: pd.Series) -> pd.Series:
    # ZZY sid is internal id. Do not pad it as securityid.
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


# =============================================================================
# Cross-sectional score transforms
# =============================================================================


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


def _cs_z_one(s: pd.Series, clip_z: float) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    mu = x.mean(skipna=True)
    sd = x.std(skipna=True)
    if not np.isfinite(sd) or sd <= 1e-12:
        return pd.Series(0.0, index=s.index)
    z = (x - mu) / sd
    return z.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-clip_z, clip_z)


def _cs_rank_score_one(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    pct = x.rank(method="average", pct=True)
    r = 2.0 * pct - 1.0
    return r.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-1.0, 1.0)


def _rank_gate_value_one(s: pd.Series, *, r0: float, z0: float, clip_z: float) -> pd.Series:
    val = _cs_robust_z_one(s, clip_z=clip_z)
    r = _cs_rank_score_one(s)
    rank_gate = ((r.abs() - r0).clip(lower=0.0) / max(1e-12, (1.0 - r0))).clip(0.0, 1.0)
    value_conv = np.tanh(val.abs() / max(1e-12, z0))
    score = np.sign(r) * rank_gate * value_conv
    return pd.Series(score, index=s.index).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def add_score_by_datetime(
    df: pd.DataFrame,
    pred_col: str,
    out_col: str,
    *,
    mode: str,
    r0: float,
    z0: float,
    clip_z: float,
) -> pd.DataFrame:
    pieces = []
    for _, g in df.groupby("datetime", sort=False):
        idx = g.index
        if mode == "raw":
            score = pd.to_numeric(g[pred_col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
            score = score.clip(-clip_z, clip_z)
        elif mode == "csz":
            score = _cs_z_one(g[pred_col], clip_z=clip_z)
        elif mode == "robust_z":
            score = _cs_robust_z_one(g[pred_col], clip_z=clip_z)
        elif mode == "rank":
            score = _cs_rank_score_one(g[pred_col])
        elif mode == "rank_gate":
            score = _rank_gate_value_one(g[pred_col], r0=r0, z0=z0, clip_z=clip_z)
        else:
            raise ValueError(f"unknown score mode: {mode}")
        pieces.append(pd.Series(score, index=idx))
    df[out_col] = pd.concat(pieces).sort_index().astype(float)
    return df


# =============================================================================
# Config and data loading
# =============================================================================


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
    max_opt_names: int
    active_l1_limit: float
    active_l1_relax_list: List[float]
    rebalance_step_minutes: int
    turnover_limit: float
    daily_turnover_budget: float
    cash_buffer: float
    alpha20_scale: float
    alpha10_delta_scale: float
    lambda_turnover: float
    lambda_active: float
    lambda_ridge: float
    lambda_trade_ridge: float
    score_mode_h20: str
    score_mode_h10: str
    score_r0: float
    score_z0: float
    score_clip_z: float
    liquidity_participation: float
    turnover_participation: float
    solvers: List[str]


GLOBAL_CFG: Config


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

    if h20_sig != "pred_ret_h20":
        base = base.rename(columns={h20_sig: "pred_ret_h20"})
    h10_small = h10[["date", "ts", "sid", h10_sig]].rename(columns={h10_sig: "pred_ret_h10"})
    df = base.merge(h10_small, on=["date", "ts", "sid"], how="left")

    df["pred_ret_h20"] = pd.to_numeric(df["pred_ret_h20"], errors="coerce").fillna(0.0)
    df["pred_ret_h10"] = pd.to_numeric(df["pred_ret_h10"], errors="coerce").fillna(0.0)

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

    for c in ["bid_volume1", "ask_volume1", "turnover_amount"]:
        if c not in df.columns:
            df[c] = np.nan
        else:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "benchmark_weight" not in df.columns:
        df["benchmark_weight"] = 0.0
    df["benchmark_weight"] = pd.to_numeric(df["benchmark_weight"], errors="coerce").fillna(0.0).clip(lower=0.0)

    if "securityid" not in df.columns:
        df["securityid"] = df["sid"].astype(str)
    else:
        df["securityid"] = df["securityid"].astype(str).str.zfill(6)

    if "limit_up" not in df.columns:
        df["limit_up"] = np.nan
    if "limit_down" not in df.columns:
        df["limit_down"] = np.nan

    df = df.sort_values(["datetime", "securityid"]).reset_index(drop=True)

    df = add_score_by_datetime(
        df,
        "pred_ret_h20",
        "s20_position",
        mode=GLOBAL_CFG.score_mode_h20,
        r0=GLOBAL_CFG.score_r0,
        z0=GLOBAL_CFG.score_z0,
        clip_z=GLOBAL_CFG.score_clip_z,
    )
    df = add_score_by_datetime(
        df,
        "pred_ret_h10",
        "s10_delta",
        mode=GLOBAL_CFG.score_mode_h10,
        r0=GLOBAL_CFG.score_r0,
        z0=GLOBAL_CFG.score_z0,
        clip_z=GLOBAL_CFG.score_clip_z,
    )

    return df


# =============================================================================
# Portfolio state
# =============================================================================


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
    used_px: Dict[str, float] = {}
    for sid, sh in state.actual_shares.items():
        px = safe_float(price_map.get(sid, np.nan), np.nan)
        if np.isfinite(px) and px > 0 and sh != 0:
            mv += sh * px
            used_px[sid] = px
    equity = float(state.cash + mv)
    return equity, used_px


def get_weight_arrays(state: PortfolioState, g: pd.DataFrame, equity: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def is_limit_up_row(row) -> bool:
    lu = safe_float(getattr(row, "limit_up", np.nan), np.nan)
    ask = safe_float(getattr(row, "ask1", np.nan), np.nan)
    return bool(np.isfinite(lu) and lu > 0 and np.isfinite(ask) and ask >= lu * 0.999)


def is_limit_down_row(row) -> bool:
    ld = safe_float(getattr(row, "limit_down", np.nan), np.nan)
    bid = safe_float(getattr(row, "bid1", np.nan), np.nan)
    return bool(np.isfinite(ld) and ld > 0 and np.isfinite(bid) and bid <= ld * 1.001)


# =============================================================================
# Optimization and execution
# =============================================================================


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


def solve_dual_alpha_optimizer(
    g: pd.DataFrame,
    state: PortfolioState,
    cfg: Config,
    active_limit: float,
    turnover_limit_now: float,
) -> Tuple[np.ndarray, np.ndarray, str, Dict[str, float]]:
    """Solve one rebalance step.

    Fast top-K version:
    - If cfg.max_opt_names <= 0, solve full universe.
    - Otherwise solve only a selected subset of names.
    - Non-selected positions are held fixed and enter gross/active constraints as constants.
    - Returned buy_w / sell_w are full-length arrays aligned to g.
    """
    n_full = len(g)
    equity, _ = mark_equity(state, g)
    if equity <= 0:
        return np.zeros(n_full), np.zeros(n_full), "bad_equity", {}

    w_cur_all, w_sellable_all, w_locked_all = get_weight_arrays(state, g, equity)

    b_all = pd.to_numeric(g["benchmark_weight"], errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(dtype=float)
    if b_all.sum() > 0:
        b_all = b_all / b_all.sum() * cfg.gross_target
    else:
        b_all = np.ones(n_full, dtype=float) / max(n_full, 1) * cfg.gross_target

    s20_all = pd.to_numeric(g["s20_position"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    s10_all = pd.to_numeric(g["s10_delta"], errors="coerce").fillna(0.0).to_numpy(dtype=float)

    # ------------------------------------------------------------------
    # Select optimization universe.
    # ------------------------------------------------------------------
    max_k = int(getattr(cfg, "max_opt_names", 0) or 0)
    if max_k <= 0 or n_full <= max_k:
        opt_idx = np.arange(n_full)
    else:
        candidates = set()

        def add_top(arr: np.ndarray, k: int):
            if k <= 0:
                return
            k = min(k, n_full)
            idx = np.argpartition(-np.nan_to_num(arr, nan=0.0), k - 1)[:k]
            candidates.update(map(int, idx))

        # Use overlapping buckets. If union is too large, prune by a combined priority.
        add_top(np.abs(s20_all), int(max_k * 0.35))
        add_top(np.abs(s10_all), int(max_k * 0.25))
        add_top(np.abs(w_cur_all - b_all), int(max_k * 0.20))
        add_top(w_cur_all, int(max_k * 0.15))
        add_top(b_all, int(max_k * 0.10))

        cand = np.array(sorted(candidates), dtype=int)
        if len(cand) == 0:
            cand = np.arange(min(max_k, n_full))

        if len(cand) > max_k:
            priority = (
                2.0 * np.abs(s20_all[cand])
                + 1.0 * np.abs(s10_all[cand])
                + 8.0 * np.abs(w_cur_all[cand] - b_all[cand])
                + 5.0 * w_cur_all[cand]
                + 2.0 * b_all[cand]
                + 1.0 * w_locked_all[cand]
            )
            keep = np.argpartition(-priority, max_k - 1)[:max_k]
            cand = cand[keep]

        opt_idx = np.array(sorted(set(map(int, cand))), dtype=int)

    mask = np.zeros(n_full, dtype=bool)
    mask[opt_idx] = True
    fixed_mask = ~mask

    g_sub = g.iloc[opt_idx].copy().reset_index(drop=True)
    n = len(g_sub)

    w_cur = w_cur_all[opt_idx]
    w_sellable = w_sellable_all[opt_idx]
    w_locked = w_locked_all[opt_idx]
    b = b_all[opt_idx]
    s20 = s20_all[opt_idx]
    s10 = s10_all[opt_idx]

    fixed_gross = float(w_cur_all[fixed_mask].sum())
    fixed_active_l1 = float(np.abs(w_cur_all[fixed_mask] - b_all[fixed_mask]).sum())
    fixed_active_sum = float((w_cur_all[fixed_mask] - b_all[fixed_mask]).sum())

    # If fixed positions alone violate active too much, this solve may still fallback,
    # but relax grid in caller can enlarge active_limit.

    buy_cap = np.full(n, np.inf, dtype=float)
    sell_cap = w_sellable.copy()

    mid = pd.to_numeric(g_sub["mid_price"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    ask = pd.to_numeric(g_sub["ask1"], errors="coerce").fillna(pd.Series(mid)).to_numpy(dtype=float)
    bid = pd.to_numeric(g_sub["bid1"], errors="coerce").fillna(pd.Series(mid)).to_numpy(dtype=float)
    ask_vol = pd.to_numeric(g_sub.get("ask_volume1", pd.Series(np.nan, index=g_sub.index)), errors="coerce").to_numpy(dtype=float)
    bid_vol = pd.to_numeric(g_sub.get("bid_volume1", pd.Series(np.nan, index=g_sub.index)), errors="coerce").to_numpy(dtype=float)
    turnover_amt = pd.to_numeric(g_sub.get("turnover_amount", pd.Series(np.nan, index=g_sub.index)), errors="coerce").to_numpy(dtype=float)

    for i, row in enumerate(g_sub.itertuples(index=False)):
        if is_limit_up_row(row):
            buy_cap[i] = 0.0
        if is_limit_down_row(row):
            sell_cap[i] = 0.0
        if cfg.liquidity_participation > 0 and np.isfinite(ask_vol[i]) and ask_vol[i] > 0 and np.isfinite(ask[i]) and ask[i] > 0:
            buy_cap[i] = min(buy_cap[i], cfg.liquidity_participation * ask_vol[i] * ask[i] / max(equity, 1e-12))
        if cfg.liquidity_participation > 0 and np.isfinite(bid_vol[i]) and bid_vol[i] > 0 and np.isfinite(bid[i]) and bid[i] > 0:
            sell_cap[i] = min(sell_cap[i], cfg.liquidity_participation * bid_vol[i] * bid[i] / max(equity, 1e-12))
        if cfg.turnover_participation > 0 and np.isfinite(turnover_amt[i]) and turnover_amt[i] > 0:
            cap = cfg.turnover_participation * turnover_amt[i] / max(equity, 1e-12)
            buy_cap[i] = min(buy_cap[i], cap)
            sell_cap[i] = min(sell_cap[i], cap)

    buy_cap = np.nan_to_num(buy_cap, nan=np.inf, posinf=np.inf, neginf=0.0)
    # Avoid infinite constants in CVXPY constraints.
    buy_cap = np.minimum(buy_cap, min(cfg.single_name_cap, turnover_limit_now, max(cfg.gross_max, 1e-12)))
    sell_cap = np.nan_to_num(sell_cap, nan=0.0, posinf=0.0, neginf=0.0).clip(min=0.0)

    buy = cp.Variable(n, nonneg=True)
    sell = cp.Variable(n, nonneg=True)
    w_next = w_cur + buy - sell
    delta = buy - sell
    turnover = buy + sell

    fee_rate = cfg.fee_bps / 10000.0
    cash_weight = state.cash / max(equity, 1e-12)

    constraints = [
        buy <= buy_cap,
        sell <= sell_cap,
        sell <= w_sellable,
        w_next >= w_locked,
        w_next >= 0,
        w_next <= cfg.single_name_cap,
        fixed_gross + cp.sum(w_next) >= cfg.gross_min,
        fixed_gross + cp.sum(w_next) <= cfg.gross_max,

        # v24 benchmark-relative 95% stock core:
        # b is normalized to cfg.gross_target * benchmark_weight.
        # active_w = w_next - b.
        # Enforce dollar-neutral active exposure around benchmark base.
        # soft BR95: no hard active-sum equality; gross band controls total exposure.
        fixed_active_l1 + cp.norm1(w_next - b) <= active_limit,

        cp.sum(turnover) <= turnover_limit_now,
        (1.0 + fee_rate) * cp.sum(buy) <= cash_weight + (1.0 - fee_rate) * cp.sum(sell) - cfg.cash_buffer,
    ]

    objective = cp.Maximize(
        cfg.alpha20_scale * (s20 @ (w_next - b))
        + cfg.alpha10_delta_scale * (s10 @ delta)
        - cfg.lambda_turnover * cp.sum(turnover)
        - cfg.lambda_active * cp.sum_squares(w_next - b)
        - cfg.lambda_ridge * cp.sum_squares(w_next)
        - cfg.lambda_trade_ridge * cp.sum_squares(delta)
    )

    prob = cp.Problem(objective, constraints)
    dt0 = g["datetime"].iloc[0]
    print(
        f"[solve] dt={dt0} n_full={n_full} n_opt={n} fixed_gross={fixed_gross:.4f} fixed_active_l1={fixed_active_l1:.4f} fixed_active_sum={fixed_active_sum:.4e} active={active_limit:.3f} turn={turnover_limit_now:.4f}",
        flush=True,
    )
    ok, status = solve_cvx_problem(prob, cfg.solvers)
    print(f"[solve done] dt={dt0} status={status}", flush=True)

    diag = {
        "equity": equity,
        "n_full": int(n_full),
        "n_opt": int(n),
        "fixed_gross": fixed_gross,
        "fixed_active_l1": fixed_active_l1,
        "fixed_active_sum": fixed_active_sum,
        "active_limit_used": active_limit,
        "turnover_limit_used": turnover_limit_now,
        "gross_before": float(w_cur_all.sum()),
        "alpha20_mean": float(np.mean(s20)),
        "alpha20_std": float(np.std(s20)),
        "alpha10_mean": float(np.mean(s10)),
        "alpha10_std": float(np.std(s10)),
    }

    buy_full = np.zeros(n_full, dtype=float)
    sell_full = np.zeros(n_full, dtype=float)

    if not ok or buy.value is None or sell.value is None:
        return buy_full, sell_full, f"fallback_{status}", diag

    buy_w = np.nan_to_num(np.asarray(buy.value, dtype=float), nan=0.0, posinf=0.0, neginf=0.0).clip(min=0.0)
    sell_w = np.nan_to_num(np.asarray(sell.value, dtype=float), nan=0.0, posinf=0.0, neginf=0.0).clip(min=0.0)

    buy_full[opt_idx] = buy_w
    sell_full[opt_idx] = sell_w

    w_next_sub = w_cur + buy_w - sell_w
    w_next_all_sum = fixed_gross + float(w_next_sub.sum())

    diag.update({
        "gross_after_cont": w_next_all_sum,
        "buy_weight_cont": float(buy_w.sum()),
        "sell_weight_cont": float(sell_w.sum()),
        "turnover_weight_cont": float((buy_w + sell_w).sum()),
        "delta_alpha10_cont": float(s10 @ (buy_w - sell_w)),
        "pos_alpha20_cont": float(s20 @ (w_next_sub - b)),
    })
    return buy_full, sell_full, status, diag

def execute_trades(g: pd.DataFrame, state: PortfolioState, buy_w: np.ndarray, sell_w: np.ndarray, cfg: Config) -> Dict[str, float]:
    equity, _ = mark_equity(state, g)
    fee_rate = cfg.fee_bps / 10000.0

    sec = g["securityid"].astype(str).to_numpy()
    bid = pd.to_numeric(g["bid1"], errors="coerce").fillna(pd.to_numeric(g["mid_price"], errors="coerce")).to_numpy(dtype=float)
    ask = pd.to_numeric(g["ask1"], errors="coerce").fillna(pd.to_numeric(g["mid_price"], errors="coerce")).to_numpy(dtype=float)

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
        # T+1: bought shares are not added to sellable today.
        state.cash -= notional + fee
        buy_notional += notional
        buy_fee += fee
        buy_names += 1

    return {
        "buy_notional": buy_notional,
        "sell_notional": sell_notional,
        "turnover_notional": buy_notional + sell_notional,
        "fee": buy_fee + sell_fee,
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


def should_rebalance(dt: pd.Timestamp, first_dt: pd.Timestamp, step_minutes: int) -> bool:
    diff_min = int(round((dt - first_dt).total_seconds() / 60.0))
    return diff_min >= 0 and diff_min % step_minutes == 0


# =============================================================================
# Main loop
# =============================================================================


def run_backtest(cfg: Config) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    h20_files = sorted([p for p in cfg.h20_dir.glob("*.parquet") if not p.name.startswith("_")])
    if not h20_files:
        raise FileNotFoundError(cfg.h20_dir)

    state: Optional[PortfolioState] = None
    nav_rows: List[Dict] = []
    reb_rows: List[Dict] = []
    position_rows: List[Dict] = []

    prev_equity: Optional[float] = None
    prev_bench_prices: Dict[str, float] = {}
    daily_turnover: Dict[int, float] = {}
    cumulative_cost = 0.0
    cumulative_turnover = 0.0

    for file_idx, h20_file in enumerate(h20_files, 1):
        print(f"\n===== load day {file_idx}/{len(h20_files)}: {h20_file.name} =====", flush=True)
        day_df = read_day_inputs(h20_file, cfg.h10_dir)
        d0 = int(day_df["date"].iloc[0])
        minutes = list(day_df["datetime"].drop_duplicates().sort_values())
        if not minutes:
            continue
        first_dt = minutes[0]

        if state is not None:
            # T+1 release at day start. No overnight return in curve.
            state.sellable_shares = state.actual_shares.copy()
            state.current_date = d0
            prev_equity = None
            prev_bench_prices = {}
        daily_turnover.setdefault(d0, 0.0)

        for dt in minutes:
            g = day_df[day_df["datetime"] == dt].copy().reset_index(drop=True)
            if g.empty:
                continue

            if state is None or not state.initialized:
                state = initialize_benchmark_state(g, cfg)
                state.current_date = d0
                prev_equity = None
                prev_bench_prices = {}
                print(f"[init] {dt} cash={state.cash:.2f} names={len(state.actual_shares)}", flush=True)

            assert state is not None

            is_reb = should_rebalance(dt, first_dt, cfg.rebalance_step_minutes)
            trade_info = {
                "buy_notional": 0.0,
                "sell_notional": 0.0,
                "turnover_notional": 0.0,
                "fee": 0.0,
                "buy_names": 0,
                "sell_names": 0,
            }
            status = "not_rebalance"
            diag: Dict[str, float] = {}

            if is_reb:
                equity_before, _ = mark_equity(state, g)
                budget_left = max(0.0, cfg.daily_turnover_budget - daily_turnover.get(d0, 0.0))
                turn_now = min(cfg.turnover_limit, budget_left)
                if turn_now <= 1e-10:
                    status = "skip_daily_turnover_budget"
                else:
                    active_grid = [cfg.active_l1_limit] + [x for x in cfg.active_l1_relax_list if x > cfg.active_l1_limit]
                    buy_w = sell_w = np.zeros(len(g), dtype=float)
                    for active_limit in active_grid:
                        buy_w, sell_w, status, diag = solve_dual_alpha_optimizer(
                            g, state, cfg, active_limit=active_limit, turnover_limit_now=turn_now
                        )
                        if not str(status).startswith("fallback"):
                            break
                    trade_info = execute_trades(g, state, buy_w, sell_w, cfg)
                    turn_w = trade_info["turnover_notional"] / max(equity_before, 1e-12)
                    daily_turnover[d0] = daily_turnover.get(d0, 0.0) + turn_w
                    cumulative_turnover += turn_w
                    cumulative_cost += trade_info["fee"]

                equity_after, _ = mark_equity(state, g)
                reb_rows.append({
                    "date": d0,
                    "datetime": dt,
                    "status": status,
                    "total_cost": float(trade_info["fee"]),
                    "turnover_weight": float(trade_info["turnover_notional"] / max(equity_before, 1e-12)),
                    "equity_before": equity_before,
                    "equity_after": equity_after,
                    "cash_after": state.cash,
                    "daily_turnover_used": daily_turnover.get(d0, 0.0),
                    **trade_info,
                    **diag,
                })

            equity, _ = mark_equity(state, g)
            if prev_equity is None:
                actual_ret = 0.0
            else:
                actual_ret = equity / max(prev_equity, 1e-12) - 1.0

            if not prev_bench_prices:
                bench_ret = 0.0
            else:
                bench_ret = benchmark_step_return(g, prev_bench_prices)

            # Position snapshot for canonical evaluator.
            for row in g.itertuples(index=False):
                sid = str(getattr(row, "securityid"))
                position_rows.append({
                    "date": d0,
                    "datetime": dt,
                    "securityid": sid,
                    "actual_shares_after": int(state.actual_shares.get(sid, 0)),
                })

            mid_map = dict(zip(g["securityid"].astype(str), pd.to_numeric(g["mid_price"], errors="coerce")))
            prev_bench_prices = {sid: float(px) for sid, px in mid_map.items() if np.isfinite(px) and px > 0}
            prev_equity = equity

            gross_mv = 0.0
            for sid, sh in state.actual_shares.items():
                px = prev_bench_prices.get(sid, np.nan)
                if np.isfinite(px) and px > 0:
                    gross_mv += sh * px
            gross = gross_mv / max(equity, 1e-12)

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
                "cum_cost": cumulative_cost,
                "cum_turnover": cumulative_turnover,
            })

        print(
            f"[day done] {d0} cash={state.cash:.2f} names={len(state.actual_shares)} "
            f"daily_turn={daily_turnover.get(d0, 0.0):.4f}",
            flush=True,
        )

    nav = pd.DataFrame(nav_rows).sort_values("datetime").reset_index(drop=True)
    reb = pd.DataFrame(reb_rows)
    pos = pd.DataFrame(position_rows)

    if nav.empty:
        raise RuntimeError("empty nav")

    nav["strategy_nav"] = (1.0 + nav["actual_ret"].fillna(0.0)).cumprod()
    nav["benchmark_nav"] = (1.0 + nav["benchmark_ret"].fillna(0.0)).cumprod()
    nav["alpha_ret"] = nav["actual_ret"] - nav["benchmark_ret"]
    nav["alpha_nav"] = (1.0 + nav["alpha_ret"].fillna(0.0)).cumprod()

    daily = nav.groupby("date", as_index=False).agg(
        actual_day=("actual_ret", compound_return),
        benchmark_day=("benchmark_ret", compound_return),
        alpha_day=("alpha_ret", compound_return),
        turnover=("turnover_weight", "sum"),
        cost=("total_cost", "sum"),
        avg_gross=("gross_prev_to_equity", "mean"),
    )

    fallback_count = 0
    if not reb.empty and "status" in reb.columns:
        fallback_count = int(reb["status"].astype(str).str.contains("fallback", case=False, na=False).sum())

    summary = pd.DataFrame([{
        "tag": cfg.tag,
        "start_datetime": str(nav["datetime"].min()),
        "end_datetime": str(nav["datetime"].max()),
        "actual_return_internal": compound_return(nav["actual_ret"]),
        "benchmark_return_internal": compound_return(nav["benchmark_ret"]),
        "alpha_return_internal": compound_return(nav["actual_ret"]) - compound_return(nav["benchmark_ret"]),
        "daily_excess_sharpe_internal": annualized_sharpe(daily["alpha_day"]),
        "avg_gross_prev_to_equity": float(nav["gross_prev_to_equity"].mean()),
        "turnover_weight": float(nav["turnover_weight"].sum()),
        "total_cost": float(nav["total_cost"].sum()),
        "opt_fallback_count": fallback_count,
        "n_minutes": int(len(nav)),
        "n_days": int(nav["date"].nunique()),
        "rebalance_step_minutes": cfg.rebalance_step_minutes,
        "alpha20_scale": cfg.alpha20_scale,
        "alpha10_delta_scale": cfg.alpha10_delta_scale,
        "lambda_turnover": cfg.lambda_turnover,
        "turnover_limit": cfg.turnover_limit,
        "score_mode_h20": cfg.score_mode_h20,
        "score_mode_h10": cfg.score_mode_h10,
    }])

    nav_path = cfg.output_dir / f"{cfg.tag}_nav_curve_internal.csv"
    daily_path = cfg.output_dir / f"{cfg.tag}_daily_internal.csv"
    reb_path = cfg.output_dir / "summary_by_rebalance.csv"
    pos_path = cfg.output_dir / "target_positions.csv"
    summary_path = cfg.output_dir / f"{cfg.tag}_summary_internal.csv"

    nav.to_csv(nav_path, index=False)
    daily.to_csv(daily_path, index=False)
    reb.to_csv(reb_path, index=False)
    pos.to_csv(pos_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("\n===== v16 internal summary =====")
    print(summary.T.to_string(header=False))
    print("\n[saved]")
    print(nav_path)
    print(daily_path)
    print(reb_path)
    print(pos_path)
    print(summary_path)


# =============================================================================
# CLI
# =============================================================================


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h10-dir", type=Path, required=True)
    ap.add_argument("--h20-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--tag", type=str, default="pure_cs_v16_dual_alpha")

    ap.add_argument("--capital", type=float, default=200000000.0)
    ap.add_argument("--fee-bps", type=float, default=10.0)
    ap.add_argument("--lot-size", type=int, default=100)
    ap.add_argument("--min-trade-notional", type=float, default=5000.0)
    ap.add_argument("--allow-small-exit", type=int, default=1)

    ap.add_argument("--gross-target", type=float, default=0.95)
    ap.add_argument("--gross-min", type=float, default=0.90)
    ap.add_argument("--gross-max", type=float, default=0.98)
    ap.add_argument("--single-name-cap", type=float, default=0.008)
    ap.add_argument("--max-opt-names", type=int, default=0, help="Limit each CVXPY solve to top-K tradable names; 0 means full universe.")
    ap.add_argument("--active-l1-limit", type=float, default=0.25)
    ap.add_argument("--active-l1-relax-list", type=str, default="0.35,0.50,0.75,1.00")

    ap.add_argument("--rebalance-step-minutes", type=int, default=20)
    ap.add_argument("--turnover-limit", type=float, default=0.04)
    ap.add_argument("--daily-turnover-budget", type=float, default=1.00)
    ap.add_argument("--cash-buffer", type=float, default=0.001)

    ap.add_argument("--alpha20-scale", type=float, default=0.0005)
    ap.add_argument("--alpha10-delta-scale", type=float, default=0.0002)
    ap.add_argument("--lambda-turnover", type=float, default=0.0018)
    ap.add_argument("--lambda-active", type=float, default=0.0001)
    ap.add_argument("--lambda-ridge", type=float, default=0.00001)
    ap.add_argument("--lambda-trade-ridge", type=float, default=0.00001)

    ap.add_argument("--score-mode-h20", type=str, default="raw", choices=["raw", "csz", "robust_z", "rank", "rank_gate"])
    ap.add_argument("--score-mode-h10", type=str, default="raw", choices=["raw", "csz", "robust_z", "rank", "rank_gate"])
    ap.add_argument("--score-r0", type=float, default=0.25)
    ap.add_argument("--score-z0", type=float, default=2.0)
    ap.add_argument("--score-clip-z", type=float, default=5.0)

    ap.add_argument("--liquidity-participation", type=float, default=0.0)
    ap.add_argument("--turnover-participation", type=float, default=0.0)
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
        max_opt_names=args.max_opt_names,
        active_l1_limit=args.active_l1_limit,
        active_l1_relax_list=parse_list(args.active_l1_relax_list, float),
        rebalance_step_minutes=args.rebalance_step_minutes,
        turnover_limit=args.turnover_limit,
        daily_turnover_budget=args.daily_turnover_budget,
        cash_buffer=args.cash_buffer,
        alpha20_scale=args.alpha20_scale,
        alpha10_delta_scale=args.alpha10_delta_scale,
        lambda_turnover=args.lambda_turnover,
        lambda_active=args.lambda_active,
        lambda_ridge=args.lambda_ridge,
        lambda_trade_ridge=args.lambda_trade_ridge,
        score_mode_h20=args.score_mode_h20,
        score_mode_h10=args.score_mode_h10,
        score_r0=args.score_r0,
        score_z0=args.score_z0,
        score_clip_z=args.score_clip_z,
        liquidity_participation=args.liquidity_participation,
        turnover_participation=args.turnover_participation,
        solvers=[s.strip() for s in str(args.solvers).split(",") if s.strip()],
    )
    GLOBAL_CFG = cfg

    print("===== v24 benchmark-relative 95% stock config =====")
    for k, v in cfg.__dict__.items():
        print(f"{k}: {v}")

    run_backtest(cfg)


if __name__ == "__main__":
    main()
