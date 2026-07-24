# -*- coding: utf-8 -*-
"""
Pure Cross-Sectional CVXPY Index Enhancement Optimizer.

Core idea:
    h10/h20 alpha -> pure cross-sectional benchmark-relative target weights.
    No ENTRY / ADD / HOLD / EXIT state machine.

It keeps the useful non-timing restrictions from v8:
    - opponent price cost: buy at ask, sell at bid
    - 10bp single-side fee
    - turnover penalty and turnover cap
    - limit-up cannot buy, limit-down cannot sell
    - bid/ask volume participation cap
    - optional market turnover cap
    - min trade notional and lot-size post-processing
"""

import argparse
import glob
import json
import os
import shutil
import warnings
from typing import Any, Dict, List, Optional, Tuple

import cvxpy as cp
import warnings
warnings.filterwarnings(
    "ignore",
    message="Solution may be inaccurate.*",
    category=UserWarning,
)
import numpy as np
import pandas as pd
import yaml


# =============================================================================
# Utils
# =============================================================================


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def col_exists(df: pd.DataFrame, col: Optional[str]) -> bool:
    return bool(col) and col in df.columns


def arr_num(s: pd.Series, default: float = np.nan) -> np.ndarray:
    x = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
    if not np.isnan(default):
        x[~np.isfinite(x)] = default
    return x


def normalize_symbol_value(x: Any, zfill: Optional[int]) -> str:
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    if zfill and s.isdigit():
        s = s.zfill(int(zfill))
    return s


def finite_zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x, dtype=float)
    mask = np.isfinite(x)
    if mask.sum() <= 2:
        return out
    mu = float(np.nanmean(x[mask]))
    sd = float(np.nanstd(x[mask]))
    if not np.isfinite(sd) or sd < 1e-12:
        return out
    out[mask] = (x[mask] - mu) / sd
    out[~np.isfinite(out)] = 0.0
    return out


def winsorize(x: np.ndarray, low_q: float, high_q: float) -> np.ndarray:
    x = np.asarray(x, dtype=float).copy()
    mask = np.isfinite(x)
    if mask.sum() <= 5:
        x[~mask] = 0.0
        return x
    lo = float(np.nanquantile(x[mask], low_q))
    hi = float(np.nanquantile(x[mask], high_q))
    x = np.clip(x, lo, hi)
    med = float(np.nanmedian(x[mask]))
    x[~np.isfinite(x)] = med
    return x


def cap_and_rescale_nonnegative(w: np.ndarray, target_sum: float, cap: float) -> np.ndarray:
    w = np.asarray(w, dtype=float).copy()
    n = len(w)
    if n == 0:
        return w
    w[~np.isfinite(w)] = 0.0
    w = np.maximum(w, 0.0)
    cap = float(cap)
    if cap * n < target_sum:
        return np.full(n, cap, dtype=float)
    if w.sum() <= 1e-12:
        w = np.full(n, target_sum / n, dtype=float)
    else:
        w = w / w.sum() * target_sum
    w = np.minimum(w, cap)
    for _ in range(50):
        residual = target_sum - float(w.sum())
        if abs(residual) < 1e-10:
            break
        free = w < cap - 1e-12
        if free.sum() == 0:
            break
        w[free] += residual / free.sum()
        w = np.clip(w, 0.0, cap)
    return w


# =============================================================================
# Data loading
# =============================================================================


def read_table(path: str) -> pd.DataFrame:
    low = path.lower()
    if low.endswith(".parquet"):
        return pd.read_parquet(path)
    if low.endswith(".csv") or low.endswith(".csv.gz"):
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path}")


def load_input(cfg: Dict[str, Any]) -> pd.DataFrame:
    input_cfg = cfg.get("input", {})
    files: List[str] = []

    paths = input_cfg.get("paths") or []
    if isinstance(paths, str):
        files.append(paths)
    else:
        files.extend(list(paths))

    path_glob = input_cfg.get("path_glob")
    if path_glob:
        files.extend(glob.glob(path_glob))

    files = sorted(list(dict.fromkeys(files)))
    if not files:
        raise FileNotFoundError(f"No input files found. path_glob={path_glob}, paths={paths}")

    dfs = []
    for i, p in enumerate(files, 1):
        print(f"[load] {i}/{len(files)} {p}")
        dfs.append(read_table(p))
    df = pd.concat(dfs, ignore_index=True)
    print(f"[load] rows={len(df):,}")
    return df


def prepare_df(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    cols = cfg["columns"]
    input_cfg = cfg.get("input", {})

    date_col = cols["date"]
    dt_col = cols["datetime"]
    sym_col = cols["symbol"]
    signal_col = cols["signal"]

    for c in [date_col, dt_col, sym_col, signal_col]:
        if c not in df.columns:
            raise KeyError(f"Required column not found: {c}")

    df = df.copy()
    df[dt_col] = pd.to_datetime(df[dt_col])

    zfill = input_cfg.get("symbol_zfill")
    df[sym_col] = df[sym_col].map(lambda x: normalize_symbol_value(x, zfill))

    if input_cfg.get("resample_to_minute", False):
        print("[prep] resample_to_minute=True, keep last record per symbol per minute")
        df["__minute"] = df[dt_col].dt.floor("min")
        df = df.sort_values([date_col, sym_col, dt_col])
        df = df.drop_duplicates([date_col, sym_col, "__minute"], keep="last")
        df[dt_col] = df["__minute"]
        df = df.drop(columns=["__minute"])

    df = df.sort_values([date_col, dt_col, sym_col]).reset_index(drop=True)
    print(f"[prep] rows={len(df):,}, dates={df[date_col].nunique()}, timestamps={df[dt_col].nunique()}")
    return df


def build_rebalance_timestamps(df: pd.DataFrame, cfg: Dict[str, Any]) -> set:
    cols = cfg["columns"]
    interval = int(cfg.get("strategy", {}).get("rebalance_interval_min", 10))
    by = cfg.get("strategy", {}).get("rebalance_by", "index")
    date_col = cols["date"]
    dt_col = cols["datetime"]

    uniq = df[[date_col, dt_col]].drop_duplicates().sort_values([date_col, dt_col])
    selected = []
    if by == "clock":
        for ts in uniq[dt_col].tolist():
            mod = ts.hour * 60 + ts.minute
            if mod % interval == 0:
                selected.append(ts)
    else:
        for _, g in uniq.groupby(date_col, sort=True):
            selected.extend(g[dt_col].tolist()[::interval])

    out = set(pd.to_datetime(selected))
    print(f"[rebalance] interval={interval}, by={by}, selected={len(out):,}")
    return out


# =============================================================================
# Cross-section construction
# =============================================================================


def get_prices(g: pd.DataFrame, cfg: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    cols = cfg["columns"]
    price_col = cols.get("price")
    bid_col = cols.get("bid")
    ask_col = cols.get("ask")

    if col_exists(g, price_col):
        mid = arr_num(g[price_col])
    elif col_exists(g, bid_col) and col_exists(g, ask_col):
        bid_tmp = arr_num(g[bid_col])
        ask_tmp = arr_num(g[ask_col])
        mid = (bid_tmp + ask_tmp) / 2.0
    else:
        raise KeyError("Need price column or both bid and ask columns.")

    bid = arr_num(g[bid_col]) if col_exists(g, bid_col) else mid.copy()
    ask = arr_num(g[ask_col]) if col_exists(g, ask_col) else mid.copy()

    bad_mid = ~np.isfinite(mid) | (mid <= 0)
    mid[bad_mid] = np.nan
    bid[(~np.isfinite(bid)) | (bid <= 0)] = mid[(~np.isfinite(bid)) | (bid <= 0)]
    ask[(~np.isfinite(ask)) | (ask <= 0)] = mid[(~np.isfinite(ask)) | (ask <= 0)]
    return mid, bid, ask


def get_benchmark(g: pd.DataFrame, cfg: Dict[str, Any], gross_target: float) -> np.ndarray:
    cols = cfg["columns"]
    b_col = cols.get("benchmark_weight")
    n = len(g)
    if col_exists(g, b_col):
        b = arr_num(g[b_col], default=0.0)
        b = np.maximum(b, 0.0)
        if b.sum() <= 1e-12:
            b = np.ones(n)
    else:
        b = np.ones(n)
    b = b / b.sum() * gross_target
    b[~np.isfinite(b)] = 0.0
    return b


def neutralize(y: np.ndarray, g: pd.DataFrame, cfg: Dict[str, Any]) -> np.ndarray:
    alpha_cfg = cfg.get("alpha", {})
    cols = cfg["columns"]
    if not alpha_cfg.get("neutralize", False):
        return y

    n = len(y)
    xs = [np.ones(n)]

    for c in alpha_cfg.get("neutralize_style_cols", []):
        if c in g.columns:
            xs.append(finite_zscore(arr_num(g[c])))

    ind_col = cols.get("industry")
    if col_exists(g, ind_col):
        d = pd.get_dummies(g[ind_col].astype(str), dummy_na=False)
        if d.shape[1] > 1:
            xs.append(d.iloc[:, 1:].astype(float).to_numpy())

    X = np.column_stack(xs)
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    if mask.sum() <= X.shape[1] + 5:
        return y
    try:
        beta = np.linalg.lstsq(X[mask], y[mask], rcond=None)[0]
        r = y - X @ beta
        r[~np.isfinite(r)] = 0.0
        return r
    except Exception as exc:
        warnings.warn(f"neutralize failed: {exc}")
        return y


def get_alpha(g: pd.DataFrame, cfg: Dict[str, Any]) -> np.ndarray:
    cols = cfg["columns"]
    a_cfg = cfg.get("alpha", {})
    x = arr_num(g[cols["signal"]])
    x = winsorize(x, float(a_cfg.get("winsor_low", 0.01)), float(a_cfg.get("winsor_high", 0.99)))

    method = a_cfg.get("method", "raw")
    if method == "raw":
        score = x
    elif method == "rank":
        score = pd.Series(x).rank(pct=True).to_numpy(dtype=float) - 0.5
    elif method == "zscore":
        score = finite_zscore(x)
    else:
        raise ValueError(f"Unknown alpha.method: {method}")

    score = neutralize(score, g, cfg)
    if a_cfg.get("standardize_after_neutralize", False):
        score = finite_zscore(score)
    alpha = score * float(a_cfg.get("scale", 1.0))
    alpha[~np.isfinite(alpha)] = 0.0
    return alpha


def get_limit_flags(g: pd.DataFrame, cfg: Dict[str, Any], bid: np.ndarray, ask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    exe = cfg.get("execution_constraints", {})
    cols = cfg["columns"]
    n = len(g)
    if not exe.get("use_limit_up_down", True):
        return np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)

    up = np.zeros(n, dtype=bool)
    down = np.zeros(n, dtype=bool)
    up_col = cols.get("limit_up_price")
    down_col = cols.get("limit_down_price")

    if col_exists(g, up_col):
        v = arr_num(g[up_col])
        raw = g[up_col]
        vals = set(raw.dropna().astype(str).unique().tolist())
        if vals and vals.issubset({"0", "1", "False", "True", "false", "true"}):
            up = raw.astype(bool).to_numpy()
        else:
            up = np.isfinite(v) & np.isfinite(ask) & (ask >= v * (1.0 - 1e-6))

    if col_exists(g, down_col):
        v = arr_num(g[down_col])
        raw = g[down_col]
        vals = set(raw.dropna().astype(str).unique().tolist())
        if vals and vals.issubset({"0", "1", "False", "True", "false", "true"}):
            down = raw.astype(bool).to_numpy()
        else:
            down = np.isfinite(v) & np.isfinite(bid) & (bid <= v * (1.0 + 1e-6))

    return up, down


def get_trade_caps(
    g: pd.DataFrame,
    cfg: Dict[str, Any],
    bid: np.ndarray,
    ask: np.ndarray,
    capital: float,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    exe = cfg.get("execution_constraints", {})
    cols = cfg["columns"]
    buy_caps = []
    sell_caps = []

    if exe.get("use_volume_cap", True):
        p = float(exe.get("participation", 0.10))
        ask_vol_col = cols.get("ask_volume")
        bid_vol_col = cols.get("bid_volume")
        if col_exists(g, ask_vol_col):
            ask_vol = arr_num(g[ask_vol_col], default=0.0)
            buy_caps.append(np.maximum(p * ask * ask_vol / capital, 0.0))
        if col_exists(g, bid_vol_col):
            bid_vol = arr_num(g[bid_vol_col], default=0.0)
            sell_caps.append(np.maximum(p * bid * bid_vol / capital, 0.0))

    if exe.get("use_market_turnover_cap", False):
        c = cols.get("market_turnover_amount")
        if col_exists(g, c):
            p = float(exe.get("market_turnover_participation", 0.02))
            cap = np.maximum(p * arr_num(g[c], default=0.0) / capital, 0.0)
            buy_caps.append(cap)
            sell_caps.append(cap)

    buy_cap = np.minimum.reduce(buy_caps) if buy_caps else None
    sell_cap = np.minimum.reduce(sell_caps) if sell_caps else None
    return buy_cap, sell_cap


def get_style_matrix(g: pd.DataFrame, cfg: Dict[str, Any]) -> Optional[np.ndarray]:
    r_cfg = cfg.get("risk_constraints", {})
    if not r_cfg.get("use_style_constraint", False):
        return None
    xs = []
    for c in r_cfg.get("style_cols", []):
        if c in g.columns:
            xs.append(finite_zscore(arr_num(g[c])))
    if not xs:
        return None
    X = np.column_stack(xs)
    X[~np.isfinite(X)] = 0.0
    return X


def get_industry_matrix(g: pd.DataFrame, cfg: Dict[str, Any]) -> Optional[np.ndarray]:
    r_cfg = cfg.get("risk_constraints", {})
    cols = cfg["columns"]
    if not r_cfg.get("use_industry_constraint", False):
        return None
    c = cols.get("industry")
    if not col_exists(g, c):
        return None
    d = pd.get_dummies(g[c].astype(str), dummy_na=False)
    if d.shape[1] <= 1:
        return None
    return d.astype(float).to_numpy()


# =============================================================================
# CVXPY optimization
# =============================================================================


def installed_solvers_from_cfg(cfg: Dict[str, Any]) -> List[str]:
    requested = cfg.get("solver", {}).get("solvers", ["OSQP", "CLARABEL", "SCS"])
    installed = set(cp.installed_solvers())
    out = [s for s in requested if s in installed]
    for s in ["OSQP", "CLARABEL", "SCS"]:
        if s in installed and s not in out:
            out.append(s)
    if not out:
        raise RuntimeError(f"No CVXPY solver available. installed={installed}")
    return out


def solve_problem(prob: cp.Problem, cfg: Dict[str, Any]) -> Tuple[bool, str, Optional[str]]:
    verbose = bool(cfg.get("solver", {}).get("verbose", False))
    last_status = "not_solved"
    for solver in installed_solvers_from_cfg(cfg):
        try:
            kwargs: Dict[str, Any] = {"solver": solver, "verbose": verbose, "warm_start": True}
            if solver == "OSQP":
                kwargs.update({"eps_abs": 1e-6, "eps_rel": 1e-6, "max_iter": 20000, "polish": True})
            prob.solve(**kwargs)
            last_status = str(prob.status)
            if prob.status in (cp.OPTIMAL,):
                return True, str(prob.status), solver
        except Exception as exc:
            last_status = f"{solver}_error: {exc}"
    return False, last_status, None


def solve_once(
    alpha: np.ndarray,
    b: np.ndarray,
    w_prev: np.ndarray,
    buy_cost: np.ndarray,
    sell_cost: np.ndarray,
    is_limit_up: np.ndarray,
    is_limit_down: np.ndarray,
    buy_cap: Optional[np.ndarray],
    sell_cap: Optional[np.ndarray],
    style_mat: Optional[np.ndarray],
    industry_mat: Optional[np.ndarray],
    turnover_cap: Optional[float],
    single_name_cap: float,
    cfg: Dict[str, Any],
    strict_gross: bool,
    use_liquidity: bool,
    use_risk: bool,
) -> Tuple[Optional[np.ndarray], str, Optional[str], Optional[float]]:
    p_cfg = cfg.get("portfolio", {})
    c_cfg = cfg.get("cost", {})
    r_cfg = cfg.get("risk_constraints", {})

    n = len(alpha)
    w = cp.Variable(n)
    buy = cp.Variable(n, nonneg=True)
    sell = cp.Variable(n, nonneg=True)
    active = w - b

    constraints = [
        w - w_prev == buy - sell,
        w >= 0.0,
        w <= single_name_cap,
        sell <= w_prev + 1e-12,
    ]

    gross_target = float(p_cfg.get("gross_target", 0.95))
    if strict_gross:
        constraints.append(cp.sum(w) == gross_target)
    else:
        constraints.append(cp.sum(w) >= float(p_cfg.get("gross_min", 0.90)))
        constraints.append(cp.sum(w) <= float(p_cfg.get("gross_max", 0.98)))

    active_l1 = float(p_cfg.get("active_l1_limit", 0.35))
    if active_l1 > 0:
        constraints.append(cp.norm1(active) <= active_l1)

    if turnover_cap is not None:
        constraints.append(cp.sum(buy + sell) <= float(turnover_cap))

    idx = np.where(is_limit_up)[0]
    if len(idx) > 0:
        constraints.append(buy[idx] == 0.0)
    idx = np.where(is_limit_down)[0]
    if len(idx) > 0:
        constraints.append(sell[idx] == 0.0)

    if use_liquidity:
        if buy_cap is not None:
            constraints.append(buy <= np.maximum(buy_cap, 0.0))
        if sell_cap is not None:
            constraints.append(sell <= np.maximum(sell_cap, 0.0))

    if use_risk:
        if style_mat is not None:
            tol = float(r_cfg.get("style_tol", 0.02))
            constraints.append(style_mat.T @ active <= tol)
            constraints.append(style_mat.T @ active >= -tol)
        if industry_mat is not None:
            tol = float(r_cfg.get("industry_tol", 0.02))
            constraints.append(industry_mat.T @ active <= tol)
            constraints.append(industry_mat.T @ active >= -tol)

    objective = cp.Maximize(
        alpha @ active
        - buy_cost @ buy
        - sell_cost @ sell
        - float(c_cfg.get("lambda_turnover", 0.0005)) * cp.sum(buy + sell)
        - float(p_cfg.get("lambda_active", 1e-4)) * cp.sum_squares(active)
    )
    prob = cp.Problem(objective, constraints)
    ok, status, solver = solve_problem(prob, cfg)
    if not ok or w.value is None:
        return None, status, solver, None

    out = np.asarray(w.value, dtype=float)
    out[~np.isfinite(out)] = 0.0
    out = np.clip(out, 0.0, single_name_cap)
    return out, status, solver, float(prob.value) if prob.value is not None else None


def fallback_target(b: np.ndarray, w_prev: np.ndarray, turnover_cap: Optional[float], single_name_cap: float, cfg: Dict[str, Any]) -> np.ndarray:
    gross_target = float(cfg.get("portfolio", {}).get("gross_target", 0.95))
    w = cap_and_rescale_nonnegative(b, gross_target, single_name_cap)
    if turnover_cap is not None:
        delta = w - w_prev
        tv = float(np.sum(np.abs(delta)))
        if tv > turnover_cap + 1e-12:
            w = w_prev + delta * (turnover_cap / tv)
            w = np.clip(w, 0.0, single_name_cap)
    return w


def solve_with_fallbacks(
    alpha: np.ndarray,
    b: np.ndarray,
    w_prev: np.ndarray,
    buy_cost: np.ndarray,
    sell_cost: np.ndarray,
    is_limit_up: np.ndarray,
    is_limit_down: np.ndarray,
    buy_cap: Optional[np.ndarray],
    sell_cap: Optional[np.ndarray],
    style_mat: Optional[np.ndarray],
    industry_mat: Optional[np.ndarray],
    turnover_cap: Optional[float],
    single_name_cap: float,
    cfg: Dict[str, Any],
) -> Tuple[np.ndarray, str, Optional[str], Optional[float], str]:
    gross_target = float(cfg.get("portfolio", {}).get("gross_target", 0.95))
    current_gross = float(np.sum(w_prev))
    loose_tc = None if turnover_cap is None else max(float(turnover_cap), abs(gross_target - current_gross) + 0.02)

    attempts = [
        ("strict", True, turnover_cap, True, True),
        ("loose_gross", False, turnover_cap, True, True),
        ("loose_turnover", False, loose_tc, True, True),
        ("drop_liquidity", False, loose_tc, False, True),
        ("drop_risk", False, loose_tc, False, False),
    ]
    last_status = "not_attempted"
    last_solver = None
    for name, strict, tc, use_liq, use_risk in attempts:
        w, status, solver, obj = solve_once(
            alpha=alpha,
            b=b,
            w_prev=w_prev,
            buy_cost=buy_cost,
            sell_cost=sell_cost,
            is_limit_up=is_limit_up,
            is_limit_down=is_limit_down,
            buy_cap=buy_cap,
            sell_cap=sell_cap,
            style_mat=style_mat,
            industry_mat=industry_mat,
            turnover_cap=tc,
            single_name_cap=single_name_cap,
            cfg=cfg,
            strict_gross=strict,
            use_liquidity=use_liq,
            use_risk=use_risk,
        )
        last_status = status
        last_solver = solver
        if w is not None:
            return w, status, solver, obj, name

    w = fallback_target(b, w_prev, turnover_cap, single_name_cap, cfg)
    return w, f"fallback_after_{last_status}", last_solver, None, "fallback"


# =============================================================================
# Post-processing and main loop
# =============================================================================


def postprocess_weights(
    w_raw: np.ndarray,
    w_prev: np.ndarray,
    mid: np.ndarray,
    capital: float,
    cfg: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    t_cfg = cfg.get("turnover", {})
    min_notional = float(t_cfg.get("min_trade_notional", 0.0))
    lot = int(t_cfg.get("lot_size", 1))

    w = w_raw.copy()
    if min_notional > 0:
        small = np.abs(w - w_prev) * capital < min_notional
        w[small] = w_prev[small]

    if bool(t_cfg.get("renormalize_after_min_trade", False)):
        p_cfg = cfg.get("portfolio", {})
        w = cap_and_rescale_nonnegative(
            w,
            float(p_cfg.get("gross_target", 0.95)),
            float(p_cfg.get("single_name_cap", 0.01)),
        )

    notional = w * capital
    shares = np.maximum(notional / mid, 0.0)
    shares[~np.isfinite(shares)] = 0.0
    if lot > 1:
        shares = np.floor(shares / lot) * lot
    w_final = shares * mid / capital
    w_final[~np.isfinite(w_final)] = 0.0
    return w_final, w_final * capital, shares


def run_optimizer(config_path: str) -> None:
    cfg = load_yaml(config_path)
    out_dir = cfg["output"]["dir"]
    ensure_dir(out_dir)
    shutil.copyfile(config_path, os.path.join(out_dir, cfg["output"].get("config_snapshot_file", "config_used.yaml")))

    df = prepare_df(load_input(cfg), cfg)
    rebalance_set = build_rebalance_timestamps(df, cfg)

    cols = cfg["columns"]
    date_col = cols["date"]
    dt_col = cols["datetime"]
    sym_col = cols["symbol"]

    capital = float(cfg.get("strategy", {}).get("capital", 200_000_000))
    p_cfg = cfg.get("portfolio", {})
    t_cfg = cfg.get("turnover", {})
    c_cfg = cfg.get("cost", {})

    gross_target = float(p_cfg.get("gross_target", 0.95))
    gross_hard_min = float(p_cfg.get("gross_hard_min", 0.80))
    base_cap = float(p_cfg.get("single_name_cap", 0.01))
    fee_rate = float(c_cfg.get("fee_rate", 0.001))

    shares_map: Dict[str, float] = {}
    used_turnover_by_date: Dict[str, float] = {}
    target_parts: List[pd.DataFrame] = []
    summary_rows: List[Dict[str, Any]] = []

    solved = 0
    for (date_value, ts), g_all in df.groupby([date_col, dt_col], sort=True):
        if ts not in rebalance_set:
            continue

        g_all = g_all.copy()
        mid_all, bid_all, ask_all = get_prices(g_all, cfg)
        sig_all = arr_num(g_all[cols["signal"]])
        valid = (
            np.isfinite(mid_all) & (mid_all > 0)
            & np.isfinite(bid_all) & (bid_all > 0)
            & np.isfinite(ask_all) & (ask_all > 0)
            & np.isfinite(sig_all)
        )
        if int(valid.sum()) < 20:
            print(f"[skip] {ts} valid={int(valid.sum())}")
            continue

        g = g_all.loc[valid].reset_index(drop=True)
        mid = mid_all[valid]
        bid = bid_all[valid]
        ask = ask_all[valid]
        sids = g[sym_col].astype(str).tolist()

        alpha = get_alpha(g, cfg)
        b = get_benchmark(g, cfg, gross_target)

        single_cap = base_cap
        if bool(p_cfg.get("cap_at_least_benchmark", True)) and len(b) > 0:
            single_cap = max(single_cap, float(np.max(b)) + 1e-8)
        if single_cap * len(g) < gross_target:
            new_cap = gross_target / len(g) * 1.05
            warnings.warn(f"single_name_cap too small, relax {single_cap:.6f} -> {new_cap:.6f}")
            single_cap = new_cap

        prev_shares = np.array([shares_map.get(s, 0.0) for s in sids], dtype=float)
        w_prev = prev_shares * mid / capital
        w_prev[~np.isfinite(w_prev)] = 0.0
        w_prev = np.maximum(w_prev, 0.0)
        current_gross = float(w_prev.sum())

        date_key = str(date_value)
        used_turnover_by_date.setdefault(date_key, 0.0)
        daily_budget = float(t_cfg.get("daily_budget", 0.80))
        remaining_daily = max(0.0, daily_budget - used_turnover_by_date[date_key])

        if current_gross < gross_hard_min:
            turnover_cap = float(t_cfg.get("initial_build_limit", 1.00))
            if not bool(t_cfg.get("initial_build_ignore_daily_budget", True)):
                turnover_cap = min(turnover_cap, remaining_daily)
        else:
            turnover_cap = min(float(t_cfg.get("limit_per_rebalance", 0.02)), remaining_daily)

        is_up, is_down = get_limit_flags(g, cfg, bid, ask)
        buy_cap, sell_cap = get_trade_caps(g, cfg, bid, ask, capital)
        style_mat = get_style_matrix(g, cfg)
        industry_mat = get_industry_matrix(g, cfg)

        buy_cost = ask / mid - 1.0 + fee_rate
        sell_cost = 1.0 - bid / mid + fee_rate
        buy_cost[~np.isfinite(buy_cost)] = fee_rate
        sell_cost[~np.isfinite(sell_cost)] = fee_rate
        buy_cost = np.maximum(buy_cost, fee_rate)
        sell_cost = np.maximum(sell_cost, fee_rate)

        w_raw, status, solver, obj, attempt = solve_with_fallbacks(
            alpha=alpha,
            b=b,
            w_prev=w_prev,
            buy_cost=buy_cost,
            sell_cost=sell_cost,
            is_limit_up=is_up,
            is_limit_down=is_down,
            buy_cap=buy_cap,
            sell_cap=sell_cap,
            style_mat=style_mat,
            industry_mat=industry_mat,
            turnover_cap=turnover_cap,
            single_name_cap=single_cap,
            cfg=cfg,
        )

        w_target, target_notional, target_shares = postprocess_weights(w_raw, w_prev, mid, capital, cfg)
        delta = w_target - w_prev
        buy_w = np.maximum(delta, 0.0)
        sell_w = np.maximum(-delta, 0.0)
        turnover = float(np.sum(buy_w + sell_w))
        used_turnover_by_date[date_key] += turnover

        active = w_target - b
        estimated_cost = float(buy_cost @ buy_w + sell_cost @ sell_w)
        expected_alpha = float(alpha @ active)
        gross = float(w_target.sum())
        active_l1 = float(np.abs(active).sum())

        for sid, sh in zip(sids, target_shares):
            if sh > 0:
                shares_map[sid] = float(sh)
            else:
                shares_map.pop(sid, None)

        out = pd.DataFrame({
            date_col: date_value,
            dt_col: ts,
            sym_col: sids,
            "pred_signal": arr_num(g[cols["signal"]]),
            "alpha": alpha,
            "benchmark_weight": b,
            "prev_weight": w_prev,
            "raw_target_weight": w_raw,
            "target_weight": w_target,
            "active_weight": active,
            "buy_weight": buy_w,
            "sell_weight": sell_w,
            "target_notional": target_notional,
            "target_shares": target_shares,
            "mid_price": mid,
            "bid1": bid,
            "ask1": ask,
            "buy_cost_rate": buy_cost,
            "sell_cost_rate": sell_cost,
            "is_limit_up": is_up,
            "is_limit_down": is_down,
        })
        target_parts.append(out)

        summary_rows.append({
            "date": date_value,
            "datetime": ts,
            "n": len(g),
            "status": status,
            "solver": solver,
            "attempt": attempt,
            "objective": obj,
            "gross": gross,
            "current_gross_before_trade": current_gross,
            "active_l1": active_l1,
            "turnover": turnover,
            "used_turnover_today": used_turnover_by_date[date_key],
            "turnover_cap": turnover_cap,
            "estimated_cost": estimated_cost,
            "expected_alpha": expected_alpha,
            "max_weight": float(np.max(w_target)) if len(w_target) else 0.0,
            "avg_buy_cost_rate": float(np.mean(buy_cost)),
            "avg_sell_cost_rate": float(np.mean(sell_cost)),
            "num_buy": int((buy_w > 1e-12).sum()),
            "num_sell": int((sell_w > 1e-12).sum()),
            "num_hold": int((np.abs(delta) <= 1e-12).sum()),
        })

        solved += 1
        if solved % 20 == 0:
            print(
                f"[solve] {solved} | {ts} | gross={gross:.4f} "
                f"turnover={turnover:.4f} cost={estimated_cost:.6f} "
                f"status={status} attempt={attempt}"
            )

    if not target_parts:
        raise RuntimeError("No target positions generated.")

    target_df = pd.concat(target_parts, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)

    target_path = os.path.join(out_dir, cfg["output"].get("target_file", "target_positions.csv"))
    summary_path = os.path.join(out_dir, cfg["output"].get("summary_file", "summary_by_rebalance.csv"))
    target_df.to_csv(target_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    final_summary = {
        "config": config_path,
        "output_dir": out_dir,
        "target_path": target_path,
        "summary_path": summary_path,
        "num_rebalances": int(len(summary_df)),
        "avg_gross": float(summary_df["gross"].mean()),
        "min_gross": float(summary_df["gross"].min()),
        "p10_gross": float(summary_df["gross"].quantile(0.10)),
        "avg_turnover": float(summary_df["turnover"].mean()),
        "total_turnover": float(summary_df["turnover"].sum()),
        "avg_estimated_cost": float(summary_df["estimated_cost"].mean()),
        "total_estimated_cost": float(summary_df["estimated_cost"].sum()),
        "avg_active_l1": float(summary_df["active_l1"].mean()),
        "avg_expected_alpha": float(summary_df["expected_alpha"].mean()),
    }

    json_path = os.path.join(out_dir, "final_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(final_summary, f, ensure_ascii=False, indent=2, default=str)

    print("\n========== DONE ==========")
    print(f"target:  {target_path}")
    print(f"summary: {summary_path}")
    print(f"json:    {json_path}")
    print("---------- final summary ----------")
    for k, v in final_summary.items():
        print(f"{k}: {v}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to pure_cs_cvxpy yaml config.")
    args = parser.parse_args()
    run_optimizer(args.config)


if __name__ == "__main__":
    main()
