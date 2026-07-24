# -*- coding: utf-8 -*-
import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import cvxpy as cp


@dataclass
class Config:
    h10_dir: Path
    h20_dir: Path
    output_dir: Path
    tag: str

    capital: float = 200000000.0
    fee_bps: float = 10.0
    lot_size: int = 100
    min_trade_notional: float = 5000.0

    stock_core_gross: float = 0.95
    target_total_exposure: float = 1.00
    single_name_cap: float = 0.008

    active_l1_budget: float = 0.25
    target_top_frac: float = 0.10
    target_bottom_frac: float = 0.10
    target_weighting: str = "benchmark"  # benchmark / equal / score
    target_update_step_minutes: int = 20

    rebalance_step_minutes: int = 10
    turnover_limit: float = 0.025
    cash_buffer: float = 0.0005

    alpha10_delta_scale: float = 0.0003
    lambda_track: float = 1.0
    lambda_gross: float = 20.0
    lambda_turnover: float = 0.0018
    lambda_trade_ridge: float = 0.00001

    score_mode_h20: str = "raw"
    score_mode_h10: str = "rank_gate"
    score_r0: float = 0.25
    score_z0: float = 2.0
    score_clip_z: float = 5.0

    solvers: List[str] = None


@dataclass
class PortfolioState:
    actual_shares: Dict[str, int]
    sellable_shares: Dict[str, int]
    cash: float
    initialized: bool = False


def norm_date(s):
    return s.astype(str).str.replace(r"\D", "", regex=True).str.slice(0, 8).astype(int)


def norm_sid_series(df):
    if "sid" in df.columns:
        return pd.to_numeric(df["sid"], errors="coerce").astype("Int64")
    if "securityid" in df.columns:
        return pd.to_numeric(df["securityid"].astype(str).str.extract(r"(\d+)")[0], errors="coerce").astype("Int64")
    if "SecurityID" in df.columns:
        return pd.to_numeric(df["SecurityID"].astype(str).str.extract(r"(\d+)")[0], errors="coerce").astype("Int64")
    raise KeyError(f"cannot find sid/securityid columns: {df.columns.tolist()}")


def pick_col(df, candidates, name, required=True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"cannot find {name}; candidates={candidates}; columns={df.columns.tolist()}")
    return None


def score_transform(x, mode, r0=0.25, z0=2.0, clip_z=5.0):
    s = pd.to_numeric(pd.Series(x), errors="coerce").fillna(0.0)
    if mode == "raw":
        return s.to_numpy(dtype=float)

    mu = s.mean()
    sd = s.std(ddof=0)
    z = (s - mu) / sd if sd and sd > 1e-12 else s * 0.0
    z = z.clip(-clip_z, clip_z)

    if mode == "csz":
        return z.to_numpy(dtype=float)

    pct = s.rank(pct=True).fillna(0.5)
    r = (pct - 0.5) * 2.0

    if mode == "rank":
        return r.to_numpy(dtype=float)

    if mode == "rank_gate":
        gate = ((r.abs() >= r0) | (z.abs() >= z0)).astype(float)
        return (r * gate).to_numpy(dtype=float)

    raise ValueError(f"unknown score mode: {mode}")


def should_step(dt, first_dt, step_minutes):
    delta_min = int((dt - first_dt).total_seconds() // 60)
    return delta_min >= 0 and delta_min % int(step_minutes) == 0


def read_day_inputs(h20_file: Path, h10_dir: Path) -> pd.DataFrame:
    h20 = pd.read_parquet(h20_file)

    if "date" in h20.columns:
        h20["date"] = norm_date(h20["date"])
    else:
        h20["date"] = int(h20_file.stem.replace("optimizer_input_", ""))

    h20["datetime"] = pd.to_datetime(h20["datetime"])
    h20["sid"] = norm_sid_series(h20)
    h20["securityid"] = h20["sid"].astype(str).str.zfill(6)

    px_col = pick_col(h20, ["mid_price", "tmid", "price", "mark_price"], "mid price")
    h20["mid_price"] = pd.to_numeric(h20[px_col], errors="coerce")

    if "benchmark_weight" not in h20.columns:
        h20["benchmark_weight"] = 0.0
    h20["benchmark_weight"] = pd.to_numeric(h20["benchmark_weight"], errors="coerce").fillna(0.0).clip(lower=0.0)

    if "pred_ret_h20" not in h20.columns:
        if "pred_ret" in h20.columns:
            h20["pred_ret_h20"] = h20["pred_ret"]
        elif "pred" in h20.columns:
            h20["pred_ret_h20"] = h20["pred"]
        else:
            raise KeyError(f"{h20_file} has no pred_ret_h20/pred_ret/pred")

    h20["pred_ret_h20"] = pd.to_numeric(h20["pred_ret_h20"], errors="coerce").fillna(0.0)

    h10_file = h10_dir / h20_file.name
    if h10_file.exists():
        h10 = pd.read_parquet(h10_file)
        h10["datetime"] = pd.to_datetime(h10["datetime"])
        h10["sid"] = norm_sid_series(h10)

        if "pred_ret_h10" not in h10.columns:
            if "pred_ret" in h10.columns:
                h10["pred_ret_h10"] = h10["pred_ret"]
            elif "pred" in h10.columns:
                h10["pred_ret_h10"] = h10["pred"]
            else:
                h10["pred_ret_h10"] = 0.0

        h10 = h10[["datetime", "sid", "pred_ret_h10"]].drop_duplicates(["datetime", "sid"], keep="last")
        df = h20.merge(h10, on=["datetime", "sid"], how="left")
    else:
        df = h20.copy()
        df["pred_ret_h10"] = 0.0

    df["pred_ret_h10"] = pd.to_numeric(df["pred_ret_h10"], errors="coerce").fillna(0.0)

    keep = [
        "date", "datetime", "sid", "securityid", "mid_price",
        "benchmark_weight", "pred_ret_h20", "pred_ret_h10",
    ]
    df = df[keep].dropna(subset=["mid_price", "sid", "datetime"]).copy()
    df = df.sort_values(["datetime", "sid"]).reset_index(drop=True)
    return df


def initialize_portfolio(g: pd.DataFrame, cfg: Config) -> PortfolioState:
    b = pd.to_numeric(g["benchmark_weight"], errors="coerce").fillna(0.0).clip(lower=0.0)
    if b.sum() <= 1e-12:
        b = pd.Series(1.0 / len(g), index=g.index)
    else:
        b = b / b.sum()

    mid = pd.to_numeric(g["mid_price"], errors="coerce").fillna(0.0)
    shares = {}
    cash = float(cfg.capital)
    fee = cfg.fee_bps / 10000.0

    for idx, row in g.iterrows():
        sid = str(row["securityid"])
        px = float(mid.loc[idx])
        if px <= 0:
            continue
        target_notional = cfg.capital * cfg.stock_core_gross * float(b.loc[idx])
        sh = int(target_notional // (px * cfg.lot_size)) * cfg.lot_size
        if sh <= 0:
            continue
        cost = sh * px * (1.0 + fee)
        if cost <= cash:
            shares[sid] = sh
            cash -= cost

    return PortfolioState(actual_shares=shares.copy(), sellable_shares=shares.copy(), cash=cash, initialized=True)


def get_current_arrays(state: PortfolioState, g: pd.DataFrame, equity: float):
    n = len(g)
    actual = np.zeros(n)
    sellable = np.zeros(n)
    mid = g["mid_price"].to_numpy(dtype=float)
    for i, sid in enumerate(g["securityid"].astype(str)):
        sh = state.actual_shares.get(sid, 0)
        sel = state.sellable_shares.get(sid, 0)
        actual[i] = sh * mid[i] / max(equity, 1e-12)
        sellable[i] = sel * mid[i] / max(equity, 1e-12)
    locked = np.maximum(actual - sellable, 0.0)
    return actual, sellable, locked


def compute_equity(state: PortfolioState, g: pd.DataFrame):
    px = dict(zip(g["securityid"].astype(str), g["mid_price"].astype(float)))
    pos_val = 0.0
    for sid, sh in state.actual_shares.items():
        if sid in px:
            pos_val += int(sh) * float(px[sid])
    return float(state.cash + pos_val), float(pos_val)


def build_alpha_target(g: pd.DataFrame, cfg: Config) -> np.ndarray:
    n = len(g)
    b_raw = pd.to_numeric(g["benchmark_weight"], errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(dtype=float)
    if b_raw.sum() <= 1e-12:
        b = np.ones(n) / max(n, 1) * cfg.stock_core_gross
    else:
        b = b_raw / b_raw.sum() * cfg.stock_core_gross

    score = score_transform(
        g["pred_ret_h20"],
        cfg.score_mode_h20,
        cfg.score_r0,
        cfg.score_z0,
        cfg.score_clip_z,
    )

    w = b.copy()

    active_oneway = float(cfg.active_l1_budget) / 2.0
    if active_oneway <= 1e-12:
        return w

    top_n = max(1, int(n * cfg.target_top_frac))
    bottom_n = max(1, int(n * cfg.target_bottom_frac))

    order = np.argsort(score)
    bottom_idx = order[:bottom_n]
    top_idx = order[-top_n:]

    sell_budget = min(active_oneway, float(w[bottom_idx].sum()))
    if sell_budget <= 1e-12:
        return w

    # sell bottom proportional to existing benchmark-core weight
    bw_bottom = w[bottom_idx].copy()
    if bw_bottom.sum() > 1e-12:
        sell_alloc = sell_budget * bw_bottom / bw_bottom.sum()
        w[bottom_idx] -= sell_alloc

    if cfg.target_weighting == "benchmark":
        top_base = b[top_idx].clip(min=0.0)
        if top_base.sum() > 1e-12:
            buy_alloc = sell_budget * top_base / top_base.sum()
        else:
            buy_alloc = np.ones(len(top_idx)) * sell_budget / len(top_idx)
    elif cfg.target_weighting == "score":
        s = score[top_idx] - np.min(score[top_idx])
        s = np.maximum(s, 0.0) + 1e-12
        buy_alloc = sell_budget * s / s.sum()
    else:
        buy_alloc = np.ones(len(top_idx)) * sell_budget / len(top_idx)

    w[top_idx] += buy_alloc

    w = np.clip(w, 0.0, cfg.single_name_cap)
    s = w.sum()
    if s > 1e-12:
        w = w * cfg.stock_core_gross / s
    return w


def solve_execution_step(
    g: pd.DataFrame,
    state: PortfolioState,
    cfg: Config,
    target_w: np.ndarray,
    dt0: pd.Timestamp,
) -> Tuple[np.ndarray, np.ndarray, str, Dict]:
    equity, pos_val = compute_equity(state, g)
    w_cur, w_sellable, w_locked = get_current_arrays(state, g, equity)

    n = len(g)
    score10 = score_transform(
        g["pred_ret_h10"],
        cfg.score_mode_h10,
        cfg.score_r0,
        cfg.score_z0,
        cfg.score_clip_z,
    )

    buy = cp.Variable(n, nonneg=True)
    sell = cp.Variable(n, nonneg=True)
    delta = buy - sell
    turnover = buy + sell
    w_next = w_cur + delta

    fee = cfg.fee_bps / 10000.0
    cash_weight = state.cash / max(equity, 1e-12)

    constraints = [
        sell <= w_sellable,
        w_next >= w_locked,
        w_next >= 0,
        w_next <= cfg.single_name_cap,
        cp.sum(turnover) <= cfg.turnover_limit,
        cp.sum(w_next) <= min(cfg.stock_core_gross + 0.03, 0.99),
        cp.sum(w_next) >= max(cfg.stock_core_gross - 0.05, 0.0),
        (1.0 + fee) * cp.sum(buy) <= cash_weight + (1.0 - fee) * cp.sum(sell) - cfg.cash_buffer,
    ]

    objective = cp.Minimize(
        cfg.lambda_track * cp.sum_squares(w_next - target_w)
        + cfg.lambda_gross * cp.square(cp.sum(w_next) - cfg.stock_core_gross)
        + cfg.lambda_turnover * cp.sum(turnover)
        + cfg.lambda_trade_ridge * cp.sum_squares(delta)
        - cfg.alpha10_delta_scale * (score10 @ delta)
    )

    prob = cp.Problem(objective, constraints)

    last_status = "not_solved"
    for solver in cfg.solvers:
        try:
            prob.solve(solver=solver, warm_start=True, verbose=False)
            last_status = str(prob.status)
            if prob.status in ("optimal", "optimal_inaccurate") and buy.value is not None and sell.value is not None:
                return np.asarray(buy.value).clip(min=0.0), np.asarray(sell.value).clip(min=0.0), last_status, {
                    "equity": equity,
                    "pos_val": pos_val,
                    "cash_weight": cash_weight,
                    "gross_before": float(w_cur.sum()),
                    "target_gross": float(target_w.sum()),
                    "track_l2_before": float(np.sqrt(np.mean((w_cur - target_w) ** 2))),
                    "turnover_cont": float(np.sum(np.asarray(buy.value) + np.asarray(sell.value))),
                    "gross_after_cont": float(np.sum(w_cur + np.asarray(buy.value) - np.asarray(sell.value))),
                }
        except Exception as e:
            last_status = f"{solver}_error:{type(e).__name__}"

    return np.zeros(n), np.zeros(n), f"fallback_{last_status}", {
        "equity": equity,
        "pos_val": pos_val,
        "cash_weight": cash_weight,
        "gross_before": float(w_cur.sum()),
        "target_gross": float(target_w.sum()),
        "track_l2_before": float(np.sqrt(np.mean((w_cur - target_w) ** 2))),
        "turnover_cont": 0.0,
        "gross_after_cont": float(w_cur.sum()),
    }


def apply_trades(g: pd.DataFrame, state: PortfolioState, buy_w: np.ndarray, sell_w: np.ndarray, equity: float, cfg: Config):
    fee = cfg.fee_bps / 10000.0
    lot = int(cfg.lot_size)

    mid = g["mid_price"].to_numpy(dtype=float)
    sids = g["securityid"].astype(str).to_numpy()

    total_sell_notional = 0.0
    total_buy_notional = 0.0

    # sells first
    for sid, px, sw in zip(sids, mid, sell_w):
        if px <= 0 or sw <= 0:
            continue
        desired_notional = float(sw) * equity
        sh = int(desired_notional // (px * lot)) * lot
        sh = min(sh, int(state.sellable_shares.get(sid, 0)))
        sh = min(sh, int(state.actual_shares.get(sid, 0)))
        if sh <= 0:
            continue
        notional = sh * px
        state.actual_shares[sid] = int(state.actual_shares.get(sid, 0) - sh)
        state.sellable_shares[sid] = int(state.sellable_shares.get(sid, 0) - sh)
        if state.actual_shares.get(sid, 0) <= 0:
            state.actual_shares.pop(sid, None)
        if state.sellable_shares.get(sid, 0) <= 0:
            state.sellable_shares.pop(sid, None)
        state.cash += notional * (1.0 - fee)
        total_sell_notional += notional

    # buys
    for sid, px, bw in zip(sids, mid, buy_w):
        if px <= 0 or bw <= 0:
            continue
        desired_notional = float(bw) * equity
        max_cash_notional = max(state.cash - cfg.cash_buffer * equity, 0.0) / (1.0 + fee)
        desired_notional = min(desired_notional, max_cash_notional)
        if desired_notional < cfg.min_trade_notional:
            continue
        sh = int(desired_notional // (px * lot)) * lot
        if sh <= 0:
            continue
        cost = sh * px * (1.0 + fee)
        if cost > state.cash:
            continue
        state.actual_shares[sid] = int(state.actual_shares.get(sid, 0) + sh)
        # T+1: bought shares not sellable today
        state.cash -= cost
        total_buy_notional += sh * px

    return total_buy_notional, total_sell_notional


def dump_positions_rows(g: pd.DataFrame, state: PortfolioState):
    return pd.DataFrame({
        "date": g["date"].iloc[0],
        "datetime": g["datetime"],
        "securityid": g["securityid"].astype(str).values,
        "actual_shares_after": [int(state.actual_shares.get(sid, 0)) for sid in g["securityid"].astype(str).values],
    })


def run(cfg: Config):
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    print("===== v25 config =====")
    for k, v in asdict(cfg).items():
        print(f"{k}: {v}")

    h20_files = sorted(Path(cfg.h20_dir).glob("optimizer_input_*.parquet"))
    if not h20_files:
        raise FileNotFoundError(cfg.h20_dir)

    state = PortfolioState(actual_shares={}, sellable_shares={}, cash=cfg.capital, initialized=False)
    last_date = None

    all_pos = []
    reb_rows = []

    last_target_by_sid = None

    for day_i, f in enumerate(h20_files, start=1):
        df = read_day_inputs(f, cfg.h10_dir)
        d0 = int(df["date"].iloc[0])
        print(f"\n===== load day {day_i}/{len(h20_files)}: {f.name} date={d0} rows={len(df)} =====", flush=True)

        if last_date is None or d0 != last_date:
            if state.initialized:
                state.sellable_shares = state.actual_shares.copy()
            last_date = d0

        times = sorted(df["datetime"].unique())
        first_dt = pd.Timestamp(times[0])

        for dt in times:
            dt = pd.Timestamp(dt)
            g = df[df["datetime"] == dt].copy().reset_index(drop=True)

            if not state.initialized:
                state = initialize_portfolio(g, cfg)
                print(f"[init] {dt} cash={state.cash:.2f} names={len(state.actual_shares)}", flush=True)

            if should_step(dt, first_dt, cfg.target_update_step_minutes) or last_target_by_sid is None:
                target = build_alpha_target(g, cfg)
                last_target_by_sid = dict(zip(g["securityid"].astype(str), target))
                target_status = "target_updated"
            else:
                target = np.array([last_target_by_sid.get(sid, 0.0) for sid in g["securityid"].astype(str)])
                target_status = "target_held"

            if should_step(dt, first_dt, cfg.rebalance_step_minutes):
                buy_w, sell_w, status, info = solve_execution_step(g, state, cfg, target, dt)
                equity_before = info["equity"]

                buy_notional, sell_notional = apply_trades(g, state, buy_w, sell_w, equity_before, cfg)

                equity_after, pos_val_after = compute_equity(state, g)
                gross_after = pos_val_after / max(equity_after, 1e-12)
                turnover_weight = (buy_notional + sell_notional) / max(equity_before, 1e-12)
                cost = (buy_notional + sell_notional) * cfg.fee_bps / 10000.0

                row = {
                    "date": d0,
                    "datetime": dt,
                    "status": status,
                    "target_status": target_status,
                    "equity_before": equity_before,
                    "equity_after": equity_after,
                    "cash_after": state.cash,
                    "gross_before": info["gross_before"],
                    "target_gross": info["target_gross"],
                    "gross_after_cont": info["gross_after_cont"],
                    "gross_after_actual": gross_after,
                    "track_l2_before": info["track_l2_before"],
                    "turnover_cont": info["turnover_cont"],
                    "turnover_weight": turnover_weight,
                    "buy_notional": buy_notional,
                    "sell_notional": sell_notional,
                    "cost": cost,
                    "cost_notional": cost,
                    "fee_cost": cost,
                    "trade_cost": cost,
                    "turnover_notional": buy_notional + sell_notional,
                    "n_hold": len(state.actual_shares),
                }
                reb_rows.append(row)

                all_pos.append(dump_positions_rows(g, state))

                print(
                    f"[reb] {dt} status={status} gross_actual={gross_after:.4f} "
                    f"target={info['target_gross']:.4f} turn={turnover_weight:.4f} cash={state.cash:.2f}",
                    flush=True,
                )

    pos = pd.concat(all_pos, ignore_index=True)
    reb = pd.DataFrame(reb_rows)

    pos_path = cfg.output_dir / "target_positions.csv"
    reb_path = cfg.output_dir / "summary_by_rebalance.csv"
    cfg_path = cfg.output_dir / "config_used.csv"

    pos.to_csv(pos_path, index=False)
    reb.to_csv(reb_path, index=False)
    pd.DataFrame([asdict(cfg)]).to_csv(cfg_path, index=False)

    print("\n===== DONE =====")
    print("positions:", pos_path)
    print("rebalance:", reb_path)
    print("config:", cfg_path)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h10-dir", required=True, type=Path)
    ap.add_argument("--h20-dir", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--tag", required=True)

    ap.add_argument("--capital", type=float, default=200000000.0)
    ap.add_argument("--fee-bps", type=float, default=10.0)
    ap.add_argument("--lot-size", type=int, default=100)
    ap.add_argument("--min-trade-notional", type=float, default=5000.0)

    ap.add_argument("--stock-core-gross", type=float, default=0.95)
    ap.add_argument("--target-total-exposure", type=float, default=1.0)
    ap.add_argument("--single-name-cap", type=float, default=0.008)

    ap.add_argument("--active-l1-budget", type=float, default=0.25)
    ap.add_argument("--target-top-frac", type=float, default=0.10)
    ap.add_argument("--target-bottom-frac", type=float, default=0.10)
    ap.add_argument("--target-weighting", choices=["benchmark", "equal", "score"], default="benchmark")
    ap.add_argument("--target-update-step-minutes", type=int, default=20)

    ap.add_argument("--rebalance-step-minutes", type=int, default=10)
    ap.add_argument("--turnover-limit", type=float, default=0.025)
    ap.add_argument("--cash-buffer", type=float, default=0.0005)

    ap.add_argument("--alpha10-delta-scale", type=float, default=0.0003)
    ap.add_argument("--lambda-track", type=float, default=1.0)
    ap.add_argument("--lambda-gross", type=float, default=20.0)
    ap.add_argument("--lambda-turnover", type=float, default=0.0018)
    ap.add_argument("--lambda-trade-ridge", type=float, default=0.00001)

    ap.add_argument("--score-mode-h20", default="raw")
    ap.add_argument("--score-mode-h10", default="rank_gate")
    ap.add_argument("--score-r0", type=float, default=0.25)
    ap.add_argument("--score-z0", type=float, default=2.0)
    ap.add_argument("--score-clip-z", type=float, default=5.0)

    ap.add_argument("--solvers", default="OSQP,SCS,CLARABEL")
    args = ap.parse_args()

    solvers = [x.strip() for x in args.solvers.split(",") if x.strip()]

    return Config(
        h10_dir=args.h10_dir,
        h20_dir=args.h20_dir,
        output_dir=args.output_dir,
        tag=args.tag,
        capital=args.capital,
        fee_bps=args.fee_bps,
        lot_size=args.lot_size,
        min_trade_notional=args.min_trade_notional,
        stock_core_gross=args.stock_core_gross,
        target_total_exposure=args.target_total_exposure,
        single_name_cap=args.single_name_cap,
        active_l1_budget=args.active_l1_budget,
        target_top_frac=args.target_top_frac,
        target_bottom_frac=args.target_bottom_frac,
        target_weighting=args.target_weighting,
        target_update_step_minutes=args.target_update_step_minutes,
        rebalance_step_minutes=args.rebalance_step_minutes,
        turnover_limit=args.turnover_limit,
        cash_buffer=args.cash_buffer,
        alpha10_delta_scale=args.alpha10_delta_scale,
        lambda_track=args.lambda_track,
        lambda_gross=args.lambda_gross,
        lambda_turnover=args.lambda_turnover,
        lambda_trade_ridge=args.lambda_trade_ridge,
        score_mode_h20=args.score_mode_h20,
        score_mode_h10=args.score_mode_h10,
        score_r0=args.score_r0,
        score_z0=args.score_z0,
        score_clip_z=args.score_clip_z,
        solvers=solvers,
    )


if __name__ == "__main__":
    run(parse_args())
