# -*- coding: utf-8 -*-
"""
Pure CS Full-Invested Active-Sleeve Optimizer.

Portfolio:
  total weight = base_weight + active_sleeve_weight
  base sleeve  = broad ZZ2000 equal-weight, low turnover
  active sleeve = concentrated stateful topN from cross-sectional signal

Design:
  - full invested: sum target weights ~= 1.0
  - no market timing
  - no derisk
  - no up-ratio / market regime
  - active sleeve is long-only topN, funded by reducing base sleeve
  - benchmark for report: ZZY tradable EW benchmark, e.g. -7.95%

Recommended first case:
  res h20, rebalance 20min
  base_gross=0.85
  active_gross=0.15
  entry top100, exit500
"""

import argparse
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd


BASE_SCRIPT = Path("TakerOptimizer/scripts/build_zzy_pure_cs_fullinvested_active_positions.py")

spec = importlib.util.spec_from_file_location("purecs_fullinv_base", BASE_SCRIPT)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


def allocate_equal_with_caps(chosen_sids, caps_by_sid, target_gross: float) -> pd.Series:
    if len(chosen_sids) == 0 or target_gross <= 0:
        return pd.Series(dtype=float)

    caps = pd.Series(
        [float(caps_by_sid.get(sid, 0.0)) for sid in chosen_sids],
        index=list(chosen_sids),
        dtype=float,
    )
    caps = caps.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
    caps = caps[caps > 0]

    if caps.empty:
        return pd.Series(dtype=float)

    target = min(float(target_gross), float(caps.sum()))
    weights = pd.Series(0.0, index=caps.index, dtype=float)

    free = caps.copy()
    remaining = target

    for _ in range(100):
        if remaining <= 1e-12 or free.empty:
            break

        add = remaining / len(free)
        hit = free <= add + 1e-15

        if hit.any():
            weights.loc[free.index[hit]] += free.loc[hit]
            remaining -= float(free.loc[hit].sum())
            free = free.loc[~hit]
        else:
            weights.loc[free.index] += add
            remaining = 0.0
            break

    return weights[weights > 0]


def build_targets(df: pd.DataFrame, args) -> pd.DataFrame:
    df = df.copy()

    df["rank"] = (
        df.groupby(["date", "ts"])["score"]
        .rank(method="first", ascending=False)
    )

    df["base_weight"] = 0.0
    df["active_sleeve_weight"] = 0.0
    df["effective_target_weight"] = 0.0
    df["target_weight"] = 0.0
    df["desired_target_weight"] = 0.0
    df["gross_weight"] = 0.0
    df["active_selected"] = False
    df["hold_age"] = 0

    total_target = float(args.base_gross) + float(args.active_gross)
    if abs(total_target - 1.0) > 1e-9:
        raise ValueError(f"base_gross + active_gross must be 1.0, got {total_target}")

    for d, day_idx in df.groupby("date", sort=True).indices.items():
        day_idx = np.asarray(day_idx)
        day = df.iloc[day_idx].copy()
        ts_list = sorted(day["ts"].unique().tolist())

        if not ts_list:
            continue

        # Daily base universe: first rebalance snapshot of the day.
        # This makes the broad base sleeve much more stable than re-EW every 20min.
        first_ts = ts_list[0]
        first_idx = day_idx[day["ts"].to_numpy() == first_ts]
        base_universe = set(df.iloc[first_idx]["securityid"].astype(str).tolist())

        held_age = {}

        for ts in ts_list:
            idx = day_idx[day["ts"].to_numpy() == ts]
            g = df.iloc[idx].copy()

            # Base sleeve.
            if int(args.base_mode_daily_first) == 1:
                current_base = g[g["securityid"].astype(str).isin(base_universe)]["securityid"].astype(str).tolist()
            else:
                current_base = g["securityid"].astype(str).tolist()

            if len(current_base) > 0:
                base_w = float(args.base_gross) / len(current_base)
                row_locs = g.index[g["securityid"].astype(str).isin(set(current_base))]
                df.loc[row_locs, "base_weight"] = base_w

            # Active sleeve ranking.
            valid = g.sort_values("rank").copy()
            rank_by_sid = valid.set_index("securityid")["rank"].to_dict()

            keep = []
            for sid, age in held_age.items():
                if sid not in rank_by_sid:
                    continue

                force_keep = age < int(args.min_hold_rebalances)
                rank_keep = rank_by_sid[sid] <= int(args.exit_rank)

                if force_keep or rank_keep:
                    keep.append(sid)

            keep = sorted(
                keep,
                key=lambda sid: (
                    0 if held_age.get(sid, 0) < int(args.min_hold_rebalances) else 1,
                    rank_by_sid.get(sid, 1e18),
                ),
            )
            keep = keep[:int(args.target_n)]

            chosen = list(keep)
            chosen_set = set(chosen)

            entry_candidates = valid.loc[
                valid["rank"] <= int(args.entry_rank), "securityid"
            ].astype(str).tolist()

            for sid in entry_candidates:
                if sid not in chosen_set:
                    chosen.append(sid)
                    chosen_set.add(sid)
                if len(chosen) >= int(args.target_n):
                    break

            if int(args.fill_to_target) == 1 and len(chosen) < int(args.target_n):
                for sid in valid["securityid"].astype(str).tolist():
                    if sid not in chosen_set:
                        chosen.append(sid)
                        chosen_set.add(sid)
                    if len(chosen) >= int(args.target_n):
                        break

            chosen = chosen[:int(args.target_n)]

            if not chosen:
                held_age = {}
                continue

            valid_idxed = valid.set_index("securityid")

            caps = {}
            for sid in chosen:
                # active sleeve cap is constrained by total single-name cap minus base weight.
                sid_row = valid_idxed.loc[sid]
                base_w_sid = float(df.loc[sid_row.name if isinstance(sid_row, pd.Series) else sid_row.index[0], "base_weight"]) if False else 0.0

                # simpler and robust: get base weight from current df by sid
                base_rows = df.loc[idx]
                bw = float(base_rows.loc[base_rows["securityid"].astype(str) == sid, "base_weight"].iloc[0])

                cap_total_room = max(float(args.single_name_cap) - bw, 0.0)
                cap_active = min(float(args.active_single_name_cap), cap_total_room)

                if int(args.use_capacity_cap) == 1:
                    cap_capacity = float(valid_idxed.loc[sid, "capacity_weight"])
                    cap_active = min(cap_active, max(cap_capacity - bw, 0.0))

                caps[sid] = max(cap_active, 0.0)

            weights = allocate_equal_with_caps(
                chosen_sids=chosen,
                caps_by_sid=caps,
                target_gross=float(args.active_gross),
            )

            if weights.empty:
                held_age = {}
                continue

            row_map = valid.reset_index().set_index("securityid")["index"].to_dict()

            new_held_age = {}
            for sid, w in weights.items():
                ridx = int(row_map[sid])
                df.loc[ridx, "active_sleeve_weight"] = float(w)
                df.loc[ridx, "active_selected"] = True

                age = held_age.get(sid, 0) + 1
                df.loc[ridx, "hold_age"] = age
                new_held_age[sid] = age

            held_age = new_held_age

    df["effective_target_weight"] = df["base_weight"] + df["active_sleeve_weight"]
    df["target_weight"] = df["effective_target_weight"]
    df["desired_target_weight"] = df["effective_target_weight"]
    df["gross_weight"] = df["effective_target_weight"]

    df["selected"] = df["effective_target_weight"] > 0
    df["state"] = np.where(
        df["active_selected"],
        "FULLINV_BASE_PLUS_ACTIVE_SLEEVE",
        "FULLINV_BASE_ONLY",
    )
    df["side"] = np.where(df["selected"], "BUY", "NONE")
    df["blocked_reason"] = "none"

    # Current-snapshot EW benchmark weight, for diagnostics only.
    n_by_t = df.groupby(["date", "ts"])["securityid"].transform("size")
    df["ew_benchmark_weight"] = 1.0 / n_by_t

    df["active_vs_ew_weight"] = df["effective_target_weight"] - df["ew_benchmark_weight"]

    return df


def print_diagnostics(df: pd.DataFrame):
    g = (
        df.groupby(["execution_date", "execution_datetime"])
        .agg(
            gross=("effective_target_weight", "sum"),
            base_gross=("base_weight", "sum"),
            active_gross=("active_sleeve_weight", "sum"),
            n_hold=("selected", "sum"),
            n_active=("active_selected", "sum"),
            max_weight=("effective_target_weight", "max"),
            active_vs_ew_l1=("active_vs_ew_weight", lambda x: np.abs(x).sum()),
            avg_spread_bps=("spread_bps", "mean"),
        )
        .reset_index()
    )

    print("\n===== active-sleeve target diagnostics =====")
    print("snapshots:", len(g))

    for c in ["gross", "base_gross", "active_gross", "n_hold", "n_active", "max_weight", "active_vs_ew_l1", "avg_spread_bps"]:
        print(f"\n[{c}]")
        print(g[c].describe())

    # Target turnover proxy.
    total_turns = []
    active_turns = []
    day_total_turns = []
    day_active_turns = []

    for d, gd in df.groupby("execution_date", sort=True):
        prev = {}
        prev_active = {}
        day_total = 0.0
        day_active = 0.0

        for dt, gt in gd.groupby("execution_datetime", sort=True):
            cur = dict(zip(gt["securityid"], gt["effective_target_weight"]))
            cur_active = dict(zip(gt["securityid"], gt["active_sleeve_weight"]))

            names = set(prev) | set(cur)
            turn = sum(abs(cur.get(x, 0.0) - prev.get(x, 0.0)) for x in names)

            active_names = set(prev_active) | set(cur_active)
            active_turn = sum(abs(cur_active.get(x, 0.0) - prev_active.get(x, 0.0)) for x in active_names)

            total_turns.append(turn)
            active_turns.append(active_turn)

            day_total += turn
            day_active += active_turn

            prev = cur
            prev_active = cur_active

        day_total_turns.append(day_total)
        day_active_turns.append(day_active)

    print("\n===== target turnover proxy =====")
    print("avg_total_turnover_per_rebalance:", float(np.mean(total_turns)) if total_turns else np.nan)
    print("avg_total_turnover_per_day:", float(np.mean(day_total_turns)) if day_total_turns else np.nan)
    print("avg_active_turnover_per_rebalance:", float(np.mean(active_turns)) if active_turns else np.nan)
    print("avg_active_turnover_per_day:", float(np.mean(day_active_turns)) if day_active_turns else np.nan)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--data-root", default="/mnt/data1/zzy/optimizer_data")
    ap.add_argument("--model", choices=["res", "ts"], default="res")
    ap.add_argument("--horizon-min", type=int, required=True)
    ap.add_argument("--score-col", default="pred_z")
    ap.add_argument("--rebalance-min", type=int, required=True)

    ap.add_argument("--base-gross", type=float, default=0.85)
    ap.add_argument("--active-gross", type=float, default=0.15)

    ap.add_argument("--entry-rank", type=int, default=100)
    ap.add_argument("--exit-rank", type=int, default=500)
    ap.add_argument("--target-n", type=int, default=100)
    ap.add_argument("--min-hold-rebalances", type=int, default=0)
    ap.add_argument("--fill-to-target", type=int, default=1)

    ap.add_argument("--single-name-cap", type=float, default=0.003)
    ap.add_argument("--active-single-name-cap", type=float, default=0.003)

    ap.add_argument("--base-mode-daily-first", type=int, default=1)

    ap.add_argument("--max-spread-bps", type=float, default=50.0)
    ap.add_argument("--capital", type=float, default=200_000_000.0)
    ap.add_argument("--price-multiplier", type=float, default=0.01)
    ap.add_argument("--participation-rate", type=float, default=0.03)
    ap.add_argument("--capacity-cap-scale", type=float, default=3.0)
    ap.add_argument("--use-capacity-cap", type=int, default=0)

    ap.add_argument("--output", required=True)

    args = ap.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    df = base.load_joined_data(args)
    df = build_targets(df, args)

    out_cols = [
        "execution_date", "execution_datetime", "date", "minute", "securityid",

        "mid_price", "bid_price", "ask_price",
        "spread_bps", "spread_bps_realized",

        "base_weight", "active_sleeve_weight", "effective_target_weight",
        "target_weight", "desired_target_weight", "gross_weight",
        "ew_benchmark_weight", "active_vs_ew_weight",

        "selected", "active_selected", "state", "side", "blocked_reason",

        "score", "rank", "hold_age", "label_y_raw", "capacity_weight", "vol",
    ]

    df[out_cols].sort_values(
        ["execution_date", "execution_datetime", "securityid"]
    ).to_csv(out, index=False)

    print("\n[saved]", out)
    print("[rows]", len(df))
    print("[selected rows]", int(df["selected"].sum()))
    print("[active selected rows]", int(df["active_selected"].sum()))
    print_diagnostics(df)


if __name__ == "__main__":
    main()
