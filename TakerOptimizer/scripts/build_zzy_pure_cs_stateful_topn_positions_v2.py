# -*- coding: utf-8 -*-
import argparse
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

BASE_SCRIPT = Path("TakerOptimizer/scripts/build_zzy_pure_cs_stateful_topn_positions.py")

spec = importlib.util.spec_from_file_location("purecs_base", BASE_SCRIPT)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


def build_stateful_targets_v2(df: pd.DataFrame, args) -> pd.DataFrame:
    df = df.copy()

    df["rank"] = np.nan
    mask = df["valid_rank"].to_numpy()

    df.loc[mask, "rank"] = (
        df.loc[mask]
        .groupby(["date", "ts"])["score"]
        .rank(method="first", ascending=False)
    )

    n = len(df)
    target = np.zeros(n, dtype=float)
    selected = np.zeros(n, dtype=bool)
    hold_age_arr = np.zeros(n, dtype=np.int32)

    for d, day_idx in df.groupby("date", sort=True).indices.items():
        day_idx = np.asarray(day_idx)
        day = df.iloc[day_idx]
        ts_list = sorted(day["ts"].unique().tolist())

        flat_ts = set()
        if args.force_flat_eod:
            flat_ts = set(ts_list[-int(args.flat_last_n_rebalances):])

        # held_age: sid -> number of rebalances already held
        held_age = {}

        for ts in ts_list:
            idx = day_idx[day["ts"].to_numpy() == ts]
            g = df.iloc[idx]

            if ts in flat_ts:
                held_age = {}
                continue

            valid = g[g["rank"].notna()].copy()
            if valid.empty:
                held_age = {}
                continue

            valid = valid.sort_values("rank")
            rank_by_sid = valid.set_index("securityid")["rank"].to_dict()

            # keep if within min hold OR still inside exit band
            keep = []
            for sid, age in held_age.items():
                if sid not in rank_by_sid:
                    continue

                force_keep = age < int(args.min_hold_rebalances)
                rank_keep = rank_by_sid[sid] <= int(args.exit_rank)

                if force_keep or rank_keep:
                    keep.append(sid)

            # sort: forced min-hold names first, then better ranks
            keep = sorted(
                keep,
                key=lambda sid: (
                    0 if held_age.get(sid, 0) < int(args.min_hold_rebalances) else 1,
                    rank_by_sid.get(sid, 1e18),
                )
            )
            keep = keep[:int(args.target_n)]

            chosen = list(keep)
            chosen_set = set(chosen)

            # add fresh entry names
            entry_candidates = valid.loc[
                valid["rank"] <= int(args.entry_rank), "securityid"
            ].tolist()

            for sid in entry_candidates:
                if sid not in chosen_set:
                    chosen.append(sid)
                    chosen_set.add(sid)
                if len(chosen) >= int(args.target_n):
                    break

            # optionally fill to target_n with best-ranked valid names
            if args.fill_to_target and len(chosen) < int(args.target_n):
                for sid in valid["securityid"].tolist():
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

            if args.use_capacity_cap:
                caps = {}
                for sid in chosen:
                    cap = min(
                        float(args.single_name_limit),
                        float(valid_idxed.loc[sid, "capacity_weight"]),
                    )
                    caps[sid] = max(cap, 0.0)
            else:
                caps = {sid: float(args.single_name_limit) for sid in chosen}

            weights = base.allocate_equal_with_caps(
                chosen_sids=chosen,
                caps_by_sid=caps,
                target_gross=float(args.target_gross),
            )

            if weights.empty:
                held_age = {}
                continue

            row_map = valid.reset_index().set_index("securityid")["index"].to_dict()

            new_held_age = {}
            for sid, w in weights.items():
                ridx = int(row_map[sid])
                target[ridx] = float(w)
                selected[ridx] = True

                age = held_age.get(sid, 0) + 1
                new_held_age[sid] = age
                hold_age_arr[ridx] = age

            held_age = new_held_age

    df["effective_target_weight"] = target
    df["target_weight"] = target
    df["desired_target_weight"] = target
    df["gross_weight"] = target
    df["selected"] = selected
    df["hold_age"] = hold_age_arr

    df["state"] = np.where(df["selected"], "PURE_CS_STATEFUL_TOPN_V2", "FLAT")
    df["side"] = np.where(df["selected"], "BUY", "NONE")
    df["blocked_reason"] = "none"

    return df


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--data-root", default="/mnt/data1/zzy/optimizer_data")
    ap.add_argument("--model", choices=["res", "ts"], default="res")
    ap.add_argument("--horizon-min", type=int, required=True)
    ap.add_argument("--score-col", default="pred_z")

    ap.add_argument("--rebalance-min", type=int, required=True)
    ap.add_argument("--entry-rank", type=int, default=100)
    ap.add_argument("--exit-rank", type=int, default=500)
    ap.add_argument("--target-n", type=int, default=100)
    ap.add_argument("--min-hold-rebalances", type=int, default=0)

    ap.add_argument("--target-gross", type=float, default=0.10)
    ap.add_argument("--single-name-limit", type=float, default=0.003)

    ap.add_argument("--fill-to-target", type=int, default=1)
    ap.add_argument("--force-flat-eod", type=int, default=1)
    ap.add_argument("--flat-last-n-rebalances", type=int, default=1)

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
    df = build_stateful_targets_v2(df, args)

    out_cols = [
        "execution_date", "execution_datetime", "date", "minute", "securityid",
        "mid_price", "bid_price", "ask_price",
        "spread_bps", "spread_bps_realized",
        "effective_target_weight", "target_weight", "desired_target_weight",
        "gross_weight", "selected", "state", "side", "blocked_reason",
        "score", "rank", "hold_age", "label_y_raw", "capacity_weight", "vol",
    ]

    df[out_cols].sort_values(
        ["execution_date", "execution_datetime", "securityid"]
    ).to_csv(out, index=False)

    print("\n[saved]", out)
    print("[rows]", len(df))
    print("[selected rows]", int(df["selected"].sum()))
    base.print_target_diagnostics(df)

    if "hold_age" in df.columns:
        print("\n===== hold age describe, selected only =====")
        print(df.loc[df["selected"], "hold_age"].describe())


if __name__ == "__main__":
    main()
