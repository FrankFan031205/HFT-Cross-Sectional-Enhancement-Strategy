# -*- coding: utf-8 -*-
"""
Pure CS Stateful TopN Optimizer for ZZY ZZ2000 signals.

Design:
  - no market timing
  - no up-ratio / risk-off / derisk
  - fixed target gross
  - pure cross-sectional rank selection
  - entry rank <= entry_rank
  - hold existing names until rank > exit_rank
  - rebalance every N minutes
  - output positions directly usable by TakerModel v8

Input:
  /mnt/data1/zzy/optimizer_data/pred/<h>min/test_predictions.parquet
  /mnt/data1/zzy/optimizer_data/pred_ts/<h>min/test_predictions.parquet
  /mnt/data1/zzy/optimizer_data/quotes/TimeSeries/<date>/<sid>/quotes.parquet

Output:
  positions_for_taker_model.csv
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl


KEY = ["date", "sid", "ts"]


def get_pred_path(data_root: Path, model: str, horizon_min: int) -> Path:
    if model == "res":
        return data_root / "pred" / f"{horizon_min}min" / "test_predictions.parquet"
    if model == "ts":
        return data_root / "pred_ts" / f"{horizon_min}min" / "test_predictions.parquet"
    raise ValueError("model must be res or ts")


def allocate_equal_with_caps(chosen_sids, caps_by_sid, target_gross: float) -> pd.Series:
    """
    Equal-weight allocation with upper caps.

    If total caps < target_gross, realized gross becomes sum(caps).
    """
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


def make_execution_datetime(df: pd.DataFrame) -> pd.Series:
    tsr = df["ts_real"].astype("int64")
    div = 1000 if tsr.max() > 235959 else 1

    hhmmss = (tsr // div).astype("int64").astype(str).str.zfill(6)
    d = df["date"].astype("int64").astype(str)

    return (
        d.str.slice(0, 4) + "-" +
        d.str.slice(4, 6) + "-" +
        d.str.slice(6, 8) + " " +
        hhmmss.str.slice(0, 2) + ":" +
        hhmmss.str.slice(2, 4) + ":" +
        hhmmss.str.slice(4, 6)
    )


def load_joined_data(args) -> pd.DataFrame:
    data_root = Path(args.data_root)
    pred_path = get_pred_path(data_root, args.model, args.horizon_min)

    if not pred_path.exists():
        raise FileNotFoundError(f"prediction file not found: {pred_path}")

    quote_glob = str(data_root / "quotes" / "TimeSeries" / "*" / "*" / "quotes.parquet")
    rb_sec = int(args.rebalance_min * 60)

    print("[pred]", pred_path)
    print("[quotes]", quote_glob)
    print("[rebalance_sec]", rb_sec)

    pred_schema = pl.scan_parquet(str(pred_path)).collect_schema()
    if args.score_col in pred_schema:
        score_col = args.score_col
    elif "pred_z" in pred_schema:
        score_col = "pred_z"
    else:
        score_col = "pred"

    pred_lf = (
        pl.scan_parquet(str(pred_path))
        .with_columns([
            pl.col("date").cast(pl.Int64),
            pl.col("sid").cast(pl.Int64),
            pl.col("ts").cast(pl.Int64),
        ])
        .filter(pl.col("ts") % rb_sec == 0)
        .select([
            "date",
            "sid",
            "ts",
            pl.col(score_col).cast(pl.Float64).alias("score"),
            pl.col("y_raw").cast(pl.Float64).alias("label_y_raw"),
        ])
    )

    quotes_lf = (
        pl.scan_parquet(quote_glob)
        .with_columns([
            pl.col("date").cast(pl.Int64),
            pl.col("sid").cast(pl.Int64),
            pl.col("ts").cast(pl.Int64),
        ])
        .filter(pl.col("ts") % rb_sec == 0)
        .select([
            "date", "sid", "ts", "ts_real",
            "task", "tbid", "tmid", "tavol", "tbvol", "vol",
        ])
    )

    joined = (
        pred_lf
        .join(quotes_lf, on=KEY, how="inner")
        .with_columns([
            pl.when(pl.col("task") >= pl.col("tbid"))
              .then(pl.col("task"))
              .otherwise(pl.col("tbid"))
              .alias("exec_buy_raw"),

            pl.when(pl.col("task") <= pl.col("tbid"))
              .then(pl.col("task"))
              .otherwise(pl.col("tbid"))
              .alias("exec_sell_raw"),
        ])
        .with_columns([
            ((pl.col("exec_buy_raw") + pl.col("exec_sell_raw")) / 2.0).alias("exec_mid_raw"),
        ])
        .with_columns([
            (
                (pl.col("exec_buy_raw") - pl.col("exec_sell_raw"))
                / pl.col("exec_mid_raw")
                * 10000.0
            ).alias("exec_spread_bps"),
        ])
        .with_columns([
            (pl.col("exec_mid_raw") * float(args.price_multiplier)).alias("mid_price"),
            (pl.col("exec_sell_raw") * float(args.price_multiplier)).alias("bid_price"),
            (pl.col("exec_buy_raw") * float(args.price_multiplier)).alias("ask_price"),
            (
                pl.col("vol").fill_null(0.0)
                * pl.col("exec_mid_raw")
                * float(args.price_multiplier)
                * float(args.participation_rate)
                * float(args.capacity_cap_scale)
                / float(args.capital)
            ).alias("capacity_weight"),
        ])
        .filter(
            pl.col("exec_buy_raw").is_not_null()
            & pl.col("exec_sell_raw").is_not_null()
            & pl.col("exec_mid_raw").is_not_null()
            & (pl.col("exec_buy_raw") > 0)
            & (pl.col("exec_sell_raw") > 0)
            & (pl.col("exec_mid_raw") > 0)
            & (pl.col("mid_price") > 0)
            & (pl.col("bid_price") > 0)
            & (pl.col("ask_price") > 0)
        )
        .select([
            "date", "sid", "ts", "ts_real",
            "score", "label_y_raw",
            "mid_price", "bid_price", "ask_price",
            "exec_spread_bps", "vol", "capacity_weight",
        ])
    )

    print("[collecting joined data]")
    df = joined.collect(streaming=True).to_pandas()

    if df.empty:
        raise RuntimeError("joined data is empty")

    df = df.sort_values(["date", "ts", "sid"]).reset_index(drop=True)

    df["execution_date"] = df["date"].astype("int64")
    df["execution_datetime"] = make_execution_datetime(df)
    df["minute"] = (
        pd.to_datetime(df["execution_datetime"])
        .dt.strftime("%H%M%S")
        .astype("int64")
    )
    df["securityid"] = df["sid"].astype("int64").astype(str).str.zfill(6)

    df["spread_bps"] = df["exec_spread_bps"]
    df["spread_bps_realized"] = df["exec_spread_bps"]

    df["capacity_weight"] = (
        df["capacity_weight"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .clip(lower=0.0)
    )

    df["valid_rank"] = (
        df["score"].replace([np.inf, -np.inf], np.nan).notna()
        & (df["spread_bps"] <= float(args.max_spread_bps))
        & (df["vol"].fillna(0.0) > 0)
    )

    print("[joined rows]", len(df))
    print("[dates]", df["date"].min(), "->", df["date"].max(), "n=", df["date"].nunique())
    print("[snapshots]", df[["date", "ts"]].drop_duplicates().shape[0])
    print("[symbols]", df["securityid"].nunique())
    print("[valid rank ratio]", float(df["valid_rank"].mean()))

    return df


def build_stateful_targets(df: pd.DataFrame, args) -> pd.DataFrame:
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

    # position state is intraday only
    for d, day_idx in df.groupby("date", sort=True).indices.items():
        day_idx = np.asarray(day_idx)
        day = df.iloc[day_idx]
        ts_list = sorted(day["ts"].unique().tolist())

        flat_ts = set()
        if args.force_flat_eod:
            flat_ts = set(ts_list[-int(args.flat_last_n_rebalances):])

        held = []

        for ts in ts_list:
            idx = day_idx[day["ts"].to_numpy() == ts]
            g = df.iloc[idx]

            if ts in flat_ts:
                held = []
                continue

            valid = g[g["rank"].notna()].copy()
            if valid.empty:
                held = []
                continue

            valid = valid.sort_values("rank")
            rank_by_sid = valid.set_index("securityid")["rank"].to_dict()

            # 1. keep old names if still inside exit band
            kept = [
                sid for sid in held
                if sid in rank_by_sid and rank_by_sid[sid] <= int(args.exit_rank)
            ]

            # 2. if too many kept names, keep the best-ranked ones
            kept = sorted(kept, key=lambda x: rank_by_sid.get(x, 1e18))
            kept = kept[:int(args.target_n)]

            # 3. add new names from entry band
            chosen = list(kept)
            entry_candidates = valid.loc[
                valid["rank"] <= int(args.entry_rank), "securityid"
            ].tolist()

            chosen_set = set(chosen)
            for sid in entry_candidates:
                if sid not in chosen_set:
                    chosen.append(sid)
                    chosen_set.add(sid)
                if len(chosen) >= int(args.target_n):
                    break

            # 4. optional fill with best available names if not enough
            if args.fill_to_target and len(chosen) < int(args.target_n):
                for sid in valid["securityid"].tolist():
                    if sid not in chosen_set:
                        chosen.append(sid)
                        chosen_set.add(sid)
                    if len(chosen) >= int(args.target_n):
                        break

            chosen = chosen[:int(args.target_n)]

            if not chosen:
                held = []
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

            weights = allocate_equal_with_caps(
                chosen_sids=chosen,
                caps_by_sid=caps,
                target_gross=float(args.target_gross),
            )

            if weights.empty:
                held = []
                continue

            row_map = valid.reset_index().set_index("securityid")["index"].to_dict()

            for sid, w in weights.items():
                ridx = int(row_map[sid])
                target[ridx] = float(w)
                selected[ridx] = True

            held = list(weights.index)

    df["effective_target_weight"] = target
    df["target_weight"] = target
    df["desired_target_weight"] = target
    df["gross_weight"] = target
    df["selected"] = selected

    df["state"] = np.where(df["selected"], "PURE_CS_STATEFUL_TOPN", "FLAT")
    df["side"] = np.where(df["selected"], "BUY", "NONE")
    df["blocked_reason"] = "none"

    return df


def print_target_diagnostics(df: pd.DataFrame):
    g = (
        df.groupby(["execution_date", "execution_datetime"])
        .agg(
            gross=("effective_target_weight", "sum"),
            n_hold=("selected", "sum"),
            avg_selected_spread_bps=("spread_bps", lambda x: np.nan),
        )
        .reset_index()
    )

    sel = df[df["selected"]].copy()
    if not sel.empty:
        spread_by_t = (
            sel.groupby(["execution_date", "execution_datetime"])["spread_bps"]
            .mean()
            .rename("avg_selected_spread_bps")
            .reset_index()
        )
        g = g.drop(columns=["avg_selected_spread_bps"]).merge(
            spread_by_t,
            on=["execution_date", "execution_datetime"],
            how="left",
        )

    print("\n===== target gross describe =====")
    print(g["gross"].describe())

    print("\n===== target n_hold describe =====")
    print(g["n_hold"].describe())

    if "avg_selected_spread_bps" in g.columns:
        print("\n===== selected spread bps describe =====")
        print(g["avg_selected_spread_bps"].describe())

    # target turnover proxy
    turns = []
    day_turns = []

    for d, gd in df.groupby("execution_date", sort=True):
        prev = {}
        day_turn = 0.0

        for dt, gt in gd.groupby("execution_datetime", sort=True):
            cur = gt.loc[
                gt["effective_target_weight"] > 0,
                ["securityid", "effective_target_weight"]
            ]
            cur_dict = dict(zip(cur["securityid"], cur["effective_target_weight"]))

            names = set(prev) | set(cur_dict)
            turn = sum(abs(cur_dict.get(x, 0.0) - prev.get(x, 0.0)) for x in names)

            turns.append(turn)
            day_turn += turn
            prev = cur_dict

        day_turns.append(day_turn)

    print("\n===== target turnover proxy =====")
    print("avg_turnover_per_rebalance:", float(np.mean(turns)) if turns else np.nan)
    print("avg_turnover_per_day:", float(np.mean(day_turns)) if day_turns else np.nan)


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

    df = load_joined_data(args)
    df = build_stateful_targets(df, args)

    out_cols = [
        "execution_date",
        "execution_datetime",
        "date",
        "minute",
        "securityid",

        "mid_price",
        "bid_price",
        "ask_price",
        "spread_bps",
        "spread_bps_realized",

        "effective_target_weight",
        "target_weight",
        "desired_target_weight",
        "gross_weight",

        "selected",
        "state",
        "side",
        "blocked_reason",

        "score",
        "rank",
        "label_y_raw",
        "capacity_weight",
        "vol",
    ]

    df[out_cols].sort_values(
        ["execution_date", "execution_datetime", "securityid"]
    ).to_csv(out, index=False)

    print("\n[saved]", out)
    print("[rows]", len(df))
    print("[selected rows]", int(df["selected"].sum()))

    print_target_diagnostics(df)


if __name__ == "__main__":
    main()
