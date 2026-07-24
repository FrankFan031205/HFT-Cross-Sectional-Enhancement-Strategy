# -*- coding: utf-8 -*-
"""
Pure CS Full-Invested Active Optimizer for ZZY ZZ2000.

Purpose:
  Build a standard full-invested pure cross-sectional portfolio.

Design:
  - 100% gross / 100% net long
  - no market timing
  - no derisk
  - no up-ratio / market regime
  - benchmark = equal weight over current tradable universe
  - active tilt from cross-sectional signal rank
  - portfolio weight = benchmark_weight + active_weight
  - active weights sum to 0 before long-only/cap projection
  - output positions directly usable by TakerModel v8

Main case:
  res h20 + 20min rebalance + active_l1 0.30
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


def project_long_only_capped(raw_w, caps, target_gross=1.0, max_iter=100):
    """
    Project raw long-only weights to:
      0 <= w_i <= cap_i
      sum_i w_i = target_gross, if total caps allow.

    This is a simple water-filling style projection.
    """
    raw_w = np.asarray(raw_w, dtype=float)
    caps = np.asarray(caps, dtype=float)

    raw_w = np.nan_to_num(raw_w, nan=0.0, posinf=0.0, neginf=0.0)
    caps = np.nan_to_num(caps, nan=0.0, posinf=0.0, neginf=0.0)

    caps = np.maximum(caps, 0.0)
    if len(raw_w) == 0 or caps.sum() <= 1e-12 or target_gross <= 0:
        return np.zeros_like(raw_w)

    target = min(float(target_gross), float(caps.sum()))

    # Start from clipped positive raw weights.
    w = np.clip(raw_w, 0.0, caps)

    # If all clipped to zero, start from equal weights under caps.
    if w.sum() <= 1e-12:
        w = np.minimum(np.ones_like(w) * target / len(w), caps)

    # If above target, scale down.
    if w.sum() > target:
        w = w * (target / w.sum())
        w = np.minimum(w, caps)

    # Redistribute remaining mass to names with room, proportional to room.
    for _ in range(max_iter):
        rem = target - w.sum()
        if rem <= 1e-12:
            break

        room = caps - w
        mask = room > 1e-12
        if not mask.any():
            break

        add = rem * room[mask] / room[mask].sum()
        add = np.minimum(add, room[mask])
        w[mask] += add

    # Numerical cleanup.
    if w.sum() > target + 1e-10:
        w *= target / w.sum()

    return w


def build_active_weights_for_snapshot(g: pd.DataFrame, args):
    """
    Build full-invested active weights for one cross-section.

    benchmark_weight: equal weight over current valid universe.
    active_weight: rank-linear demeaned tilt.
    active_l1 means sum(abs(active_weight)) ~= active_l1 * target_gross
               before long-only/cap projection.
    """
    n = len(g)
    if n == 0:
        return pd.DataFrame()

    target_gross = float(args.target_gross)
    active_l1 = float(args.active_l1)

    score = g["score"].to_numpy(dtype=float)

    # rank 1 = highest score
    rank = pd.Series(score).rank(method="first", ascending=False).to_numpy()
    rank_pct = (rank - 0.5) / n

    # positive for top names, negative for bottom names
    tilt_raw = 0.5 - rank_pct
    tilt_raw = tilt_raw - tilt_raw.mean()

    benchmark_w = np.ones(n, dtype=float) * target_gross / n

    denom = np.abs(tilt_raw).sum()
    if denom <= 1e-12 or active_l1 <= 0:
        active_w_raw = np.zeros(n, dtype=float)
    else:
        active_w_raw = tilt_raw / denom * (active_l1 * target_gross)

    raw_w = benchmark_w + active_w_raw

    if int(args.use_capacity_cap) == 1:
        caps = np.minimum(
            float(args.single_name_cap),
            g["capacity_weight"].to_numpy(dtype=float),
        )
    else:
        caps = np.ones(n, dtype=float) * float(args.single_name_cap)

    final_w = project_long_only_capped(
        raw_w=raw_w,
        caps=caps,
        target_gross=target_gross,
    )

    out = pd.DataFrame({
        "securityid": g["securityid"].to_numpy(),
        "rank": rank,
        "benchmark_weight": benchmark_w,
        "active_weight_raw": active_w_raw,
        "raw_target_weight": raw_w,
        "effective_target_weight": final_w,
    })

    out["active_weight_final"] = out["effective_target_weight"] - out["benchmark_weight"]
    return out


def load_joined_data(args) -> pd.DataFrame:
    data_root = Path(args.data_root)
    pred_path = get_pred_path(data_root, args.model, args.horizon_min)

    if not pred_path.exists():
        raise FileNotFoundError(f"prediction file not found: {pred_path}")

    quote_root = data_root / "quotes" / "TimeSeries"
    quote_files_all = sorted(quote_root.glob("*/*/quotes.parquet"))

    quote_files = []
    bad_quote_files = []

    for p in quote_files_all:
        try:
            size = p.stat().st_size
            if size < 8:
                bad_quote_files.append(str(p))
                continue
            with open(p, "rb") as f:
                head = f.read(4)
                f.seek(-4, 2)
                tail = f.read(4)
            if head == b"PAR1" and tail == b"PAR1":
                quote_files.append(str(p))
            else:
                bad_quote_files.append(str(p))
        except Exception:
            bad_quote_files.append(str(p))

    if not quote_files:
        raise RuntimeError(f"no valid quote parquet files under {quote_root}")

    rb_sec = int(args.rebalance_min * 60)

    print("[pred]", pred_path)
    print("[quotes root]", quote_root)
    print("[valid quote files]", len(quote_files))
    print("[bad quote files]", len(bad_quote_files))
    if bad_quote_files:
        print("[bad quote examples]")
        for x in bad_quote_files[:20]:
            print("  ", x)
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
        pl.scan_parquet(quote_files)
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
            pl.col("score").is_not_null()
            & pl.col("score").is_finite()
            & pl.col("label_y_raw").is_not_null()
            & pl.col("exec_buy_raw").is_not_null()
            & pl.col("exec_sell_raw").is_not_null()
            & pl.col("exec_mid_raw").is_not_null()
            & (pl.col("exec_buy_raw") > 0)
            & (pl.col("exec_sell_raw") > 0)
            & (pl.col("exec_mid_raw") > 0)
            & (pl.col("mid_price") > 0)
            & (pl.col("bid_price") > 0)
            & (pl.col("ask_price") > 0)
            & (pl.col("vol").fill_null(0.0) > 0)
            & (pl.col("exec_spread_bps") <= float(args.max_spread_bps))
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

    print("[joined rows]", len(df))
    print("[dates]", df["date"].min(), "->", df["date"].max(), "n=", df["date"].nunique())
    print("[snapshots]", df[["date", "ts"]].drop_duplicates().shape[0])
    print("[symbols]", df["securityid"].nunique())
    print("[avg universe per snapshot]", df.groupby(["date", "ts"])["securityid"].size().mean())

    return df


def build_fullinvested_targets(df: pd.DataFrame, args) -> pd.DataFrame:
    rows = []

    prev_w = None
    prev_sid_order = None
    last_date = None

    smooth = float(args.smooth)

    for (d, ts), g in df.groupby(["date", "ts"], sort=True):
        g = g.sort_values("score", ascending=False).copy()

        if int(args.reset_daily) == 1 and (last_date is None or d != last_date):
            prev_w = None
            prev_sid_order = None
            last_date = d

        wdf = build_active_weights_for_snapshot(g, args)

        if wdf.empty:
            continue

        g = g.merge(wdf, on="securityid", how="left")

        # Optional smoothing at target-weight level.
        # This reduces active turnover but keeps full investment after re-projection.
        if smooth < 1.0 and prev_w is not None:
            sid = g["securityid"].to_numpy()
            raw_new = g["effective_target_weight"].to_numpy(dtype=float)
            old = np.array([prev_w.get(x, 0.0) for x in sid], dtype=float)

            mixed_raw = (1.0 - smooth) * old + smooth * raw_new

            if int(args.use_capacity_cap) == 1:
                caps = np.minimum(
                    float(args.single_name_cap),
                    g["capacity_weight"].to_numpy(dtype=float),
                )
            else:
                caps = np.ones(len(g), dtype=float) * float(args.single_name_cap)

            mixed_final = project_long_only_capped(
                raw_w=mixed_raw,
                caps=caps,
                target_gross=float(args.target_gross),
            )

            g["raw_target_weight"] = mixed_raw
            g["effective_target_weight"] = mixed_final
            g["target_weight"] = mixed_final
            g["desired_target_weight"] = mixed_final
            g["active_weight_final"] = g["effective_target_weight"] - g["benchmark_weight"]
        else:
            g["target_weight"] = g["effective_target_weight"]
            g["desired_target_weight"] = g["effective_target_weight"]

        g["gross_weight"] = g["effective_target_weight"]
        g["selected"] = g["effective_target_weight"] > 0
        g["state"] = "PURE_CS_FULLINVESTED_ACTIVE"
        g["side"] = "BUY"
        g["blocked_reason"] = "none"

        prev_w = dict(zip(g["securityid"], g["effective_target_weight"]))

        rows.append(g)

    out = pd.concat(rows, ignore_index=True)
    return out


def print_target_diagnostics(df: pd.DataFrame):
    g = (
        df.groupby(["execution_date", "execution_datetime"])
        .agg(
            gross=("effective_target_weight", "sum"),
            n_hold=("selected", "sum"),
            active_l1_final=("active_weight_final", lambda x: np.abs(x).sum()),
            active_net_final=("active_weight_final", "sum"),
            max_weight=("effective_target_weight", "max"),
            avg_weight=("effective_target_weight", "mean"),
            avg_spread_bps=("spread_bps", "mean"),
        )
        .reset_index()
    )

    print("\n===== full-invested target diagnostics =====")
    print("snapshots:", len(g))
    print("\n[gross]")
    print(g["gross"].describe())
    print("\n[n_hold]")
    print(g["n_hold"].describe())
    print("\n[active_l1_final]")
    print(g["active_l1_final"].describe())
    print("\n[max_weight]")
    print(g["max_weight"].describe())
    print("\n[avg_spread_bps]")
    print(g["avg_spread_bps"].describe())

    # target turnover proxy
    turns = []
    day_turns = []
    active_turns = []
    day_active_turns = []

    for d, gd in df.groupby("execution_date", sort=True):
        prev = {}
        prev_active = {}
        day_turn = 0.0
        day_active_turn = 0.0

        for dt, gt in gd.groupby("execution_datetime", sort=True):
            cur = dict(zip(gt["securityid"], gt["effective_target_weight"]))
            cur_active = dict(zip(gt["securityid"], gt["active_weight_final"]))

            names = set(prev) | set(cur)
            turn = sum(abs(cur.get(x, 0.0) - prev.get(x, 0.0)) for x in names)

            active_names = set(prev_active) | set(cur_active)
            active_turn = sum(
                abs(cur_active.get(x, 0.0) - prev_active.get(x, 0.0))
                for x in active_names
            )

            turns.append(turn)
            active_turns.append(active_turn)
            day_turn += turn
            day_active_turn += active_turn

            prev = cur
            prev_active = cur_active

        day_turns.append(day_turn)
        day_active_turns.append(day_active_turn)

    print("\n===== target turnover proxy =====")
    print("avg_total_turnover_per_rebalance:", float(np.mean(turns)) if turns else np.nan)
    print("avg_total_turnover_per_day:", float(np.mean(day_turns)) if day_turns else np.nan)
    print("avg_active_turnover_per_rebalance:", float(np.mean(active_turns)) if active_turns else np.nan)
    print("avg_active_turnover_per_day:", float(np.mean(day_active_turns)) if day_active_turns else np.nan)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--data-root", default="/mnt/data1/zzy/optimizer_data")
    ap.add_argument("--model", choices=["res", "ts"], default="res")
    ap.add_argument("--horizon-min", type=int, required=True)
    ap.add_argument("--score-col", default="pred_z")
    ap.add_argument("--rebalance-min", type=int, required=True)

    ap.add_argument("--target-gross", type=float, default=1.0)
    ap.add_argument("--active-l1", type=float, default=0.30)
    ap.add_argument("--single-name-cap", type=float, default=0.003)
    ap.add_argument("--smooth", type=float, default=0.50)
    ap.add_argument("--reset-daily", type=int, default=1)

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
    df = build_fullinvested_targets(df, args)

    out_cols = [
        "execution_date", "execution_datetime", "date", "minute", "securityid",
        "mid_price", "bid_price", "ask_price",
        "spread_bps", "spread_bps_realized",

        "effective_target_weight", "target_weight", "desired_target_weight",
        "gross_weight",

        "benchmark_weight", "active_weight_raw", "active_weight_final", "raw_target_weight",

        "selected", "state", "side", "blocked_reason",

        "score", "rank", "label_y_raw", "capacity_weight", "vol",
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
