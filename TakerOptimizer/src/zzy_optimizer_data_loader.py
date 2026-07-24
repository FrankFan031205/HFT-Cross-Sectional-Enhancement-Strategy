# -*- coding: utf-8 -*-
"""
zzy_optimizer_data_loader.py

Adapter for ZZY optimizer data in /mnt/data1/zzy/optimizer_data.
It loads prediction parquet files and future-60s execution quotes, aligns them by
(date, sid, ts), and converts the master table into a TakerOptimizer-friendly frame.

Important: quote columns task/tbid/tmid/tavol/tbvol/vol are future execution statistics.
They are valid for offline execution simulation / cost / liquidity constraints, but should
not be used as alpha features in live decision logic.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable, Optional, Sequence, Tuple

import polars as pl

PRED_ROOT = os.environ.get("ZZY_PRED_ROOT", "/mnt/data1/zzy/optimizer_data/pred")
PRED_TS_ROOT = os.environ.get("ZZY_PRED_TS_ROOT", "/mnt/data1/zzy/optimizer_data/pred_ts")
QUOTES_ROOT = os.environ.get("ZZY_QUOTES_ROOT", "/mnt/data1/zzy/optimizer_data/quotes")

_KEY = ["date", "sid", "ts"]
DEFAULT_HORIZONS = (2, 3, 5, 10, 20, 30)


def _as_int_list(xs: Optional[Iterable[int]]) -> Optional[list[int]]:
    if xs is None:
        return None
    return [int(x) for x in xs]


def _avail_horizons(root: str) -> list[int]:
    """Return horizons under <root>/<h>min/test_predictions.parquet."""
    if not os.path.isdir(root):
        return []
    out: list[int] = []
    for name in os.listdir(root):
        if not name.endswith("min"):
            continue
        try:
            h = int(name[:-3])
        except ValueError:
            continue
        if os.path.exists(os.path.join(root, name, "test_predictions.parquet")):
            out.append(h)
    return sorted(out)


def _cast_keys(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.col("date").cast(pl.Int64),
        pl.col("sid").cast(pl.Int64),
        pl.col("ts").cast(pl.Int64),
    )


def _add_security_id(df: pl.DataFrame) -> pl.DataFrame:
    """Add six-digit A-share code. Keep sid as int for joins."""
    try:
        return df.with_columns(pl.col("sid").cast(pl.Utf8).str.zfill(6).alias("SecurityID"))
    except Exception:
        # Compatibility fallback for very old polars. Slower, but robust.
        return df.with_columns(
            pl.col("sid").map_elements(lambda x: f"{int(x):06d}", return_dtype=pl.Utf8).alias("SecurityID")
        )


# ============================== quotes ==============================
def load_quotes(
    dates: Optional[Sequence[int]] = None,
    quotes_root: str = QUOTES_ROOT,
    n_workers: int = 48,
) -> pl.DataFrame:
    """
    Multi-thread read quotes/TimeSeries/<date>/<sid>/quotes.parquet into one long table.

    Columns expected:
      date,sid,ts,ts_real,task,tbid,tmid,tavol,tbvol,vol
    """
    ts_root = os.path.join(quotes_root, "TimeSeries")
    if dates is None:
        dates = sorted(int(d) for d in os.listdir(ts_root) if d.isdigit())
    dates = _as_int_list(dates)

    tasks: list[str] = []
    for d in dates or []:
        day_dir = os.path.join(ts_root, str(d))
        if not os.path.isdir(day_dir):
            continue
        for s in os.listdir(day_dir):
            fp = os.path.join(day_dir, s, "quotes.parquet")
            if os.path.exists(fp):
                tasks.append(fp)

    if not tasks:
        raise SystemExit(f"!! Cannot read quotes from {quotes_root}; dates={dates}")

    with ThreadPoolExecutor(max_workers=int(n_workers)) as ex:
        parts = list(ex.map(pl.read_parquet, tasks))
    return _cast_keys(pl.concat(parts, how="vertical"))


# ============================== predictions ==============================
def load_pred(
    horizon: int,
    model: str = "ts",
    dates: Optional[Sequence[int]] = None,
    pred_root: str = PRED_ROOT,
    pred_ts_root: str = PRED_TS_ROOT,
) -> pl.DataFrame:
    """
    Read one horizon/model prediction table.

    model='ts'  -> pred_ts root, label y_raw
    model='res' -> residual root, label cs_zscore
    Raw columns expected: date, ts, sid, pred, pred_z, y_raw
    """
    if model not in ("ts", "res"):
        raise ValueError("model must be 'ts' or 'res'")
    root = pred_ts_root if model == "ts" else pred_root
    fp = os.path.join(root, f"{int(horizon)}min", "test_predictions.parquet")
    if not os.path.exists(fp):
        raise SystemExit(f"!! Missing prediction file: {fp}")
    df = _cast_keys(pl.read_parquet(fp))
    if dates is not None:
        df = df.filter(pl.col("date").is_in(_as_int_list(dates)))
    return df.sort(_KEY)


def load_preds(
    dates: Optional[Sequence[int]] = None,
    horizons: Optional[Sequence[int]] = None,
    models: Sequence[str] = ("ts", "res"),
    pred_root: str = PRED_ROOT,
    pred_ts_root: str = PRED_TS_ROOT,
) -> pl.DataFrame:
    """
    Load all selected predictions into a wide prediction-only table keyed by (date,sid,ts).

    Output columns:
      date,sid,ts,
      pred_ts_<h>, predz_ts_<h>, y_<h>,
      pred_res_<h>
    """
    models = tuple(models)
    bad = set(models) - {"ts", "res"}
    if bad:
        raise ValueError(f"unknown models: {sorted(bad)}")

    if horizons is None:
        ts_hs = _avail_horizons(pred_ts_root)
        res_hs = _avail_horizons(pred_root)
    else:
        hs = [int(h) for h in horizons]
        ts_hs = hs
        res_hs = hs

    base: Optional[pl.DataFrame] = None

    def _take(fp: str, renames: Sequence[Tuple[str, str]]) -> pl.DataFrame:
        d = _cast_keys(pl.read_parquet(fp))
        if dates is not None:
            d = d.filter(pl.col("date").is_in(_as_int_list(dates)))
        return d.select(_KEY + [pl.col(src).alias(dst) for src, dst in renames])

    if "ts" in models:
        for h in ts_hs:
            fp = os.path.join(pred_ts_root, f"{int(h)}min", "test_predictions.parquet")
            if os.path.exists(fp):
                d = _take(fp, [("pred", f"pred_ts_{h}"), ("pred_z", f"predz_ts_{h}"), ("y_raw", f"y_{h}")])
                base = d if base is None else base.join(d, on=_KEY, how="outer_coalesce")

    if "res" in models:
        for h in res_hs:
            fp = os.path.join(pred_root, f"{int(h)}min", "test_predictions.parquet")
            if os.path.exists(fp):
                d = _take(fp, [("pred", f"pred_res_{h}")])
                base = d if base is None else base.join(d, on=_KEY, how="outer_coalesce")

    if base is None:
        raise SystemExit("!! No predictions loaded.")
    return base


# ============================== master table ==============================
def load_master(
    dates: Optional[Sequence[int]] = None,
    horizons: Optional[Sequence[int]] = None,
    models: Sequence[str] = ("ts", "res"),
    n_workers: int = 48,
    pred_root: str = PRED_ROOT,
    pred_ts_root: str = PRED_TS_ROOT,
    quotes_root: str = QUOTES_ROOT,
) -> pl.DataFrame:
    """Load quotes and predictions, left-join predictions onto quote grid by (date,sid,ts)."""
    q = load_quotes(dates=dates, quotes_root=quotes_root, n_workers=n_workers)
    p = load_preds(dates=dates, horizons=horizons, models=models, pred_root=pred_root, pred_ts_root=pred_ts_root)
    return q.join(p, on=_KEY, how="left").sort(_KEY)


def build_optimizer_frame(
    master: pl.DataFrame,
    horizon: int = 5,
    signal_model: str = "res",
    participation_rate: float = 0.03,
    min_vol_60s: float = 0.0,
    require_executable: bool = True,
    keep_all_preds: bool = False,
) -> pl.DataFrame:
    """
    Convert master table into a generic optimizer input frame.

    signal_model:
      'res' -> use pred_res_<h>, compute cross-sectional zscore by (date,ts)
      'ts'  -> use pred_ts_<h>, use predz_ts_<h> if present, else compute zscore

    Execution columns:
      raw_task/raw_tbid/raw_tmid keep the original quote columns for audit.
      exec_buy_price  = max(task, tbid)  (conservative taker buy price)
      exec_sell_price = min(task, tbid)  (conservative taker sell price)
      exec_mid_price  = (exec_buy_price + exec_sell_price) / 2
      exec_spread_bps = non-negative spread based on corrected buy/sell prices
    """
    horizon = int(horizon)
    if signal_model not in ("ts", "res"):
        raise ValueError("signal_model must be 'ts' or 'res'")

    signal_col = f"pred_{signal_model}_{horizon}"
    if signal_col not in master.columns:
        raise ValueError(f"missing signal column: {signal_col}; available={master.columns}")

    y_col = f"y_{horizon}" if f"y_{horizon}" in master.columns else None
    ts_z_col = f"predz_ts_{horizon}"

    # Base filters: no prediction null; no execution null if requested.
    filters = [pl.col(signal_col).is_not_null()]
    if require_executable:
        filters += [
            pl.col("task").is_not_null(),
            pl.col("tbid").is_not_null(),
            pl.col("tmid").is_not_null(),
            (pl.col("tmid") > 0),
        ]
    if min_vol_60s > 0:
        filters.append(pl.col("vol") >= float(min_vol_60s))

    df = master.filter(pl.all_horizontal(filters))

    # Signal z-score. For residual predictions the loader keeps only pred_res_<h>, so recompute.
    if signal_model == "ts" and ts_z_col in df.columns:
        signal_z_expr = pl.col(ts_z_col).alias("signal_z")
    else:
        signal_z_expr = (
            (pl.col(signal_col) - pl.col(signal_col).mean().over(["date", "ts"]))
            / (pl.col(signal_col).std().over(["date", "ts"]) + 1e-12)
        ).alias("signal_z")

    # Some dumped quote files may have task/tbid crossed or swapped on certain dates.
    # For taker execution cost, use a conservative side-aware correction:
    # buy pays the higher of the two, sell receives the lower of the two.
    buy_px = pl.when(pl.col("task") >= pl.col("tbid")).then(pl.col("task")).otherwise(pl.col("tbid"))
    sell_px = pl.when(pl.col("task") >= pl.col("tbid")).then(pl.col("tbid")).otherwise(pl.col("task"))
    mid_px = (buy_px + sell_px) / 2.0
    spread_rel = (buy_px - sell_px) / mid_px

    base_cols = [
        "date", "sid", "SecurityID", "ts", "ts_real",
        "horizon_min", "signal_model", "signal_col",
        "pred_ret", "signal_raw", "signal_z",
        "raw_task", "raw_tbid", "raw_tmid", "raw_crossed_quote", "raw_spread_bps",
        "exec_buy_price", "exec_sell_price", "exec_mid_price",
        "exec_spread_rel", "exec_spread_bps", "buy_cost_bps", "sell_cost_bps",
        "ask_depth_60s", "bid_depth_60s", "volume_60s",
        "max_participation_shares", "max_participation_notional",
    ]
    if y_col is not None:
        base_cols.append("fwd_ret_label")

    exprs = [
        pl.lit(horizon).alias("horizon_min"),
        pl.lit(signal_model).alias("signal_model"),
        pl.lit(signal_col).alias("signal_col"),
        pl.col(signal_col).alias("pred_ret"),
        pl.col(signal_col).alias("signal_raw"),
        signal_z_expr,
        pl.col("task").alias("raw_task"),
        pl.col("tbid").alias("raw_tbid"),
        pl.col("tmid").alias("raw_tmid"),
        (pl.col("task") < pl.col("tbid")).alias("raw_crossed_quote"),
        (((pl.col("task") - pl.col("tbid")) / pl.col("tmid")) * 10000.0).alias("raw_spread_bps"),
        buy_px.alias("exec_buy_price"),
        sell_px.alias("exec_sell_price"),
        mid_px.alias("exec_mid_price"),
        spread_rel.alias("exec_spread_rel"),
        (spread_rel * 10000.0).alias("exec_spread_bps"),
        ((buy_px / mid_px - 1.0) * 10000.0).alias("buy_cost_bps"),
        ((1.0 - sell_px / mid_px) * 10000.0).alias("sell_cost_bps"),
        pl.col("tavol").alias("ask_depth_60s"),
        pl.col("tbvol").alias("bid_depth_60s"),
        pl.col("vol").alias("volume_60s"),
        (pl.col("vol") * float(participation_rate)).alias("max_participation_shares"),
        (pl.col("vol") * float(participation_rate) * mid_px).alias("max_participation_notional"),
    ]
    if y_col is not None:
        exprs.append(pl.col(y_col).alias("fwd_ret_label"))

    df = _add_security_id(df).with_columns(exprs)

    if keep_all_preds:
        pred_cols = [c for c in df.columns if c.startswith(("pred_ts_", "predz_ts_", "pred_res_", "y_"))]
        selected = base_cols + [c for c in pred_cols if c not in base_cols]
    else:
        selected = base_cols

    return df.select(selected).sort(["date", "ts", "sid"])
