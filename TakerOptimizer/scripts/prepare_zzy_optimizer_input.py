#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare per-day optimizer input from ZZY prediction + future quote data.

Example:
  cd /mnt/data1/fwz/HFT_010-dev_fwz
  python TakerOptimizer/scripts/prepare_zzy_optimizer_input.py \
    --dates 20241217 20241218 \
    --horizon 5 --signal-model res --workers 32 \
    --out-dir /mnt/data1/fwz/HFT_010-dev_fwz_outputs/TakerOptimizer/zzy_zz2000_h5_res_v1
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from zzy_optimizer_data_loader import (  # noqa: E402
    DEFAULT_HORIZONS,
    PRED_ROOT,
    PRED_TS_ROOT,
    QUOTES_ROOT,
    build_optimizer_frame,
    load_master,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dates", type=int, nargs="+", required=True, help="Trading dates, e.g. 20241217 20241218")
    ap.add_argument("--horizon", type=int, default=5, choices=list(DEFAULT_HORIZONS), help="Signal horizon in minutes")
    ap.add_argument("--signal-model", choices=["res", "ts"], default="res", help="Use residual or time-series prediction")
    ap.add_argument("--load-horizons", type=int, nargs="*", default=None, help="Horizons to load; default = selected horizon")
    ap.add_argument("--include-y", action="store_true", help="Also load ts y_<h> label for diagnostics")
    ap.add_argument("--keep-all-preds", action="store_true", help="Keep loaded raw pred columns in output")
    ap.add_argument("--participation", type=float, default=0.03, help="Volume participation cap applied to future 60s vol")
    ap.add_argument("--min-vol-60s", type=float, default=0.0, help="Drop rows with vol below this threshold")
    ap.add_argument("--max-spread-bps", type=float, default=200.0, help="Drop rows with corrected exec spread above this threshold; <=0 disables")
    ap.add_argument("--allow-non-executable", action="store_true", help="Do not drop rows with null task/tbid/tmid")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--pred-root", default=PRED_ROOT)
    ap.add_argument("--pred-ts-root", default=PRED_TS_ROOT)
    ap.add_argument("--quotes-root", default=QUOTES_ROOT)
    ap.add_argument(
        "--out-dir",
        default=None,
        help="Output directory. Default under /mnt/data1/fwz/HFT_010-dev_fwz_outputs/TakerOptimizer/",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    dates = sorted(set(int(d) for d in args.dates))

    if args.out_dir is None:
        tag = f"zzy_zz2000_h{args.horizon}_{args.signal_model}_v1"
        out_dir = Path(f"/mnt/data1/fwz/HFT_010-dev_fwz_outputs/TakerOptimizer/{tag}")
    else:
        out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load only needed models/horizons unless diagnostics are requested.
    models = {args.signal_model}
    if args.include_y or args.keep_all_preds:
        models.add("ts")
    load_horizons = args.load_horizons
    if load_horizons is None:
        load_horizons = list(DEFAULT_HORIZONS) if args.keep_all_preds else [args.horizon]

    manifest = {
        "dates": dates,
        "horizon": args.horizon,
        "signal_model": args.signal_model,
        "models_loaded": sorted(models),
        "load_horizons": load_horizons,
        "participation": args.participation,
        "min_vol_60s": args.min_vol_60s,
        "max_spread_bps": args.max_spread_bps,
        "require_executable": not args.allow_non_executable,
        "pred_root": args.pred_root,
        "pred_ts_root": args.pred_ts_root,
        "quotes_root": args.quotes_root,
        "out_dir": str(out_dir),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    rows = []
    for d in dates:
        print(f"[load] date={d} horizons={load_horizons} models={sorted(models)}", flush=True)
        master = load_master(
            dates=[d],
            horizons=load_horizons,
            models=tuple(sorted(models)),
            n_workers=args.workers,
            pred_root=args.pred_root,
            pred_ts_root=args.pred_ts_root,
            quotes_root=args.quotes_root,
        )
        print(f"[master] date={d} rows={master.height:,} cols={len(master.columns)}", flush=True)

        opt = build_optimizer_frame(
            master,
            horizon=args.horizon,
            signal_model=args.signal_model,
            participation_rate=args.participation,
            min_vol_60s=args.min_vol_60s,
            require_executable=not args.allow_non_executable,
            keep_all_preds=args.keep_all_preds,
        )
        before_filter_rows = opt.height
        valid_exec = (
            pl.col("exec_buy_price").is_not_null()
            & pl.col("exec_sell_price").is_not_null()
            & pl.col("exec_mid_price").is_not_null()
            & pl.col("exec_spread_bps").is_not_null()
            & (pl.col("exec_buy_price") > 0)
            & (pl.col("exec_sell_price") > 0)
            & (pl.col("exec_mid_price") > 0)
            & (pl.col("volume_60s") > 0)
            & (pl.col("exec_spread_bps") >= 0)
        )
        if args.max_spread_bps and args.max_spread_bps > 0:
            valid_exec = valid_exec & (pl.col("exec_spread_bps") <= float(args.max_spread_bps))

        opt = opt.filter(valid_exec)
        print(
            f"[filter] date={d} before={before_filter_rows:,} after={opt.height:,} "
            f"dropped={before_filter_rows - opt.height:,} max_spread_bps={args.max_spread_bps}",
            flush=True,
        )

        out_fp = out_dir / f"optimizer_input_{d}.parquet"
        opt.write_parquet(out_fp, compression="zstd")

        stat = opt.select(
            pl.len().alias("rows"),
            pl.col("sid").n_unique().alias("num_sids"),
            pl.col("ts").n_unique().alias("num_ts"),
            pl.col("signal_z").mean().alias("signal_z_mean"),
            pl.col("signal_z").std().alias("signal_z_std"),
            pl.col("exec_spread_bps").mean().alias("avg_spread_bps"),
            pl.col("volume_60s").mean().alias("avg_vol_60s"),
            pl.col("max_participation_notional").mean().alias("avg_cap_notional"),
        ).to_dicts()[0]
        stat = {"date": d, "file": str(out_fp), **stat}
        rows.append(stat)
        print(f"[saved] {out_fp} rows={stat['rows']:,} sids={stat['num_sids']}", flush=True)

        # Free memory explicitly before the next day.
        del master, opt

    summary_fp = out_dir / "daily_summary.csv"
    with summary_fp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[done] out_dir={out_dir}")
    print(f"[summary] {summary_fp}")


if __name__ == "__main__":
    main()
