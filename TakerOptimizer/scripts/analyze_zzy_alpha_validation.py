# -*- coding: utf-8 -*-
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl


def safe_corr(x, y, rank=False):
    x = pd.Series(x)
    y = pd.Series(y)
    m = x.notna() & y.notna()
    if m.sum() < 100:
        return np.nan
    if rank:
        x = x[m].rank()
        y = y[m].rank()
    else:
        x = x[m]
        y = y[m]
    if x.std() <= 1e-12 or y.std() <= 1e-12:
        return np.nan
    return float(x.corr(y))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--optimizer-input-dir", required=True)
    ap.add_argument("--positions", required=True)
    ap.add_argument("--pricing-path", default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--rebalance-ts-step", type=int, default=60)
    ap.add_argument("--max-spread-bps", type=float, default=50.0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(Path(args.optimizer_input_dir).glob("optimizer_input_*.parquet"))
    if not files:
        raise SystemExit("no optimizer_input_*.parquet found")

    lf0 = pl.scan_parquet([str(x) for x in files])
    max_ts_real = lf0.select(pl.col("ts_real").cast(pl.Int64).max()).collect().item()
    div = 1000 if max_ts_real > 235959 else 1

    base = (
        lf0.filter(
            (pl.col("ts") % int(args.rebalance_ts_step) == 0)
            & (pl.col("exec_spread_bps") <= float(args.max_spread_bps))
            & (pl.col("exec_mid_price") > 0)
            & (pl.col("exec_buy_price") > 0)
            & (pl.col("exec_sell_price") > 0)
            & (pl.col("volume_60s") > 0)
            & pl.col("signal_z").is_not_null()
            & pl.col("fwd_ret_label").is_not_null()
        )
        .with_columns([
            (pl.col("ts_real").cast(pl.Int64) // div).cast(pl.Int64).alias("minute"),
            pl.col("SecurityID").cast(pl.Utf8).alias("securityid"),
            pl.col("signal_z").alias("signal_z"),
            pl.col("fwd_ret_label").alias("target_ret"),
            pl.col("exec_spread_bps").alias("spread_bps"),
        ])
        .select(["date", "minute", "securityid", "signal_z", "target_ret", "spread_bps"])
        .collect()
        .to_pandas()
    )

    base["securityid"] = base["securityid"].astype(str).str.zfill(6)
    base["date"] = base["date"].astype(int)
    base["minute"] = base["minute"].astype(int)

    # Optional calibrated PricingModel prediction
    pred_col = None
    if args.pricing_path:
        ph = pd.read_csv(args.pricing_path, nrows=0).columns.tolist()
        pred_cols = [c for c in ph if c.startswith("pred_ret_")]
        if pred_cols:
            pred_col = pred_cols[0]
            pricing = pd.read_csv(
                args.pricing_path,
                usecols=["date", "minute", "securityid", pred_col],
                low_memory=False,
            )
            pricing["securityid"] = pricing["securityid"].astype(str).str.zfill(6)
            pricing["date"] = pricing["date"].astype(int)
            pricing["minute"] = pricing["minute"].astype(int)
            base = base.merge(pricing, on=["date", "minute", "securityid"], how="left")
            print("[pricing pred col]", pred_col)

    print("[base]", base.shape)
    print("[date-minute]", base[["date", "minute"]].drop_duplicates().shape[0])

    # 1. Cross-sectional IC / RankIC by minute
    rows = []
    for (d, m), g in base.groupby(["date", "minute"], sort=True):
        row = {
            "date": d,
            "minute": m,
            "n": len(g),
            "signal_ic": safe_corr(g["signal_z"], g["target_ret"], rank=False),
            "signal_rank_ic": safe_corr(g["signal_z"], g["target_ret"], rank=True),
            "universe_ret_bps": g["target_ret"].mean() * 10000.0,
        }
        if pred_col:
            row["pm_ic"] = safe_corr(g[pred_col], g["target_ret"], rank=False)
            row["pm_rank_ic"] = safe_corr(g[pred_col], g["target_ret"], rank=True)
        rows.append(row)

    ic = pd.DataFrame(rows)
    ic.to_csv(out_dir / "alpha_ic_by_minute.csv", index=False)

    # 2. Decile monotonicity
    base["signal_rank_pct"] = base.groupby(["date", "minute"])["signal_z"].rank(pct=True)
    base["signal_decile"] = np.ceil(base["signal_rank_pct"] * 10).clip(1, 10).astype(int)

    decile = (
        base.groupby("signal_decile")
        .agg(
            n=("target_ret", "size"),
            avg_target_bps=("target_ret", lambda x: x.mean() * 10000.0),
            median_target_bps=("target_ret", lambda x: x.median() * 10000.0),
            avg_spread_bps=("spread_bps", "mean"),
        )
        .reset_index()
    )
    decile.to_csv(out_dir / "alpha_decile_summary.csv", index=False)

    top_ret = decile.loc[decile["signal_decile"] == 10, "avg_target_bps"].iloc[0]
    bot_ret = decile.loc[decile["signal_decile"] == 1, "avg_target_bps"].iloc[0]

    # 3. Selected portfolio alpha attribution
    pos = pd.read_csv(args.positions, low_memory=False)

    if "execution_date" in pos.columns and "date" not in pos.columns:
        pos["date"] = pos["execution_date"]
    if "effective_target_weight" in pos.columns:
        w_col = "effective_target_weight"
    elif "target_weight" in pos.columns:
        w_col = "target_weight"
    else:
        raise ValueError("cannot find effective_target_weight or target_weight")

    pos["date"] = pos["date"].astype(int)

    # Normalize minute key to HHMMSS integer.
    # Some TakerModel-adapted position files store minute as a timestamp string
    # such as "2024-12-17 09:30:00", while optimizer_input base uses HHMMSS.
    if "minute" in pos.columns:
        minute_num = pd.to_numeric(pos["minute"], errors="coerce")
    else:
        minute_num = pd.Series(np.nan, index=pos.index)

    if minute_num.notna().mean() > 0.95:
        pos["minute"] = minute_num.astype(int)
    else:
        dt_source = None
        if "minute" in pos.columns:
            dt_source = pd.to_datetime(pos["minute"], errors="coerce")
        if dt_source is None or dt_source.notna().mean() < 0.95:
            dt_source = pd.to_datetime(pos.get("execution_datetime"), errors="coerce")
        if dt_source is None or dt_source.notna().mean() < 0.95:
            dt_source = pd.to_datetime(pos.get("datetime"), errors="coerce")

        if dt_source is None or dt_source.notna().mean() < 0.95:
            raise ValueError("cannot normalize position minute from minute/execution_datetime/datetime")

        pos["minute"] = (
            dt_source.dt.hour * 10000
            + dt_source.dt.minute * 100
            + dt_source.dt.second
        ).astype(int)

    pos["securityid"] = pos["securityid"].astype(str).str.zfill(6)
    pos[w_col] = pd.to_numeric(pos[w_col], errors="coerce").fillna(0.0)

    pos_small = pos[["date", "minute", "securityid", w_col]].copy()
    pos_small = pos_small[pos_small[w_col] > 0]

    bench_min = (
        base.groupby(["date", "minute"])
        .agg(
            ew_target_ret=("target_ret", "mean"),
            n_universe=("target_ret", "size"),
        )
        .reset_index()
    )

    merged = pos_small.merge(
        base[["date", "minute", "securityid", "target_ret", "signal_z", "spread_bps"]],
        on=["date", "minute", "securityid"],
        how="left",
    )
    merged = merged.dropna(subset=["target_ret"])

    sel = (
        merged.groupby(["date", "minute"])
        .apply(lambda g: pd.Series({
            "net_weight": g[w_col].sum(),
            "n_hold": (g[w_col] > 0).sum(),
            "weighted_target_ret": np.sum(g[w_col] * g["target_ret"]) / max(g[w_col].sum(), 1e-12),
            "portfolio_ret_on_capital": np.sum(g[w_col] * g["target_ret"]),
            "avg_signal_z": np.average(g["signal_z"], weights=g[w_col]),
            "avg_spread_bps": np.average(g["spread_bps"], weights=g[w_col]),
        }))
        .reset_index()
    )

    sel = sel.merge(bench_min, on=["date", "minute"], how="left")
    sel["scaled_ew_ret_on_capital"] = sel["net_weight"] * sel["ew_target_ret"]
    sel["selection_excess_on_capital"] = sel["portfolio_ret_on_capital"] - sel["scaled_ew_ret_on_capital"]
    sel["selection_excess_per_gross"] = sel["weighted_target_ret"] - sel["ew_target_ret"]

    sel.to_csv(out_dir / "selected_alpha_by_minute.csv", index=False)

    daily_sel = (
        sel.groupby("date")
        .agg(
            avg_net_weight=("net_weight", "mean"),
            avg_n_hold=("n_hold", "mean"),
            avg_weighted_target_bps=("weighted_target_ret", lambda x: x.mean() * 10000.0),
            avg_ew_target_bps=("ew_target_ret", lambda x: x.mean() * 10000.0),
            avg_selection_excess_per_gross_bps=("selection_excess_per_gross", lambda x: x.mean() * 10000.0),
            sum_selection_excess_on_capital_bps=("selection_excess_on_capital", lambda x: x.sum() * 10000.0),
        )
        .reset_index()
    )
    daily_sel.to_csv(out_dir / "selected_alpha_by_day.csv", index=False)

    # 4. Summary
    summary = {
        "rows": len(base),
        "date_minutes": base[["date", "minute"]].drop_duplicates().shape[0],
        "signal_ic_mean": ic["signal_ic"].mean(),
        "signal_rank_ic_mean": ic["signal_rank_ic"].mean(),
        "signal_ic_ir": ic["signal_ic"].mean() / (ic["signal_ic"].std() + 1e-12),
        "signal_rank_ic_ir": ic["signal_rank_ic"].mean() / (ic["signal_rank_ic"].std() + 1e-12),
        "top_decile_target_bps": top_ret,
        "bottom_decile_target_bps": bot_ret,
        "top_minus_bottom_bps": top_ret - bot_ret,
        "selected_weighted_target_bps": sel["weighted_target_ret"].mean() * 10000.0,
        "selected_ew_target_bps": sel["ew_target_ret"].mean() * 10000.0,
        "selected_excess_per_gross_bps": sel["selection_excess_per_gross"].mean() * 10000.0,
        "selected_excess_per_gross_tstat": (
            sel["selection_excess_per_gross"].mean()
            / (sel["selection_excess_per_gross"].std() + 1e-12)
            * np.sqrt(len(sel))
        ),
        "avg_selected_net_weight": sel["net_weight"].mean(),
        "avg_n_hold": sel["n_hold"].mean(),
    }

    if pred_col:
        summary["pm_ic_mean"] = ic["pm_ic"].mean()
        summary["pm_rank_ic_mean"] = ic["pm_rank_ic"].mean()
        summary["pm_ic_ir"] = ic["pm_ic"].mean() / (ic["pm_ic"].std() + 1e-12)
        summary["pm_rank_ic_ir"] = ic["pm_rank_ic"].mean() / (ic["pm_rank_ic"].std() + 1e-12)

    s = pd.DataFrame([summary])
    s.to_csv(out_dir / "alpha_validation_summary.csv", index=False)

    print("\n===== alpha validation summary =====")
    print(s.T)

    print("\n===== decile =====")
    print(decile)

    print("\n===== daily selected alpha tail =====")
    print(daily_sel.tail(10))

    print("\n[saved]", out_dir)


if __name__ == "__main__":
    main()
