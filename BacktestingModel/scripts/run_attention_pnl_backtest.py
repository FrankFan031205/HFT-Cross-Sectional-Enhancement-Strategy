import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.io import load_yaml, parse_datetime_series, save_csv


def standardize_securityid(s):
    return s.astype(str).str.replace(".0", "", regex=False).str.zfill(6)


def load_snapshot_window(cfg, fills):
    path = Path(cfg["input"]["snapshot_state_path"])
    if not path.exists():
        raise RuntimeError(f"snapshot_state_path not found: {path}")

    signal_col = cfg.get("signal", {}).get("col", "hidden_factor_attention_h60")
    horizon_sec = int(cfg["backtest"].get("horizon_sec", 60))
    chunksize = int(cfg["backtest"].get("snapshot_chunksize", 1000000))

    fills = fills.copy()
    fills["decision_time"] = pd.to_datetime(fills["decision_time"])
    fills["securityid"] = standardize_securityid(fills["securityid"])

    start_dt = fills["decision_time"].min() - pd.Timedelta(seconds=2)
    end_dt = fills["decision_time"].max() + pd.Timedelta(seconds=horizon_sec + 2)
    symbols = set(fills["securityid"].unique())

    header = pd.read_csv(path, nrows=0)
    usecols = ["datetime", "securityid", "mid_price"]

    optional = [
        "spread",
        "spread_ticks",
        "bid1",
        "ask1",
        "microprice",
        "liquidity_state",
    ]

    for c in optional:
        if c in header.columns and c not in usecols:
            usecols.append(c)

    # Signal may come from enriched fills / quote_decision.
    # Do not require it to exist in the big snapshot csv.
    if signal_col not in header.columns:
        print(f"[snapshot] signal column not found in snapshot csv: {signal_col}; will use signal from fills if available.")

    print(f"[snapshot] source: {path}")
    print(f"[snapshot] signal_col: {signal_col}")
    print(f"[snapshot] time window: {start_dt} -> {end_dt}")
    print(f"[snapshot] symbols: {len(symbols)}")
    print(f"[snapshot] usecols: {usecols}")

    parts = []
    total = 0
    matched = 0

    for i, chunk in enumerate(pd.read_csv(path, usecols=usecols, chunksize=chunksize, low_memory=False)):
        total += len(chunk)

        chunk["datetime"] = parse_datetime_series(chunk["datetime"], "snapshot datetime")
        chunk["securityid"] = standardize_securityid(chunk["securityid"])

        m = (
            (chunk["datetime"] >= start_dt)
            & (chunk["datetime"] <= end_dt)
            & (chunk["securityid"].isin(symbols))
        )

        out = chunk.loc[m].copy()
        if len(out):
            parts.append(out)
            matched += len(out)

        if i % 10 == 0:
            print(f"[snapshot] chunk={i}, scanned={total}, matched={matched}")

    if not parts:
        raise RuntimeError("No snapshot rows loaded in target window.")

    snap = pd.concat(parts, ignore_index=True)
    snap = snap.drop_duplicates(["datetime", "securityid"])
    snap["datetime"] = pd.to_datetime(snap["datetime"])
    snap["securityid"] = standardize_securityid(snap["securityid"])
    snap = snap.sort_values(["securityid", "datetime"]).reset_index(drop=True)

    print(f"[snapshot] final rows={len(snap)}")
    print("[snapshot] datetime range:", snap["datetime"].min(), "->", snap["datetime"].max())
    if signal_col in snap.columns:
        print("[snapshot] signal missing ratio:", snap[signal_col].isna().mean())
    else:
        print(f"[snapshot] signal column {signal_col} not in snapshot; will use signal from fills.")

    return snap


def merge_asof_by_symbol(left, right, left_time, right_time, direction="nearest", tolerance_ms=1000):
    outs = []

    for sid, lg in left.groupby("securityid", sort=False):
        rg = right[right["securityid"] == sid]
        if len(rg) == 0:
            continue

        lg = lg.sort_values(left_time).copy()
        rg = rg.sort_values(right_time).copy()

        merged = pd.merge_asof(
            lg,
            rg,
            left_on=left_time,
            right_on=right_time,
            direction=direction,
            tolerance=pd.Timedelta(milliseconds=tolerance_ms),
            suffixes=("", "_snap"),
        )

        outs.append(merged)

    if not outs:
        return left.iloc[0:0].copy()

    return pd.concat(outs, ignore_index=True)


def compute_pnl(fills, snap, cfg):
    signal_col = cfg.get("signal", {}).get("col", "hidden_factor_attention_h60")
    horizon_sec = int(cfg["backtest"].get("horizon_sec", 60))
    fee_cfg = cfg.get("fee", {})
    commission_rate = float(fee_cfg.get("commission_rate", fee_cfg.get("fee_rate", 0.0)))
    transfer_fee_rate = float(fee_cfg.get("transfer_fee_rate", 0.0))
    handling_fee_rate = float(fee_cfg.get("handling_fee_rate", 0.0))
    regulatory_fee_rate = float(fee_cfg.get("regulatory_fee_rate", 0.0))
    stamp_duty_rate = float(fee_cfg.get("stamp_duty_rate", 0.0))

    buy_fee_rate = commission_rate + transfer_fee_rate + handling_fee_rate + regulatory_fee_rate
    sell_fee_rate = buy_fee_rate + stamp_duty_rate

    df = fills.copy()
    df["decision_time"] = pd.to_datetime(df["decision_time"])
    df["fill_time"] = pd.to_datetime(df["fill_time"])
    df["securityid"] = standardize_securityid(df["securityid"])
    df["markout_time"] = df["decision_time"] + pd.Timedelta(seconds=horizon_sec)

    # decision snapshot: decision mid / spread.
    # Signal can come from enriched fills. Only add signal from snapshot if it exists there
    # and is not already present in fills.
    decision_cols = ["securityid", "datetime", "mid_price"]

    signal_in_fills = signal_col in df.columns
    if (not signal_in_fills) and (signal_col in snap.columns):
        decision_cols.append(signal_col)

    for c in ["spread", "spread_ticks", "bid1", "ask1", "microprice", "liquidity_state"]:
        if c in snap.columns and c not in decision_cols:
            decision_cols.append(c)

    decision_snap = snap[decision_cols].copy()
    decision_snap = decision_snap.rename(columns={
        "datetime": "decision_snapshot_time",
        "mid_price": "decision_mid",
    })

    df = merge_asof_by_symbol(
        left=df,
        right=decision_snap,
        left_time="decision_time",
        right_time="decision_snapshot_time",
        direction="nearest",
        tolerance_ms=1000,
    )

    # future snapshot: future mid
    future_snap = snap[["securityid", "datetime", "mid_price"]].copy()
    future_snap = future_snap.rename(columns={
        "datetime": "future_snapshot_time",
        "mid_price": "future_mid",
    })

    df = merge_asof_by_symbol(
        left=df,
        right=future_snap,
        left_time="markout_time",
        right_time="future_snapshot_time",
        direction="nearest",
        tolerance_ms=1000,
    )

    print("[pnl] rows before drop:", len(df))
    print("[pnl] missing signal ratio:", df[signal_col].isna().mean() if len(df) else np.nan)
    print("[pnl] missing future_mid ratio:", df["future_mid"].isna().mean() if len(df) else np.nan)

    debug_cols = [
        "decision_time",
        "decision_snapshot_time",
        "markout_time",
        "future_snapshot_time",
        "securityid",
        "side",
        "fill_price",
        "future_mid",
        signal_col,
    ]
    print("\n[pnl] sample merged rows:")
    print(df[[c for c in debug_cols if c in df.columns]].head(10).to_string(index=False))

    if signal_col not in df.columns:
        raise RuntimeError(
            f"Signal column {signal_col} not found after merge. "
            "Run scripts/enrich_fills_with_attention.py first, or set cfg['signal']['col'] to an existing fills column."
        )

    df = df.dropna(subset=["future_mid", signal_col]).copy()

    print("[pnl] rows after drop:", len(df))

    if len(df) == 0:
        return df

    df["fill_price"] = pd.to_numeric(df["fill_price"], errors="coerce")
    df["fill_qty"] = pd.to_numeric(df["fill_qty"], errors="coerce")
    df["future_mid"] = pd.to_numeric(df["future_mid"], errors="coerce")

    buy = df["side"].astype(str).str.upper() == "BUY"
    sell = df["side"].astype(str).str.upper() == "SELL"

    df["notional"] = df["fill_price"] * df["fill_qty"]

    df["fee_rate_applied"] = np.nan
    df.loc[buy, "fee_rate_applied"] = buy_fee_rate
    df.loc[sell, "fee_rate_applied"] = sell_fee_rate
    df["fee"] = df["notional"] * df["fee_rate_applied"]

    df["gross_pnl"] = np.nan
    df.loc[buy, "gross_pnl"] = (df.loc[buy, "future_mid"] - df.loc[buy, "fill_price"]) * df.loc[buy, "fill_qty"]
    df.loc[sell, "gross_pnl"] = (df.loc[sell, "fill_price"] - df.loc[sell, "future_mid"]) * df.loc[sell, "fill_qty"]

    df["net_pnl"] = df["gross_pnl"] - df["fee"]
    df["pnl_bps"] = df["gross_pnl"] / df["notional"] * 10000
    df["net_pnl_bps"] = df["net_pnl"] / df["notional"] * 10000
    df["win"] = df["net_pnl"] > 0
    df["date"] = df["decision_time"].dt.strftime("%Y%m%d")

    return df


def make_summary(trades):
    if len(trades) == 0:
        return pd.DataFrame([{
            "num_trades": 0,
            "num_buy": 0,
            "num_sell": 0,
            "total_notional": 0.0,
            "total_fee": 0.0,
            "total_gross_pnl": 0.0,
            "total_net_pnl": 0.0,
            "avg_net_pnl": np.nan,
            "median_net_pnl": np.nan,
            "win_rate": np.nan,
            "avg_net_pnl_bps": np.nan,
        }])

    row = {
        "num_trades": len(trades),
        "num_buy": int((trades["side"] == "BUY").sum()),
        "num_sell": int((trades["side"] == "SELL").sum()),
        "total_notional": trades["notional"].sum(),
        "total_fee": trades["fee"].sum(),
        "total_gross_pnl": trades["gross_pnl"].sum(),
        "total_net_pnl": trades["net_pnl"].sum(),
        "avg_net_pnl": trades["net_pnl"].mean(),
        "median_net_pnl": trades["net_pnl"].median(),
        "win_rate": trades["win"].mean(),
        "avg_net_pnl_bps": trades["net_pnl_bps"].mean(),
    }

    for side in ["BUY", "SELL"]:
        x = trades[trades["side"] == side]
        row[f"{side.lower()}_num"] = len(x)
        row[f"{side.lower()}_total_net_pnl"] = x["net_pnl"].sum()
        row[f"{side.lower()}_avg_net_pnl"] = x["net_pnl"].mean() if len(x) else np.nan
        row[f"{side.lower()}_win_rate"] = x["win"].mean() if len(x) else np.nan
        row[f"{side.lower()}_avg_net_pnl_bps"] = x["net_pnl_bps"].mean() if len(x) else np.nan

    return pd.DataFrame([row])


def make_factor_pnl(trades, cfg):
    signal_col = cfg.get("signal", {}).get("col", "hidden_factor_attention_h60")
    n_bins = int(cfg.get("signal", {}).get("n_bins", 5))

    if len(trades) == 0:
        return pd.DataFrame(columns=[
            "factor_bucket", "side", "num_trades", "signal_mean", "signal_min", "signal_max",
            "total_net_pnl", "avg_net_pnl", "win_rate", "avg_net_pnl_bps"
        ])

    df = trades.dropna(subset=[signal_col]).copy()

    if df[signal_col].nunique() < 2:
        df["factor_bucket"] = 1
    else:
        df["factor_bucket"] = pd.qcut(
            df[signal_col],
            q=min(n_bins, df[signal_col].nunique()),
            labels=False,
            duplicates="drop",
        ) + 1

    bucket = (
        df.groupby(["factor_bucket", "side"])
        .agg(
            num_trades=("net_pnl", "size"),
            signal_mean=(signal_col, "mean"),
            signal_min=(signal_col, "min"),
            signal_max=(signal_col, "max"),
            total_net_pnl=("net_pnl", "sum"),
            avg_net_pnl=("net_pnl", "mean"),
            win_rate=("win", "mean"),
            avg_net_pnl_bps=("net_pnl_bps", "mean"),
        )
        .reset_index()
        .sort_values(["factor_bucket", "side"])
    )

    overall = (
        df.groupby("factor_bucket")
        .agg(
            num_trades=("net_pnl", "size"),
            signal_mean=(signal_col, "mean"),
            signal_min=(signal_col, "min"),
            signal_max=(signal_col, "max"),
            total_net_pnl=("net_pnl", "sum"),
            avg_net_pnl=("net_pnl", "mean"),
            win_rate=("win", "mean"),
            avg_net_pnl_bps=("net_pnl_bps", "mean"),
        )
        .reset_index()
    )
    overall["side"] = "ALL"

    cols = [
        "factor_bucket", "side", "num_trades", "signal_mean", "signal_min", "signal_max",
        "total_net_pnl", "avg_net_pnl", "win_rate", "avg_net_pnl_bps"
    ]

    return pd.concat([overall[cols], bucket[cols]], ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/backtest.yaml")
    parser.add_argument("--fills", default="outputs/fills/fills_touched_mlp2_h60_202410_100.csv")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    pnl_path = args.output or cfg["output"]["pnl_path"]
    daily_path = cfg["output"]["daily_pnl_path"]
    summary_path = cfg["output"]["summary_path"]
    factor_pnl_path = cfg["output"].get("factor_pnl_path", "outputs/metrics/factor_pnl_attention.csv")

    print("[1] loading fills:", args.fills)
    fills = pd.read_csv(args.fills, low_memory=False)

    if len(fills) == 0:
        raise RuntimeError("fills file is empty")

    fills["decision_time"] = pd.to_datetime(fills["decision_time"])
    fills["securityid"] = standardize_securityid(fills["securityid"])

    print("fills shape:", fills.shape)
    print("decision range:", fills["decision_time"].min(), "->", fills["decision_time"].max())
    print("side counts:")
    print(fills["side"].value_counts())

    print("[2] loading attention signal and future mid")
    snap = load_snapshot_window(cfg, fills)

    print("[3] computing pnl")
    trades = compute_pnl(fills, snap, cfg)

    print("[4] saving outputs")
    save_csv(trades, pnl_path)

    if len(trades):
        daily = (
            trades.groupby("date")
            .agg(
                num_trades=("net_pnl", "size"),
                total_notional=("notional", "sum"),
                total_fee=("fee", "sum"),
                gross_pnl=("gross_pnl", "sum"),
                net_pnl=("net_pnl", "sum"),
                avg_net_pnl=("net_pnl", "mean"),
                win_rate=("win", "mean"),
                avg_net_pnl_bps=("net_pnl_bps", "mean"),
            )
            .reset_index()
        )
    else:
        daily = pd.DataFrame(columns=[
            "date", "num_trades", "total_notional", "total_fee", "gross_pnl",
            "net_pnl", "avg_net_pnl", "win_rate", "avg_net_pnl_bps"
        ])

    save_csv(daily, daily_path)

    summary = make_summary(trades)
    save_csv(summary, summary_path)

    factor_pnl = make_factor_pnl(trades, cfg)
    save_csv(factor_pnl, factor_pnl_path)

    print("\n===== summary =====")
    print(summary.T.to_string())

    print("\n===== pnl by attention bucket =====")
    print(factor_pnl.to_string(index=False))

    print("\n===== pnl by side =====")
    if len(trades):
        print(
            trades.groupby("side")
            .agg(
                num=("net_pnl", "size"),
                total_net_pnl=("net_pnl", "sum"),
                avg_net_pnl=("net_pnl", "mean"),
                win_rate=("win", "mean"),
                avg_net_pnl_bps=("net_pnl_bps", "mean"),
            )
            .to_string()
        )
    else:
        print("empty trades")


if __name__ == "__main__":
    main()
