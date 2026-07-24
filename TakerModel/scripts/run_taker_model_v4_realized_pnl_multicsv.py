import argparse
import os
import sys
import yaml
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from run_taker_model_v3_exit_control import run_exit_control
from run_taker_model_v3_exit_control_multicsv import (
    load_yaml,
    lower_columns,
    normalize_symbol,
    prepare_optimizer,
)
from run_taker_model_v2_turnover_control import (
    ensure_dir,
    make_minute_metrics,
    make_daily_metrics,
    make_reason_metrics,
    make_summary,
)


def load_market_first_snapshot_realized_multicsv(opt, cfg):
    data_cfg = cfg["data"]
    cols = cfg["columns"]

    market_dir = data_cfg["market_data_dir"]
    pattern = data_cfg.get("market_file_pattern", "market_return_{date}_742.csv")

    datetime_col = cols["datetime_col"].lower()
    symbol_col = cols["symbol_col"].lower()
    bid_col = cols["bid_col"].lower()
    ask_col = cols["ask_col"].lower()
    mid_col = cols["mid_col"].lower()

    parts = []

    for date, opt_day in opt.groupby("execution_date"):
        file_path = os.path.join(market_dir, pattern.format(date=date))

        if not os.path.exists(file_path):
            print("WARNING: market file not found, skip:", file_path)
            continue

        needed_symbols = set(opt_day["securityid"].dropna().astype(int).unique())
        needed_minutes = set(pd.to_datetime(opt_day["execution_minute"].dropna().unique()))

        print("loading market csv:", file_path)
        print("  symbols:", len(needed_symbols), "needed minutes:", len(needed_minutes))

        header = pd.read_csv(file_path, nrows=0)
        header_lower = [str(c).strip().lower() for c in header.columns]
        lower_to_orig = {str(c).strip().lower(): c for c in header.columns}

        need_cols = [datetime_col, symbol_col, bid_col, ask_col, mid_col]
        missing = [c for c in need_cols if c not in header_lower]
        if missing:
            raise ValueError(f"{file_path} missing columns: {missing}")

        usecols = [lower_to_orig[c] for c in need_cols]

        day_parts = []

        for chunk in pd.read_csv(
            file_path,
            usecols=usecols,
            chunksize=int(cfg["runtime"].get("market_chunksize", 2000000)),
            low_memory=False,
        ):
            chunk = lower_columns(chunk)

            chunk[datetime_col] = pd.to_datetime(chunk[datetime_col], errors="coerce")
            chunk[symbol_col] = normalize_symbol(chunk[symbol_col])

            chunk = chunk[chunk[symbol_col].notna()].copy()
            chunk["_sid"] = chunk[symbol_col].astype(int)
            chunk = chunk[chunk["_sid"].isin(needed_symbols)].copy()

            if chunk.empty:
                continue

            chunk = chunk.rename(columns={
                datetime_col: "market_datetime",
                symbol_col: "securityid",
                bid_col: "bid_price",
                ask_col: "ask_price",
                mid_col: "mid_price",
            })

            for c in ["bid_price", "ask_price", "mid_price"]:
                chunk[c] = pd.to_numeric(chunk[c], errors="coerce")

            chunk = chunk[
                chunk["market_datetime"].notna()
                & chunk["securityid"].notna()
                & chunk["mid_price"].notna()
                & (chunk["mid_price"] > 0)
            ].copy()

            if chunk.empty:
                continue

            chunk["execution_minute"] = chunk["market_datetime"].dt.floor("min")

            # 先取每分钟 first snapshot，但暂时不要按 needed_minutes 过滤。
            # 因为 realized_ret_1m 需要下一分钟 mid_price。
            chunk = (
                chunk.sort_values(["securityid", "execution_minute", "market_datetime"])
                     .drop_duplicates(["securityid", "execution_minute"], keep="first")
            )

            day_parts.append(chunk[[
                "securityid",
                "execution_minute",
                "market_datetime",
                "bid_price",
                "ask_price",
                "mid_price",
            ]].copy())

        if not day_parts:
            print("  WARNING: no market rows for date", date)
            continue

        one_day = pd.concat(day_parts, ignore_index=True)
        one_day["securityid"] = normalize_symbol(one_day["securityid"])
        one_day["execution_minute"] = pd.to_datetime(one_day["execution_minute"], errors="coerce")

        one_day = (
            one_day.sort_values(["securityid", "execution_minute", "market_datetime"])
                   .drop_duplicates(["securityid", "execution_minute"], keep="first")
        )

        # realized next-minute mark-to-market return
        one_day = one_day.sort_values(["securityid", "execution_minute"]).reset_index(drop=True)
        one_day["next_mid_price"] = one_day.groupby("securityid")["mid_price"].shift(-1)
        one_day["next_execution_minute"] = one_day.groupby("securityid")["execution_minute"].shift(-1)

        one_day["label"] = one_day["next_mid_price"] / one_day["mid_price"] - 1.0
        one_day["pnl_return_source"] = "realized_next_minute_mid_return"

        # 只保留 optimizer 需要的 execution minutes
        one_day = one_day[one_day["execution_minute"].isin(needed_minutes)].copy()

        if one_day.empty:
            print("  WARNING: no matched market rows after needed minute filter for date", date)
            continue

        parts.append(one_day[[
            "securityid",
            "execution_minute",
            "market_datetime",
            "bid_price",
            "ask_price",
            "mid_price",
            "next_mid_price",
            "next_execution_minute",
            "label",
            "pnl_return_source",
        ]].copy())

        print("  loaded rows:", len(one_day))

    if not parts:
        return pd.DataFrame(columns=[
            "securityid",
            "execution_minute",
            "market_datetime",
            "bid_price",
            "ask_price",
            "mid_price",
            "next_mid_price",
            "next_execution_minute",
            "label",
            "pnl_return_source",
        ])

    out = pd.concat(parts, ignore_index=True)
    out["securityid"] = normalize_symbol(out["securityid"])
    out["execution_minute"] = pd.to_datetime(out["execution_minute"], errors="coerce")
    return out.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    optimizer_path = cfg["data"]["optimizer_output_path"]

    print("loading optimizer:", optimizer_path)
    opt = prepare_optimizer(optimizer_path, cfg)

    print("optimizer rows:", len(opt))
    print("optimizer symbols:", opt["securityid"].nunique())
    print("execution date range:", opt["execution_date"].min(), opt["execution_date"].max())
    print("execution minute range:", opt["execution_minute"].min(), opt["execution_minute"].max())

    print("loading market from daily csv files with realized next-minute return")
    mkt = load_market_first_snapshot_realized_multicsv(opt, cfg)

    print("market rows:", len(mkt))
    print("market symbols:", mkt["securityid"].nunique() if not mkt.empty else 0)

    opt["securityid"] = pd.to_numeric(opt["securityid"], errors="coerce").astype("Int64")
    mkt["securityid"] = pd.to_numeric(mkt["securityid"], errors="coerce").astype("Int64")
    opt["execution_minute"] = pd.to_datetime(opt["execution_minute"], errors="coerce")
    mkt["execution_minute"] = pd.to_datetime(mkt["execution_minute"], errors="coerce")

    if mkt.empty:
        raise ValueError("market data loaded from CSV is empty")

    print("merging")
    merged = opt.merge(mkt, on=["securityid", "execution_minute"], how="left")

    missing_mid = merged["mid_price"].isna().mean()
    missing_label = merged["label"].isna().mean()

    print("merged shape:", merged.shape)
    print("missing mid rate:", missing_mid)
    print("missing realized return rate:", missing_label)

    max_missing = float(cfg.get("runtime", {}).get("max_missing_rate", 0.2))
    if missing_mid > max_missing or missing_label > max_missing:
        raise ValueError(f"merge failed: missing_mid={missing_mid}, missing_realized_return={missing_label}")

    print("running exit-control taker model with realized PnL")
    result = run_exit_control(merged, cfg)

    minute_metrics = make_minute_metrics(result)
    daily_metrics = make_daily_metrics(result)
    reason_metrics = make_reason_metrics(result)
    summary = make_summary(result, minute_metrics, cfg)

    paths = cfg["output"]
    for p in paths.values():
        ensure_dir(p)

    result.to_csv(paths["position_output_path"], index=False)
    minute_metrics.to_csv(paths["minute_metrics_path"], index=False)
    daily_metrics.to_csv(paths["daily_metrics_path"], index=False)
    reason_metrics.to_csv(paths["trade_reason_path"], index=False)
    summary.to_csv(paths["summary_path"], index=False)

    print()
    print("saved position output:", paths["position_output_path"])
    print("saved minute metrics:", paths["minute_metrics_path"])
    print("saved daily metrics:", paths["daily_metrics_path"])
    print("saved trade reason:", paths["trade_reason_path"])
    print("saved summary:", paths["summary_path"])

    print()
    print("===== summary =====")
    print(summary.T)

    print()
    print("===== trade reason =====")
    print(reason_metrics.to_string(index=False))


if __name__ == "__main__":
    main()
