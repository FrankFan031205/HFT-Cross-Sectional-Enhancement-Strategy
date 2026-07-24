import argparse
import os
import sys
import yaml
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from run_taker_model_v3_exit_control import run_exit_control
from run_taker_model_v2_turnover_control import (
    ensure_dir,
    make_minute_metrics,
    make_daily_metrics,
    make_reason_metrics,
    make_summary,
)


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def lower_columns(df):
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def normalize_symbol(s):
    x = pd.to_numeric(s, errors="coerce")
    if x.notna().mean() > 0.99:
        return x.astype("Int64")
    return s.astype(str).str.strip()


def to_bool(s):
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").fillna(0).astype(float) != 0
    return s.astype(str).str.strip().str.lower().isin(["1", "true", "yes", "y", "valid", "selected"])



def parse_optimizer_datetime(x):
    s = x.astype(str).str.strip()

    # format like 20241022_093059500 = 2024-10-22 09:30:59.500
    m = s.str.match(r"^\d{8}_\d{9}$")

    out = pd.Series(pd.NaT, index=x.index, dtype="datetime64[ns]")

    if m.any():
        part = s[m]
        date_part = part.str.slice(0, 8)
        time_part = part.str.slice(9, 18)

        hh = time_part.str.slice(0, 2)
        mm = time_part.str.slice(2, 4)
        ss = time_part.str.slice(4, 6)
        ms = time_part.str.slice(6, 9)

        dt_str = (
            date_part + " " +
            hh + ":" + mm + ":" + ss + "." + ms
        )

        out.loc[m] = pd.to_datetime(
            dt_str,
            format="%Y%m%d %H:%M:%S.%f",
            errors="coerce"
        )

    if (~m).any():
        out.loc[~m] = pd.to_datetime(s[~m], errors="coerce")

    return out


def prepare_optimizer(path, cfg):
    cols = cfg["columns"]
    ta = cfg["time_alignment"]
    data_cfg = cfg["data"]

    datetime_col = cols["datetime_col"].lower()
    symbol_col = cols["symbol_col"].lower()
    target_weight_col = cols["target_weight_col"].lower()

    start_date = str(data_cfg.get("start_date", ""))
    end_date = str(data_cfg.get("end_date", ""))

    opt = pd.read_csv(path, low_memory=False)
    opt = lower_columns(opt)

    if target_weight_col not in opt.columns:
        fallback_cols = [
            "effective_target_weight",
            "target_weight",
            "weight",
            "raw_target_weight",
            "final_weight",
        ]
        found = None
        for c in fallback_cols:
            if c in opt.columns:
                found = c
                break
        if found is None:
            raise ValueError(f"cannot find target weight column. wanted={target_weight_col}, columns={opt.columns.tolist()[:80]}")
        print(f"WARNING: target_weight_col={target_weight_col} not found, using {found}")
        target_weight_col = found

    for c in [datetime_col, symbol_col, target_weight_col]:
        if c not in opt.columns:
            raise ValueError(f"optimizer missing column: {c}")

    opt[datetime_col] = parse_optimizer_datetime(opt[datetime_col])
    opt[symbol_col] = normalize_symbol(opt[symbol_col])
    opt[target_weight_col] = pd.to_numeric(opt[target_weight_col], errors="coerce").fillna(0.0)

    if "date" in opt.columns:
        opt["date"] = opt["date"].astype(str)
        if start_date:
            opt = opt[opt["date"] >= start_date].copy()
        if end_date:
            opt = opt[opt["date"] <= end_date].copy()

    role = str(ta.get("optimizer_datetime_role", "decision_time")).lower()
    lag = int(ta.get("execution_lag_minutes", 1))

    if role == "decision_time":
        decision_datetime = opt[datetime_col]
        execution_datetime = decision_datetime + pd.to_timedelta(lag, unit="m")
    elif role == "execution_time":
        execution_datetime = opt[datetime_col]
        decision_datetime = execution_datetime - pd.to_timedelta(lag, unit="m")
    else:
        raise ValueError("optimizer_datetime_role must be decision_time or execution_time")

    out = pd.DataFrame({
        "decision_datetime": decision_datetime,
        "execution_datetime": execution_datetime,
        "execution_minute": execution_datetime.dt.floor("min"),
        "execution_date": execution_datetime.dt.strftime("%Y%m%d"),
        "securityid": opt[symbol_col],
        "desired_target_weight": opt[target_weight_col].astype(float),
    })

    extra_cols = [
        "side",
        "optimizer_status",
        "state",
        "net_alpha_bps",
        "buy_edge_bps",
        "sell_edge_bps",
        "alpha_bps",
        "spread_bps",
        "cost_bps",
        "selected",
        "valid_market",
        "blocked_reason",
        "abs_delta_notional",
        "raw_target_weight",
        "target_weight",
        "effective_target_weight",
        "target_notional",
        "current_qty",
        "sellable_qty",
        "target_qty",
        "delta_qty_raw",
        "delta_qty_executable",
        "effective_target_qty",
        "position_source",
        "alpha_rank",
    ]

    for c in extra_cols:
        if c in opt.columns:
            out[c] = opt[c].values

    if "side" in out.columns:
        out["optimizer_side"] = out["side"].astype(str).str.upper()
    else:
        out["optimizer_side"] = "UNKNOWN"

    if "optimizer_status" in out.columns:
        out["optimizer_status"] = out["optimizer_status"].astype(str)
    else:
        out["optimizer_status"] = "unknown"

    # Build a unified edge column.
    # Older optimizer files may have net_alpha_bps directly.
    # Newer rank-target files have buy_edge_bps / sell_edge_bps instead.
    if "buy_edge_bps" in out.columns:
        out["buy_edge_bps"] = pd.to_numeric(out["buy_edge_bps"], errors="coerce")
    else:
        out["buy_edge_bps"] = np.nan

    if "sell_edge_bps" in out.columns:
        out["sell_edge_bps"] = pd.to_numeric(out["sell_edge_bps"], errors="coerce")
    else:
        out["sell_edge_bps"] = np.nan

    if "net_alpha_bps" in out.columns:
        out["net_alpha_bps"] = pd.to_numeric(out["net_alpha_bps"], errors="coerce")
    else:
        target_sign = np.sign(out["desired_target_weight"].fillna(0.0))

        out["net_alpha_bps"] = np.where(
            out["optimizer_side"].eq("BUY"),
            out["buy_edge_bps"],
            np.where(
                out["optimizer_side"].eq("SELL"),
                out["sell_edge_bps"],
                np.where(
                    target_sign > 0,
                    out["buy_edge_bps"],
                    np.where(target_sign < 0, out["sell_edge_bps"], np.nan)
                )
            )
        )

    if "valid_market" in out.columns:
        out["valid_market_bool"] = to_bool(out["valid_market"])
    else:
        out["valid_market_bool"] = True

    out = out.dropna(subset=["execution_datetime", "execution_minute", "securityid"])
    out = out.sort_values(["securityid", "execution_datetime"]).reset_index(drop=True)

    return out


def load_market_first_snapshot_multicsv(opt, cfg):
    data_cfg = cfg["data"]
    cols = cfg["columns"]

    market_dir = data_cfg["market_data_dir"]
    pattern = data_cfg.get("market_file_pattern", "market_return_{date}_742.csv")

    datetime_col = cols["datetime_col"].lower()
    symbol_col = cols["symbol_col"].lower()
    bid_col = cols["bid_col"].lower()
    ask_col = cols["ask_col"].lower()
    mid_col = cols["mid_col"].lower()
    label_col = cols["label_col"].lower()

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

        need_cols = [datetime_col, symbol_col, bid_col, ask_col, mid_col, label_col]
        missing = [c for c in need_cols if c not in header_lower]
        if missing:
            raise ValueError(f"{file_path} missing columns: {missing}")

        usecols = [lower_to_orig[c] for c in need_cols]

        day_parts = []

        for chunk in pd.read_csv(file_path, usecols=usecols, chunksize=int(cfg["runtime"].get("market_chunksize", 2000000)), low_memory=False):
            chunk = lower_columns(chunk)

            chunk[datetime_col] = pd.to_datetime(chunk[datetime_col], errors="coerce")
            chunk[symbol_col] = normalize_symbol(chunk[symbol_col])

            chunk = chunk[chunk[symbol_col].notna()].copy()
            chunk["_sid"] = chunk[symbol_col].astype(int)
            chunk = chunk[chunk["_sid"].isin(needed_symbols)].copy()

            if chunk.empty:
                continue

            chunk["execution_minute"] = chunk[datetime_col].dt.floor("min")
            chunk = chunk[chunk["execution_minute"].isin(needed_minutes)].copy()

            if chunk.empty:
                continue

            chunk = chunk.rename(columns={
                datetime_col: "market_datetime",
                symbol_col: "securityid",
                bid_col: "bid_price",
                ask_col: "ask_price",
                mid_col: "mid_price",
                label_col: "label",
            })

            for c in ["bid_price", "ask_price", "mid_price", "label"]:
                chunk[c] = pd.to_numeric(chunk[c], errors="coerce")

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
                "label",
            ]].copy())

        if day_parts:
            one_day = pd.concat(day_parts, ignore_index=True)
            one_day = (
                one_day.sort_values(["securityid", "execution_minute", "market_datetime"])
                       .drop_duplicates(["securityid", "execution_minute"], keep="first")
            )
            parts.append(one_day)
            print("  loaded rows:", len(one_day))
        else:
            print("  WARNING: no matched market rows for date", date)

    if not parts:
        return pd.DataFrame(columns=[
            "securityid",
            "execution_minute",
            "market_datetime",
            "bid_price",
            "ask_price",
            "mid_price",
            "label",
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

    print("loading market from daily csv files")
    mkt = load_market_first_snapshot_multicsv(opt, cfg)

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
    print("missing label rate:", missing_label)

    max_missing = float(cfg.get("runtime", {}).get("max_missing_rate", 0.2))
    if missing_mid > max_missing or missing_label > max_missing:
        raise ValueError(f"merge failed: missing_mid={missing_mid}, missing_label={missing_label}")

    print("running exit-control taker model")
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
