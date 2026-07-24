import os
import argparse
import random
import numpy as np
import pandas as pd

from multi_factor_eval import calc_one_stock
from multi_factor_eval import get_date_security_info
from multi_factor_eval import data_loader


FACTORS = [
    "fwz1_ret_1",
    "fwz1_ret_3",
    "fwz1_ret_5",
    "fwz1_ret_10",
    "fwz1_ret_30",
    "fwz1_ret_check",
    "fwz1_ret_45",
    "fwz2_obi_1",
    "fwz2_obi_3",
    "fwz2_obi_5",
    "fwz2_obi_10",
    "fwz2_weighted_obi_5",
    "fwz2_microprice_deviation",
    "fwz2_depth_imbalance_5",
    "fwz2_depth_imbalance_10",
    "fwz2_total_depth_5",
    "fwz2_total_depth_10",
    "fwz2_book_slope_imbalance_5",
    "fwz2_bid_concentration_5",
    "fwz2_ask_concentration_5",
    "fwz2_relative_spread",
    "fwz2_volatility_20",
    "fwz2_volatility_60",
    "fwz2_active_buy_ratio_10",
    "fwz2_trade_imbalance_5",
    "fwz2_trade_imbalance_10",
    "fwz2_trade_imbalance_20",
    "fwz2_trade_imbalance_30",
    "fwz2_amount_imbalance_10",
    "fwz2_vwap_deviation_10",
    "fwz2_trade_count_imbalance_10",
    "fwz2_trade_intensity_10",
    "fwz2_large_trade_imbalance_10",
    "fwz2_order_imbalance_5",
    "fwz2_order_imbalance_10",
    "fwz2_order_imbalance_20",
    "fwz2_order_imbalance_30",
    "fwz2_order_count_imbalance_10",
    "fwz2_aggressive_order_imbalance_10",
    "fwz2_order_distance_imbalance_10",
    "fwz2_cancel_pressure_10",
    "fwz2_cancel_pressure_20",
    "fwz2_near_cancel_pressure_10",
    "fwz2_buy_cancel_ratio_10",
    "fwz2_sell_cancel_ratio_10",
    "fwz2_cancel_ratio_imbalance_10",
    "fwz2_fake_liquidity_imbalance",
    "fwz2_trade_order_linear_10",
    "fwz2_trade_obi_linear_10",
    "fwz2_order_obi_linear_10",
    "fwz2_trade_order_obi_linear_10",
    "fwz2_trade_cancel_linear_10",
    "fwz2_obi_cancel_adjusted_10",
    "fwz2_liquidity_adjusted_trade_order_10",
    "fwz2_aggressive_ratio_imbalance_10",
    "fwz2_fill_efficiency_imbalance_10",
    "fwz2_large_trade_ratio_imbalance_10",
    "cf_fwz1_ret_1_mean",
    "cf_fwz1_ret_30_mean",
    "cf_fwz1_ret_check_mean",
]


def ensure_dir(path):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def is_polars_df(df):
    return hasattr(df, "write_csv") and hasattr(df, "select")


def to_pandas(df):
    if is_polars_df(df):
        return df.to_pandas()
    return df


def find_col(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    return None


def infer_id_cols(df):
    cols = list(df.columns)

    datetime_col = find_col(
        cols,
        ["datetime", "timestamp", "time", "date_time", "localtime", "update_time"],
    )

    symbol_col = find_col(
        cols,
        ["securityid", "symbol", "stock", "code", "ticker", "instrument"],
    )

    return datetime_col, symbol_col


def infer_label_cols(df, horizons):
    cols = list(df.columns)
    label_cols = []
    rename_map = {}

    for h in horizons:
        candidates = [
            f"label_{h}",
            f"ret_{h}",
            f"return_{h}",
            f"future_return_{h}",
            f"future_ret_{h}",
            f"y_{h}",
            f"pred_{h}",
            f"target_{h}",
        ]

        found = find_col(cols, candidates)

        if found is None:
            continue

        target = f"label_{h}"
        label_cols.append(target)

        if found != target:
            rename_map[found] = target

    return label_cols, rename_map


def infer_factor_cols(df):
    cols = list(df.columns)
    factor_cols = [c for c in cols if c in FACTORS]
    return factor_cols


def clean_one_df(df, horizons):
    df = to_pandas(df).copy()

    datetime_col, symbol_col = infer_id_cols(df)
    if datetime_col is None:
        raise RuntimeError(f"cannot find datetime column. columns={list(df.columns)}")

    label_cols, rename_map = infer_label_cols(df, horizons)
    if rename_map:
        df = df.rename(columns=rename_map)

    label_cols, _ = infer_label_cols(df, horizons)
    if len(label_cols) == 0:
        raise RuntimeError(f"cannot find label columns. columns={list(df.columns)}")

    factor_cols = infer_factor_cols(df)
    if len(factor_cols) == 0:
        raise RuntimeError(f"cannot find factor columns. columns={list(df.columns)}")

    if symbol_col is None:
        df["securityid"] = ""
        symbol_col = "securityid"

    keep_cols = [datetime_col, symbol_col] + factor_cols + label_cols
    out = df[keep_cols].copy()

    if datetime_col != "datetime":
        out = out.rename(columns={datetime_col: "datetime"})

    if symbol_col != "securityid":
        out = out.rename(columns={symbol_col: "securityid"})

    numeric_cols = factor_cols + label_cols
    out[numeric_cols] = out[numeric_cols].replace([np.inf, -np.inf], np.nan)

    out = out.dropna(subset=label_cols)

    for c in factor_cols:
        out[c] = out[c].fillna(0)

    return out, factor_cols, label_cols


def append_csv(df, path):
    ensure_dir(path)
    header = not os.path.exists(path)
    df.to_csv(path, mode="a", header=header, index=False)


def save_feature_yaml(feature_cols, path):
    ensure_dir(path)
    with open(path, "w") as f:
        f.write("features:\n")
        for c in feature_cols:
            f.write(f"  - {c}\n")


def save_summary(path, rows, feature_cols, label_cols, errors):
    ensure_dir(path)
    with open(path, "w") as f:
        f.write(f"rows: {rows}\n")
        f.write(f"num_features: {len(feature_cols)}\n")
        f.write("feature_cols:\n")
        for c in feature_cols:
            f.write(f"  - {c}\n")
        f.write("label_cols:\n")
        for c in label_cols:
            f.write(f"  - {c}\n")
        f.write(f"errors: {len(errors)}\n")
        for e in errors[:200]:
            f.write(str(e) + "\n")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--sample", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--horizons", type=str, default="30,60,90,120")

    parser.add_argument(
        "--output",
        type=str,
        default="FactorModel/data/raw/factor_features_202410_100.csv",
    )
    parser.add_argument(
        "--feature_yaml",
        type=str,
        default="FactorModel/data/raw/feature_cols_202410_100.yaml",
    )
    parser.add_argument(
        "--summary",
        type=str,
        default="FactorModel/data/raw/raw_dataset_summary_202410_100.txt",
    )

    args = parser.parse_args()

    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]

    if os.path.exists(args.output):
        os.remove(args.output)
    if os.path.exists(args.feature_yaml):
        os.remove(args.feature_yaml)
    if os.path.exists(args.summary):
        os.remove(args.summary)

    print("factors:")
    for f in FACTORS:
        print("  ", f)
    print("horizons:", horizons)

    date_list = get_date_security_info.get_date_list(args.start, args.end)
    print("date_list:", date_list)

    total_rows = 0
    saved_feature_cols = None
    saved_label_cols = None
    errors = []

    for date in date_list:
        print("loading date:", date)

        data_loader.init_clickhouse_client(date)

        securityids = get_date_security_info.get_securityid_list(date)

        random.seed(args.seed)
        if args.sample > 0 and len(securityids) > args.sample:
            securityids = random.sample(securityids, args.sample)

        for i, securityid in enumerate(securityids):
            try:
                print(date, i + 1, "/", len(securityids), securityid)

                df_one = calc_one_stock(
                    str(date),
                    str(securityid),
                    FACTORS,
                    horizons,
                )

                out, factor_cols, label_cols = clean_one_df(df_one, horizons)

                if len(out) == 0:
                    print("  empty after clean")
                    continue

                out["securityid"] = str(securityid)
                out.insert(0, "date", int(date))
                out.insert(1, "time", out["datetime"].copy())
                out["datetime"] = (
                    out["date"].astype(str)
                    + "_"
                    + out["time"].astype(str).str.zfill(9)
                )

                append_csv(out, args.output)

                total_rows += len(out)

                if saved_feature_cols is None:
                    saved_feature_cols = factor_cols
                    saved_label_cols = label_cols
                    save_feature_yaml(saved_feature_cols, args.feature_yaml)

                print(
                    "  appended rows:",
                    len(out),
                    "total rows:",
                    total_rows,
                    "num factors:",
                    len(factor_cols),
                    "labels:",
                    label_cols,
                )

            except Exception as e:
                msg = (date, securityid, repr(e))
                errors.append(msg)
                print("stock error:", date, securityid, repr(e))

    if total_rows == 0:
        raise RuntimeError("no raw data generated")

    save_summary(
        args.summary,
        total_rows,
        saved_feature_cols or [],
        saved_label_cols or [],
        errors,
    )

    print("saved raw data to:", args.output)
    print("saved feature yaml to:", args.feature_yaml)
    print("saved summary to:", args.summary)
    print("total rows:", total_rows)
    print("num features:", len(saved_feature_cols or []))
    print("label cols:", saved_label_cols)


if __name__ == "__main__":
    main()