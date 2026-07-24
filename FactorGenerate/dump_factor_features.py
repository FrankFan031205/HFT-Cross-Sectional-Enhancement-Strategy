from typing import Any


import argparse
import random
import numpy as np
import pandas as pd
import polars as pl
import os
import re
import yaml

from config import config
from utils import data_loader, get_date_security_info
from formula_additional_feature import function_dict as AdditionalFeatureFormulaDict
from formula_factor import function_dict as FactorFormulaDict

def _df_columns(df):
    return list(df.columns)


def _is_polars(df):
    return hasattr(df, "write_csv") and hasattr(df, "select")


def _infer_factor_cols(cols):
    factor_cols = []
    for c in cols:
        if c.startswith("fwz"):
            factor_cols.append(c)
    return factor_cols


def _infer_label_cols(cols, horizons):
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
        ]

        found = None
        for c in candidates:
            if c in cols:
                found = c
                break

        if found is not None:
            target = f"label_{h}"
            label_cols.append(target)
            if found != target:
                rename_map[found] = target

    return label_cols, rename_map


def dump_raw_factor_data(df, out_path, feature_yaml_path, horizons,
                         datetime_col="datetime", symbol_col="symbol"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    os.makedirs(os.path.dirname(feature_yaml_path), exist_ok=True)

    cols = _df_columns(df)
    factor_cols = _infer_factor_cols(cols)
    label_cols, rename_map = _infer_label_cols(cols, horizons)

    if len(factor_cols) == 0:
        print("[raw dump skip] no fwz factor columns")
        return

    if len(label_cols) == 0:
        print("[raw dump skip] no label columns")
        return

    if datetime_col not in cols:
        if "timestamp" in cols:
            datetime_col = "timestamp"
        elif "time" in cols:
            datetime_col = "time"
        else:
            print("[raw dump skip] no datetime column")
            return

    if symbol_col not in cols:
        if "stock" in cols:
            symbol_col = "stock"
        elif "code" in cols:
            symbol_col = "code"
        elif "ticker" in cols:
            symbol_col = "ticker"
        else:
            print("[raw dump skip] no symbol column")
            return

    if _is_polars(df):
        import polars as pl

        if rename_map:
            df = df.rename(rename_map)

        cols_keep = [datetime_col, symbol_col] + factor_cols + label_cols
        cols_keep = [c for c in cols_keep if c in df.columns]

        out = df.select(cols_keep)

        numeric_cols = [c for c in factor_cols + label_cols if c in out.columns]

        out = out.with_columns([
            pl.when(pl.col(c).is_infinite())
              .then(None)
              .otherwise(pl.col(c))
              .alias(c)
            for c in numeric_cols
        ])

        out = out.drop_nulls(subset=label_cols)

        for c in factor_cols:
            if c in out.columns:
                out = out.with_columns(pl.col(c).fill_null(0).alias(c))

        header = not os.path.exists(out_path)

        with open(out_path, "ab") as f:
            out.write_csv(f, include_header=header)

    else:
        import numpy as np

        if rename_map:
            df = df.rename(columns=rename_map)

        cols_keep = [datetime_col, symbol_col] + factor_cols + label_cols
        cols_keep = [c for c in cols_keep if c in df.columns]

        out = df[cols_keep].copy()

        numeric_cols = [c for c in factor_cols + label_cols if c in out.columns]
        out[numeric_cols] = out[numeric_cols].replace([np.inf, -np.inf], np.nan)
        out = out.dropna(subset=label_cols)

        for c in factor_cols:
            if c in out.columns:
                out[c] = out[c].fillna(0)

        header = not os.path.exists(out_path)
        out.to_csv(out_path, mode="a", header=header, index=False)

    if not os.path.exists(feature_yaml_path):
        with open(feature_yaml_path, "w") as f:
            f.write("features:\n")
            for c in factor_cols:
                f.write(f"  - {c}\n")

    print(f"[raw dump] rows appended to {out_path}, factors={len(factor_cols)}, labels={label_cols}")

def corr_safe(x, y):
    if x.nunique() <= 1 or y.nunique() <= 1:
        return np.nan
    return x.corr(y)


def rank_corr_safe(x, y):
    if x.nunique() <= 1 or y.nunique() <= 1:
        return np.nan
    return x.rank().corr(y.rank())


def read_factor_list_from_yaml(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)

    if "factor_list" in cfg:
        factor_items = cfg["factor_list"]
    elif "factors" in cfg:
        factor_items = cfg["factors"]
    else:
        factor_items = []
        for _, v in cfg.items():
            if isinstance(v, list):
                for item in v:
                    if (
                        isinstance(item, list)
                        and len(item) == 2
                        and isinstance(item[0], str)
                    ):
                        factor_items.append(item)

    factors = []
    for item in factor_items:
        if isinstance(item, list) and len(item) >= 1:
            factors.append(item[0])
        elif isinstance(item, str):
            factors.append(item)

    return factors


def get_factors(args):
    if args.factors.lower() == "all":
        factors = read_factor_list_from_yaml(args.factor_info)
    else:
        factors = [x.strip() for x in args.factors.split(",") if x.strip()]

    factors = [x for x in factors if x in FactorFormulaDict]
    seen = set()
    factors = [x for x in factors if not (x in seen or seen.add(x))]

    if len(factors) == 0:
        raise ValueError("No valid factors found.")

    return factors


def calc_one_stock(date, securityid, factors, horizons):
    df_snapshot, df_trade, df_order, df_cancel, df_index, df_etf_order, df_etf_trade = data_loader.data_loader(
        date, securityid
    )

    df_trade, df_order, df_cancel = add_volume_alias(
        df_trade,
        df_order,
        df_cancel,
    )

    for additional_feature in config.additional_feature_list:
        if additional_feature not in AdditionalFeatureFormulaDict:
            raise ValueError(f"{additional_feature} not found in formula_additional_feature.function_dict")

        try:
            df_snapshot = AdditionalFeatureFormulaDict[additional_feature](
                df_snapshot,
                df_trade,
                df_order,
                df_cancel,
                df_index,
                df_etf_order,
                df_etf_trade,
            )
        except Exception as e:
            print("additional feature error:", additional_feature, repr(e))
            print("df_trade columns:", df_trade.columns.tolist() if df_trade is not None else None)
            print("df_order columns:", df_order.columns.tolist() if df_order is not None else None)
            print("df_cancel columns:", df_cancel.columns.tolist() if df_cancel is not None else None)
            raise

    df_snapshot = df_snapshot.sort_values("timestamp").reset_index(drop=True)

    df_snapshot["f_use_check"] = [int(i % 20) for i in range(len(df_snapshot))]
    df_snapshot.loc[: int(config.default_params["max_waiting_sequence"]), "f_use_check"] = -1

    lf = pl.DataFrame(df_snapshot)

    remaining = list(factors)
    computed = set()
    failed_last = {}

    for pass_id in range(5):
        if len(remaining) == 0:
            break

        next_remaining = []
        progress = 0

        for factor in remaining:
            if factor in computed:
                continue

            try:
                lf = lf.with_columns([FactorFormulaDict[factor]()])
                computed.add(factor)
                progress += 1
            
            except Exception as e:
                print("factor error:", date, securityid, factor, repr(e))
                failed_last[factor] = repr(e)
                next_remaining.append(factor)

        print(
            "factor pass",
            pass_id + 1,
            "computed:",
            progress,
            "remaining:",
            len(next_remaining),
        )

        if progress == 0:
            break

        remaining = next_remaining

    keep_cols = ["SecurityID", "timestamp", "midprice", "f_use_check"]
    keep_cols += [x for x in factors if x in lf.columns]

    df = lf.select(keep_cols).to_pandas()
    df["Date"] = int(date)

    for h in horizons:
        h = int(h)
        df[f"label_{h}"] = df["midprice"].shift(-h) / df["midprice"].shift(-1) - 1

    return df


def qcut_one_timestamp(x, bins):
    x = x.replace([np.inf, -np.inf], np.nan)

    if x.notna().sum() < bins or x.nunique(dropna=True) < 2:
        return pd.Series(np.nan, index=x.index)

    return pd.qcut(
        x.rank(method="first"),
        bins,
        labels=False,
        duplicates="drop",
    )


def calc_group_return(tmp, factor_name, label_name, bins):
    x = tmp[["Date", "timestamp", factor_name, label_name]].copy()

    x["group"] = x.groupby(["Date", "timestamp"])[factor_name].transform(
        lambda s: qcut_one_timestamp(s, bins)
    )

    group_ret = (
        x.dropna(subset=["group", label_name])
        .groupby("group")[label_name]
        .mean()
    )

    result = {}
    for i in range(bins):
        result[f"group_{i}"] = group_ret.get(i, np.nan)

    result["long_short"] = result.get(f"group_{bins - 1}", np.nan) - result.get("group_0", np.nan)

    return result


def evaluate_one_factor(df, factor_name, horizons, bins):
    rows = []

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df[df["f_use_check"] == 0].copy()

    if factor_name not in df.columns:
        print(f"skip {factor_name}: factor column not found")
        return rows

    for h in horizons:
        label_name = f"label_{int(h)}"

        tmp = df[
            ["Date", "SecurityID", "timestamp", factor_name, label_name]
        ].dropna()

        if len(tmp) == 0:
            rows.append({
                "factor": factor_name,
                "horizon": int(h),
                "sample_rows": 0,
                "overall_ic": np.nan,
                "overall_rankic": np.nan,
                "mean_timeseries_ic": np.nan,
                "icir": np.nan,
                "mean_timeseries_rankic": np.nan,
                "rankicir": np.nan,
            })
            continue

        overall_ic = corr_safe(tmp[factor_name], tmp[label_name])
        overall_rankic = rank_corr_safe(tmp[factor_name], tmp[label_name])

        ic_by_stock = (
            tmp.groupby(["Date", "SecurityID"])
            .apply(lambda g: corr_safe(g[factor_name], g[label_name]))
            .rename("ic")
            .reset_index()
        )

        rankic_by_stock = (
            tmp.groupby(["Date", "SecurityID"])
            .apply(lambda g: rank_corr_safe(g[factor_name], g[label_name]))
            .rename("rankic")
            .reset_index()
        )

        mean_ic = ic_by_stock["ic"].mean()
        std_ic = ic_by_stock["ic"].std()
        icir = mean_ic / std_ic if std_ic and std_ic > 0 else np.nan

        mean_rankic = rankic_by_stock["rankic"].mean()
        std_rankic = rankic_by_stock["rankic"].std()
        rankicir = mean_rankic / std_rankic if std_rankic and std_rankic > 0 else np.nan

        group_result = calc_group_return(tmp, factor_name, label_name, bins)

        row = {
            "factor": factor_name,
            "horizon": int(h),
            "sample_rows": len(tmp),
            "overall_ic": overall_ic,
            "overall_rankic": overall_rankic,
            "mean_timeseries_ic": mean_ic,
            "icir": icir,
            "mean_timeseries_rankic": mean_rankic,
            "rankicir": rankicir,
        }

        row.update(group_result)
        rows.append(row)

    return rows

def add_volume_alias(df_trade, df_order, df_cancel):
    """
    Compatibility fix:
    Some formula functions expect column name 'volume',
    while actual ClickHouse columns use 'qty' for trade/order
    and 'cancel_qty' for cancel.
    """

    if df_trade is not None:
        if "volume" not in df_trade.columns and "qty" in df_trade.columns:
            df_trade["volume"] = df_trade["qty"]
        if "qty" not in df_trade.columns and "volume" in df_trade.columns:
            df_trade["qty"] = df_trade["volume"]

    if df_order is not None:
        if "volume" not in df_order.columns and "qty" in df_order.columns:
            df_order["volume"] = df_order["qty"]
        if "qty" not in df_order.columns and "volume" in df_order.columns:
            df_order["qty"] = df_order["volume"]

    if df_cancel is not None:
        if "volume" not in df_cancel.columns and "cancel_qty" in df_cancel.columns:
            df_cancel["volume"] = df_cancel["cancel_qty"]
        if "qty" not in df_cancel.columns and "cancel_qty" in df_cancel.columns:
            df_cancel["qty"] = df_cancel["cancel_qty"]
        if "cancel_qty" not in df_cancel.columns and "volume" in df_cancel.columns:
            df_cancel["cancel_qty"] = df_cancel["volume"]

    return df_trade, df_order, df_cancel


def _format_intraday_time(x):
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    digits = re.sub(r"\D", "", s)
    if len(digits) == 0:
        return None

    # If timestamp includes date, keep only the intraday HHMMSSmmm part.
    return digits[-9:].zfill(9)


def build_wide_factor_features(df, factors, horizons):
    """
    Convert the internal evaluation dataframe into the wide feature matrix
    needed by FactorModel.

    Output columns:
        date, datetime, securityid, fwz factor columns, label_* columns

    The original script evaluated factors and wrote a summary table.
    For ML training, we need one row per stock-timestamp with all factor values.
    """

    out = df.replace([np.inf, -np.inf], np.nan).copy()

    if "f_use_check" in out.columns:
        out = out[out["f_use_check"] == 0].copy()

    factor_cols = [x for x in factors if x in out.columns]
    label_cols = [f"label_{int(h)}" for h in horizons if f"label_{int(h)}" in out.columns]

    if len(factor_cols) == 0:
        raise RuntimeError("no factor columns found in generated dataframe")

    if len(label_cols) == 0:
        raise RuntimeError("no label columns found in generated dataframe")

    out["date"] = out["Date"].astype(int)
    out["securityid"] = out["SecurityID"].astype(str).str.zfill(6)

    time_str = out["timestamp"].map(_format_intraday_time)
    out["datetime"] = out["date"].astype(str) + "_" + time_str.astype(str)

    keep_cols = ["date", "datetime", "securityid"] + factor_cols + label_cols
    out = out[keep_cols].copy()

    out = out.dropna(subset=label_cols)

    for c in factor_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
        out[c] = out[c].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    for c in label_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.sort_values(["date", "securityid", "datetime"]).reset_index(drop=True)

    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--factors", type=str, default="all")
    parser.add_argument("--factor_info", type=str, default="factor_info.yaml")
    parser.add_argument("--sample", type=int, default=50)
    parser.add_argument("--horizons", type=str, default="30,60,90,120")
    parser.add_argument("--bins", type=int, default=5)
    parser.add_argument("--seed", type=int, default=36)
    parser.add_argument("--output", type=str, default="multi_factor_eval_result.csv")
    args = parser.parse_args()

    factors = get_factors(args)
    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]

    print("factors:")
    for x in factors:
        print("  ", x)

    print("horizons:", horizons)

    date_list = get_date_security_info.get_date_list(args.start, args.end)

    all_df = []

    for date in date_list:
        print("loading date:", date)

        data_loader.init_clickhouse_client(date)

        securityids = get_date_security_info.get_securityid_list(date)

        random.seed(args.seed)
        if len(securityids) > args.sample:
            securityids = random.sample(securityids, args.sample)

        for i, securityid in enumerate(securityids):
            try:
                print(date, i + 1, "/", len(securityids), securityid)
                df_one = calc_one_stock(str(date), str(securityid), factors, horizons)
                all_df.append(df_one)
            except Exception as e:
                print("stock error:", date, securityid, repr(e))

    if len(all_df) == 0:
        raise RuntimeError("no valid data loaded")

    df = pd.concat(all_df, ignore_index=True)

    # Main deliverable: wide feature matrix for FactorModel ML training.
    wide = build_wide_factor_features(df, factors, horizons)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    wide.to_csv(args.output, index=False)

    print("\nsaved wide factor feature matrix to:", args.output)
    print("wide shape:", wide.shape)
    print("wide columns:", len(wide.columns))
    print("first columns:", wide.columns.tolist()[:30])

    # Optional diagnostic: keep the old factor evaluation summary,
    # but save it to a separate file to avoid overwriting the wide matrix.
    result_rows = []

    for factor in factors:
        print("\n==============================")
        print("evaluating factor:", factor)
        print("==============================")

        rows = evaluate_one_factor(df, factor, horizons, args.bins)
        result_rows.extend(rows)

    result = pd.DataFrame(result_rows)

    sort_cols = ["horizon", "icir", "overall_ic"]
    sort_cols = [c for c in sort_cols if c in result.columns]

    if len(result) > 0 and len(sort_cols) > 0:
        result = result.sort_values(
            sort_cols,
            ascending=[True, False, False][: len(sort_cols)],
        )

    eval_output = args.output.replace(".csv", "_eval.csv")
    result.to_csv(eval_output, index=False)

    print("\nsaved eval summary to:", eval_output)
    if len(result) > 0:
        print(result.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
