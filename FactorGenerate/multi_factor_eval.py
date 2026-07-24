from typing import Any


import argparse
import random
import numpy as np
import pandas as pd
import polars as pl
import yaml

from config import config
from utils import data_loader, get_date_security_info
from formula_additional_feature import function_dict as AdditionalFeatureFormulaDict
from formula_factor import function_dict as FactorFormulaDict


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

    for additional_feature in config.additional_feature_list:
        if additional_feature not in AdditionalFeatureFormulaDict:
            raise ValueError(f"{additional_feature} not found in formula_additional_feature.function_dict")
        df_snapshot = AdditionalFeatureFormulaDict[additional_feature](
            df_snapshot,
            df_trade,
            df_order,
            df_cancel,
            df_index,
            df_etf_order,
            df_etf_trade,
        )

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
            print("df_trade columns:", df_trade.columns.tolist())
            print("df_order columns:", df_order.columns.tolist())
            print("df_cancel columns:", df_cancel.columns.tolist())
            raise

    df_snapshot = df_snapshot.sort_values("timestamp").reset_index(drop=True)

    df_snapshot["f_use_check"] = [int(i % 20) for i in range(len(df_snapshot))]
    df_snapshot.loc[: int(config.default_params["max_waiting_sequence"]), "f_use_check"] = -1

    lf = pl.DataFrame(df_snapshot)

    for factor in factors:
        try:
            lf = lf.with_columns([FactorFormulaDict[factor]()])
        except Exception as e:
            print("factor error:", date, securityid, factor, repr(e))

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

    result = result.sort_values(
        sort_cols,
        ascending=[True, False, False][: len(sort_cols)],
    )

    result.to_csv(args.output, index=False)

    print("\nsaved result to:", args.output)
    print(result.head(30).to_string(index=False))


if __name__ == "__main__":
    main()