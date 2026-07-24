import argparse
import os
import random
import numpy as np  # pyright: ignore[reportMissingImports]
import pandas as pd  # pyright: ignore[reportMissingImports]
import polars as pl  # pyright: ignore[reportMissingImports]

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


def calc_one_stock(date, securityid, factor_name, horizons):
    df_snapshot, df_trade, df_order, df_cancel, df_index, df_etf_order, df_etf_trade = data_loader.data_loader(
        date, securityid
    )

    for additional_feature in config.additional_feature_list:
        df_snapshot = AdditionalFeatureFormulaDict[additional_feature](
            df_snapshot,
            df_trade,
            df_order,
            df_cancel,
            df_index,
            df_etf_order,
            df_etf_trade,
        )

    df_snapshot = df_snapshot.sort_values("timestamp").reset_index(drop=True)

    df_snapshot["f_use_check"] = [int(i % 20) for i in range(len(df_snapshot))]
    df_snapshot.loc[: int(config.default_params["max_waiting_sequence"]), "f_use_check"] = -1

    lf = pl.DataFrame(df_snapshot)
    lf = lf.with_columns([FactorFormulaDict[factor_name]()])

    keep_cols = ["SecurityID", "timestamp", "midprice", "f_use_check", factor_name]
    df = lf.select(keep_cols).to_pandas()
    df["Date"] = int(date)

    for h in horizons:
        h = int(h)

        # 和原项目 basic.py 里的 label 口径保持一致：
        # label_h = midprice(t+h) / midprice(t+1) - 1
        df[f"label_{h}"] = df["midprice"].shift(-h) / df["midprice"].shift(-1) - 1

    return df


def add_cross_section_group(df, factor_name, label_name, bins):
    tmp = df.copy()

    def qcut_one_timestamp(x):
        x = x.replace([np.inf, -np.inf], np.nan)
        if x.notna().sum() < bins or x.nunique(dropna=True) < 2:
            return pd.Series(np.nan, index=x.index)
        return pd.qcut(x.rank(method="first"), bins, labels=False, duplicates="drop")

    tmp["group"] = tmp.groupby(["Date", "timestamp"])[factor_name].transform(qcut_one_timestamp)

    group_ret = (
        tmp.dropna(subset=["group", label_name])
        .groupby("group")[label_name]
        .mean()
    )

    return group_ret


def evaluate_factor(df, factor_name, horizons, bins):
    df = df.replace([np.inf, -np.inf], np.nan)

    # 和原框架类似，减少重叠样本；只取 f_use_check == 0 的点
    df = df[df["f_use_check"] == 0].copy()

    print("\n==============================")
    print("factor:", factor_name)
    print("sample rows:", len(df))
    print("==============================")

    for h in horizons:
        label_name = f"label_{int(h)}"

        tmp = df[["Date", "SecurityID", "timestamp", factor_name, label_name]].dropna()

        if len(tmp) == 0:
            print(f"\nlabel {label_name}: no valid data")
            continue

        overall_ic = corr_safe(tmp[factor_name], tmp[label_name])
        overall_rank_ic = rank_corr_safe(tmp[factor_name], tmp[label_name])

        ic_by_stock = (
            tmp.groupby(["Date", "SecurityID"])
            .apply(lambda g: corr_safe(g[factor_name], g[label_name]))
            .rename("ic")
            .reset_index()
        )

        rank_ic_by_stock = (
            tmp.groupby(["Date", "SecurityID"])
            .apply(lambda g: rank_corr_safe(g[factor_name], g[label_name]))
            .rename("rank_ic")
            .reset_index()
        )

        mean_ic = ic_by_stock["ic"].mean()
        std_ic = ic_by_stock["ic"].std()
        icir = mean_ic / std_ic if std_ic and std_ic > 0 else np.nan

        mean_rank_ic = rank_ic_by_stock["rank_ic"].mean()
        std_rank_ic = rank_ic_by_stock["rank_ic"].std()
        rank_icir = mean_rank_ic / std_rank_ic if std_rank_ic and std_rank_ic > 0 else np.nan

        print(f"\n----- {label_name} -----")
        print("overall IC:", overall_ic)
        print("overall RankIC:", overall_rank_ic)
        print("mean timeseries IC:", mean_ic)
        print("ICIR:", icir)
        print("mean timeseries RankIC:", mean_rank_ic)
        print("RankICIR:", rank_icir)

        group_ret = add_cross_section_group(tmp, factor_name, label_name, bins)
        print("\nCross-section group return:")
        print(group_ret)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--factor", type=str, required=True)
    parser.add_argument("--sample", type=int, default=50)
    parser.add_argument("--horizons", type=str, default="30,60,90,120")
    parser.add_argument("--bins", type=int, default=5)
    parser.add_argument("--seed", type=int, default=36)
    args = parser.parse_args()

    factor_name = args.factor
    horizons = [int(x) for x in args.horizons.split(",")]

    if factor_name not in FactorFormulaDict:
        raise ValueError(f"{factor_name} not found in formula_factor.function_dict")

    date_list = get_date_security_info.get_date_list(args.start, args.end)

    all_df = []

    for date in date_list:
        print("loading date:", date)

        # 重要：每个日期重新初始化 index / ETF cache
        data_loader.init_clickhouse_client(date)

        securityids = get_date_security_info.get_securityid_list(date)

        random.seed(args.seed)
        if len(securityids) > args.sample:
            securityids = random.sample(securityids, args.sample)

        for i, securityid in enumerate(securityids):
            try:
                print(date, i + 1, "/", len(securityids), securityid)
                df_one = calc_one_stock(str(date), str(securityid), factor_name, horizons)
                all_df.append(df_one)
            except Exception as e:
                print("error:", date, securityid, repr(e))

    if len(all_df) == 0:
        raise RuntimeError("no valid data loaded")

    df = pd.concat(all_df, ignore_index=True)

    #out_file = f"single_factor_eval_{factor_name}.csv"
    #df.to_csv(out_file, index=False)
    #print("\nsaved raw eval data to:", out_file)

    evaluate_factor(df, factor_name, horizons, args.bins)


if __name__ == "__main__":
    main()