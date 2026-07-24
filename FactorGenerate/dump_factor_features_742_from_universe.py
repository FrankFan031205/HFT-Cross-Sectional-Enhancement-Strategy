#!/usr/bin/env python3
"""
Dump previous raw fwz factors for the 742-stock universe using the first-month period.

Run from FactorGenerate root:

cd /home/fwz/projects/HFT_010-dev_fwz/FactorGenerate

nohup python -u dump_factor_features_742_from_universe.py \
  --start 20241022 \
  --end 20241122 \
  --universe_dir ../PricingModel/data/market_return_20241022_20250114_742_by_date \
  --feature_yaml ../FactorModel/data/raw/feature_cols_20241022_20241122_100.yaml \
  --output ../FactorModel/data/raw/factor_features_20241022_20241122_742.csv \
  --overwrite \
  > ../FactorModel/logs/dump_factor_features_20241022_20241122_742.log 2>&1 &
"""

import os
import argparse
import yaml
import numpy as np
import pandas as pd

from utils import data_loader, get_date_security_info
from formula_factor import function_dict as FactorFormulaDict
from dump_factor_features import calc_one_stock


def ensure_dir(path):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def read_features_from_yaml(path):
    with open(path, "r") as f:
        obj = yaml.safe_load(f)

    if isinstance(obj, dict) and "features" in obj:
        features = list(obj["features"])
    elif isinstance(obj, list):
        features = list(obj)
    else:
        raise RuntimeError(f"Unsupported yaml format: {path}")

    features = [x for x in features if isinstance(x, str)]
    features = [x for x in features if x in FactorFormulaDict]

    seen = set()
    features = [x for x in features if not (x in seen or seen.add(x))]

    if len(features) == 0:
        raise RuntimeError("No valid factors found from feature_yaml")

    return features


def read_universe_for_date(universe_dir, date):
    path = os.path.join(universe_dir, f"market_return_{date}_742.csv")

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    header = pd.read_csv(path, nrows=0).columns.tolist()

    sid_col = None
    for c in ["securityid", "SecurityID", "symbol", "ticker", "code"]:
        if c in header:
            sid_col = c
            break

    if sid_col is None:
        raise RuntimeError(f"Cannot find security id column in {path}. columns={header[:50]}")

    df = pd.read_csv(path, usecols=[sid_col], dtype={sid_col: str})
    sids = (
        df[sid_col]
        .dropna()
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(6)
        .unique()
        .tolist()
    )

    return sorted(sids)


def format_datetime(date, timestamp):
    s = str(timestamp)

    if "." in s:
        s = s.split(".")[0]

    s = "".join([c for c in s if c.isdigit()])
    s = s.zfill(9)

    return f"{int(date)}_{s}"


def clean_one_stock_output(df, date, factors, horizons):
    if df is None or len(df) == 0:
        return None

    df = df.copy()

    if "Date" in df.columns:
        df["date"] = df["Date"].astype(int)
    else:
        df["date"] = int(date)

    if "SecurityID" in df.columns:
        df["securityid"] = df["SecurityID"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    elif "securityid" in df.columns:
        df["securityid"] = df["securityid"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    else:
        raise RuntimeError("calc_one_stock output has no SecurityID/securityid column")

    if "timestamp" not in df.columns:
        raise RuntimeError("calc_one_stock output has no timestamp column")

    df["datetime"] = df["timestamp"].map(lambda x: format_datetime(date, x))

    if "f_use_check" in df.columns:
        df = df[df["f_use_check"] == 0].copy()

    label_cols = [f"label_{int(h)}" for h in horizons]

    for c in factors:
        if c not in df.columns:
            df[c] = np.nan

    for c in factors + label_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df[factors + label_cols] = df[factors + label_cols].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=label_cols)

    for c in factors:
        df[c] = df[c].fillna(0.0)

    keep_cols = ["date", "datetime", "securityid"] + factors + label_cols
    return df[keep_cols]


def append_csv(df, output):
    if df is None or len(df) == 0:
        return 0

    ensure_dir(output)
    header = not os.path.exists(output)

    df.to_csv(output, mode="a", header=header, index=False)
    return len(df)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--universe_dir", type=str, required=True)
    parser.add_argument("--feature_yaml", type=str, required=True)
    parser.add_argument("--horizons", type=str, default="30,60,90,120")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]
    factors = read_features_from_yaml(args.feature_yaml)

    print("=" * 100, flush=True)
    print("Dump factor features for 742 universe", flush=True)
    print("start:", args.start, flush=True)
    print("end:", args.end, flush=True)
    print("universe_dir:", args.universe_dir, flush=True)
    print("feature_yaml:", args.feature_yaml, flush=True)
    print("num factors:", len(factors), flush=True)
    print("factors:", factors, flush=True)
    print("horizons:", horizons, flush=True)
    print("output:", args.output, flush=True)
    print("=" * 100, flush=True)

    if args.overwrite and os.path.exists(args.output):
        os.remove(args.output)
        print("removed existing output:", args.output, flush=True)

    if os.path.exists(args.output):
        raise RuntimeError(f"output already exists: {args.output}. Use --overwrite if you want to replace it.")

    date_list = get_date_security_info.get_date_list(args.start, args.end)

    total_rows = 0
    error_rows = []

    for date in date_list:
        print("\n" + "=" * 100, flush=True)
        print("loading date:", date, flush=True)

        data_loader.init_clickhouse_client(date)

        try:
            securityids = read_universe_for_date(args.universe_dir, date)
        except Exception as e:
            print("date universe error:", date, repr(e), flush=True)
            error_rows.append({"date": date, "securityid": "", "error": f"universe_error: {repr(e)}"})
            continue

        print("num universe stocks:", len(securityids), flush=True)

        for i, sid in enumerate(securityids):
            try:
                print(date, i + 1, "/", len(securityids), sid, flush=True)

                df_one = calc_one_stock(str(date), str(sid), factors, horizons)

                out = clean_one_stock_output(df_one, date=date, factors=factors, horizons=horizons)

                n = append_csv(out, args.output)
                total_rows += n

                print("  appended rows:", n, "total:", total_rows, flush=True)

            except Exception as e:
                print("stock error:", date, sid, repr(e), flush=True)
                error_rows.append({"date": date, "securityid": sid, "error": repr(e)})

    print("\n" + "=" * 100, flush=True)
    print("DONE", flush=True)
    print("saved:", args.output, flush=True)
    print("total rows:", total_rows, flush=True)

    if error_rows:
        err_path = args.output.replace(".csv", "_errors.csv")
        pd.DataFrame(error_rows).to_csv(err_path, index=False)
        print("errors saved:", err_path, flush=True)
        print("num errors:", len(error_rows), flush=True)


if __name__ == "__main__":
    main()
