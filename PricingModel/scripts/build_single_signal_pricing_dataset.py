import argparse
from pathlib import Path

import pandas as pd


def make_time_key_from_datetime(x):
    s = x.astype(str)

    out = s.copy()

    mask_underscore = s.str.contains("_", regex=False)
    out.loc[mask_underscore] = (
        s.loc[mask_underscore]
        .str.split("_")
        .str[-1]
        .str.replace(":", "", regex=False)
        .str.replace(".", "", regex=False)
        .str.zfill(9)
    )

    mask_space = s.str.contains(" ", regex=False)
    out.loc[mask_space] = (
        s.loc[mask_space]
        .str.split(" ")
        .str[-1]
        .str.replace(":", "", regex=False)
        .str.replace(".", "", regex=False)
        .str.zfill(9)
    )

    out = (
        out.astype(str)
        .str.replace(":", "", regex=False)
        .str.replace(".", "", regex=False)
        .str[-9:]
        .str.zfill(9)
    )

    return out


def normalize_key(df):
    df = df.copy()

    if "date" not in df.columns:
        raise KeyError("missing date column")

    if "securityid" not in df.columns:
        if "code" in df.columns:
            df = df.rename(columns={"code": "securityid"})
        elif "symbol" in df.columns:
            df = df.rename(columns={"symbol": "securityid"})
        else:
            raise KeyError("missing securityid / code / symbol column")

    if "datetime" not in df.columns:
        if "timestamp" in df.columns:
            df = df.rename(columns={"timestamp": "datetime"})
        elif "time" in df.columns:
            df["datetime"] = (
                df["date"].astype(str)
                + "_"
                + df["time"].astype(str).str.zfill(9)
            )
        else:
            raise KeyError("missing datetime / timestamp / time column")

    df["date"] = df["date"].astype(str)
    df["securityid"] = df["securityid"].astype(str).str.zfill(6)
    df["datetime"] = df["datetime"].astype(str)
    df["time_key"] = make_time_key_from_datetime(df["datetime"])

    return df


def create_split_by_date(df):
    dates = sorted(df["date"].unique())
    n = len(dates)

    if n < 3:
        raise ValueError("not enough dates to create train / valid / test split")

    train_end = int(n * 0.7)
    valid_end = int(n * 0.85)

    train_dates = set(dates[:train_end])
    valid_dates = set(dates[train_end:valid_end])

    def assign_split(d):
        if d in train_dates:
            return "train"
        if d in valid_dates:
            return "valid"
        return "test"

    df = df.copy()
    df["split"] = df["date"].map(assign_split)

    return df


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--market", required=True)
    parser.add_argument("--factor", required=True)
    parser.add_argument("--factor_col", required=True)
    parser.add_argument("--target_col", default="ret_120")
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    market_path = Path(args.market)
    factor_path = Path(args.factor)
    output_path = Path(args.output)

    if not market_path.exists():
        raise FileNotFoundError(market_path)

    if not factor_path.exists():
        raise FileNotFoundError(factor_path)

    print("loading market:", market_path)
    market = pd.read_csv(market_path)

    print("loading factor:", factor_path)
    factor = pd.read_csv(factor_path)

    market = normalize_key(market)
    factor = normalize_key(factor)

    if args.factor_col not in factor.columns:
        raise KeyError(
            f"factor_col not found: {args.factor_col}\n"
            f"factor columns: {factor.columns.tolist()}"
        )

    if args.target_col not in market.columns:
        raise KeyError(
            f"target_col not found in market file: {args.target_col}\n"
            f"market columns: {market.columns.tolist()}"
        )

    factor_keep_cols = [
        "date",
        "securityid",
        "time_key",
        args.factor_col,
    ]

    if "split" in factor.columns:
        factor_keep_cols.append("split")

    factor = factor[factor_keep_cols].drop_duplicates(
        ["date", "securityid", "time_key"],
        keep="last",
    )

    print("market shape:", market.shape)
    print("factor shape:", factor.shape)

    print("common dates:", len(set(market["date"]) & set(factor["date"])))
    print("common stocks:", len(set(market["securityid"]) & set(factor["securityid"])))

    df = market.merge(
        factor,
        on=["date", "securityid", "time_key"],
        how="inner",
    )

    if len(df) == 0:
        raise ValueError(
            "merged dataset is empty. "
            "Check date / securityid / time_key alignment."
        )

    if "split" not in df.columns:
        print("warning: split not found in factor file; creating split by date")
        df = create_split_by_date(df)

    df = df.dropna(subset=[args.factor_col, args.target_col])

    front_cols = [
        "date",
        "datetime",
        "securityid",
        "time_key",
        args.factor_col,
        "marketValue",
        "turnoverRate",
        "volatility_60",
        "split",
        "bid1",
        "ask1",
        "mid_price",
        "spread",
        "limit_up_price",
        "limit_down_price",
    ]

    label_cols = [
        "ret_30",
        "ret_60",
        "ret_90",
        "ret_120",
        "label_30",
        "label_60",
        "label_90",
        "label_120",
    ]

    keep_cols = []

    for c in front_cols + label_cols:
        if c in df.columns and c not in keep_cols:
            keep_cols.append(c)

    df = df[keep_cols].copy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print("saved:", output_path)
    print("shape:", df.shape)
    print("date range:", df["date"].min(), df["date"].max())
    print("num dates:", df["date"].nunique())
    print("num stocks:", df["securityid"].nunique())

    print("split:")
    print(df["split"].value_counts())

    check_cols = [args.factor_col, args.target_col]

    for c in ["marketValue", "turnoverRate", "volatility_60"]:
        if c in df.columns:
            check_cols.append(c)

    print("missing ratio:")
    print(df[check_cols].isna().mean())

    print("head:")
    print(df.head())


if __name__ == "__main__":
    main()