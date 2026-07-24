import pandas as pd
from pathlib import Path


MARKET_PATH = Path("data/market_return_202410_100_enriched.csv")
FACTOR_DIR = Path("../FactorModel/outputs")
OUT_PATH = Path("data/pricing_dataset_h60_202410_100.csv")


FACTOR_FILES = [
    FACTOR_DIR / "hidden_factor_lgbm_h60_202410_100.csv",
    FACTOR_DIR / "hidden_factor_ridge_h60_202410_100.csv",
    FACTOR_DIR / "hidden_factor_attention_h60_202410_100.csv",
    FACTOR_DIR / "hidden_factor_lookback_attention_h60_202410_100.csv",
    FACTOR_DIR / "hidden_factor_mlp2_h60_202410_100.csv",
]


def normalize_key(df):
    df = df.copy()

    if "securityid" not in df.columns:
        if "code" in df.columns:
            df = df.rename(columns={"code": "securityid"})
        elif "symbol" in df.columns:
            df = df.rename(columns={"symbol": "securityid"})
        else:
            raise KeyError("missing securityid/code/symbol column")

    if "datetime" not in df.columns:
        if "timestamp" in df.columns:
            df = df.rename(columns={"timestamp": "datetime"})
        else:
            raise KeyError("missing datetime/timestamp column")

    df["date"] = df["date"].astype(str)
    df["securityid"] = df["securityid"].astype(str).str.zfill(6)
    df["datetime"] = df["datetime"].astype(str)

    return df


def read_factor(path, keep_split=False):
    df = pd.read_csv(path)
    df = normalize_key(df)

    factor_cols = [c for c in df.columns if c.startswith("hidden_factor")]

    if len(factor_cols) == 0:
        raise ValueError(f"no hidden_factor column found in {path}")

    keep_cols = ["date", "datetime", "securityid"] + factor_cols

    if keep_split and "split" in df.columns:
        keep_cols.append("split")

    df = df[keep_cols].drop_duplicates(["date", "datetime", "securityid"])

    return df, factor_cols


def main():
    market = pd.read_csv(MARKET_PATH)
    market = normalize_key(market)

    existing_files = [p for p in FACTOR_FILES if p.exists()]

    print("existing factor files:")
    for p in existing_files:
        print(" ", p)

    if len(existing_files) == 0:
        raise FileNotFoundError("No hidden factor files found.")

    first_factor, first_cols = read_factor(existing_files[0], keep_split=True)

    df = market.merge(
        first_factor,
        on=["date", "datetime", "securityid"],
        how="inner",
    )

    all_factor_cols = list(first_cols)

    print("after first merge:", df.shape)

    for path in existing_files[1:]:
        fac, cols = read_factor(path, keep_split=False)

        df = df.merge(
            fac,
            on=["date", "datetime", "securityid"],
            how="left",
        )

        all_factor_cols.extend(cols)

        print("merged", path.name, df.shape)

    all_factor_cols = sorted(set(all_factor_cols))

    if "split" not in df.columns:
        print("warning: split column not found. Creating split by date.")

        dates = sorted(df["date"].unique())
        train_dates = dates[:-2]
        valid_date = dates[-2]
        test_date = dates[-1]

        def assign_split(d):
            if d in train_dates:
                return "train"
            if d == valid_date:
                return "valid"
            return "test"

        df["split"] = df["date"].map(assign_split)

    front_cols = [
        "date",
        "datetime",
        "securityid",
    ]

    signal_cols = all_factor_cols

    control_cols = [
        "marketValue",
        "turnoverRate",
        "volatility_60",
    ]

    market_cols = [
        "split",
        "bid1",
        "ask1",
        "mid_price",
        "spread",
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
    for group in [front_cols, signal_cols, control_cols, market_cols, label_cols]:
        for c in group:
            if c in df.columns and c not in keep_cols:
                keep_cols.append(c)

    df = df[keep_cols].copy()

    df.to_csv(OUT_PATH, index=False)

    print("saved:", OUT_PATH)
    print("shape:", df.shape)
    print("factor cols:", signal_cols)
    print("missing ratio:")
    check_cols = signal_cols + [c for c in control_cols if c in df.columns]
    print(df[check_cols].isna().mean())


if __name__ == "__main__":
    main()