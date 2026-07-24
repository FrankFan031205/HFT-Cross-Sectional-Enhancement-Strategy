#!/usr/bin/env python3
import os
import glob
import pandas as pd

ROOT = "/mnt/data1/fwz/HFT_010-dev_fwz"

FACTOR_IN = f"{ROOT}/FactorModel/data/raw/factor_features_20241022_20250114_742.csv"
FACTOR_OUT = f"{ROOT}/FactorModel/data/raw/factor_features_20241022_20250114_742_h360.csv"

MARKET_DIR = f"{ROOT}/PricingModel/data/market_return_20241022_20250114_742_by_date"
LABEL_DIR = f"{ROOT}/FactorModel/data/raw/label_360_742_by_date"

H = 360
CHUNKSIZE = 1_000_000


def norm_sid(s):
    return (
        s.astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(6)
    )


def get_key_col(cols):
    # Important: factor_features uses datetime, so market labels must also use datetime.
    # If we use market "time" but factor "datetime", merge will produce all-NaN label_360.
    if "datetime" in cols:
        return "datetime"
    if "time" in cols:
        return "time"
    raise RuntimeError(f"cannot find time/datetime column, cols={cols}")


def norm_key(s):
    return s.astype(str)


def build_daily_labels():
    os.makedirs(LABEL_DIR, exist_ok=True)

    files = sorted(glob.glob(f"{MARKET_DIR}/market_return_*_742.csv"))
    if not files:
        raise RuntimeError(f"no market files found: {MARKET_DIR}")

    print("num market files:", len(files), flush=True)

    for path in files:
        date = os.path.basename(path).split("_")[2]
        out = f"{LABEL_DIR}/label_360_{date}_742.csv"

        if os.path.exists(out) and os.path.getsize(out) > 0:
            print("[SKIP LABEL]", date, flush=True)
            continue

        cols = pd.read_csv(path, nrows=0).columns.tolist()
        sid_col = "securityid" if "securityid" in cols else "SecurityID"
        key_col = get_key_col(cols)

        print("[BUILD LABEL]", date, "key:", key_col, path, flush=True)

        df = pd.read_csv(
            path,
            usecols=["date", sid_col, key_col, "mid_price"],
            dtype={sid_col: str, key_col: str},
        )

        df = df.rename(columns={sid_col: "securityid", key_col: "key"})
        df["date"] = df["date"].astype(int)
        df["securityid"] = norm_sid(df["securityid"])
        df["key"] = norm_key(df["key"])

        df = df.sort_values(["date", "securityid", "key"]).reset_index(drop=True)

        future_mid = df.groupby(["date", "securityid"], sort=False)["mid_price"].shift(-H)
        df["label_360"] = ((future_mid - df["mid_price"]) / df["mid_price"]).astype("float32")

        df[["date", "securityid", "key", "label_360"]].to_csv(out, index=False)

        print(
            "[SAVED LABEL]",
            out,
            "rows:",
            len(df),
            "nan_ratio:",
            float(df["label_360"].isna().mean()),
            flush=True,
        )


def add_label_to_factor_file():
    if not os.path.exists(FACTOR_IN):
        raise FileNotFoundError(FACTOR_IN)

    factor_cols = pd.read_csv(FACTOR_IN, nrows=0).columns.tolist()
    factor_key_col = get_key_col(factor_cols)

    print("[FACTOR IN]", FACTOR_IN, flush=True)
    print("[FACTOR OUT]", FACTOR_OUT, flush=True)
    print("[FACTOR KEY]", factor_key_col, flush=True)

    if os.path.exists(FACTOR_OUT):
        os.remove(FACTOR_OUT)

    label_cache = {}
    first = True
    total = 0
    total_label_non_na = 0

    for chunk in pd.read_csv(
        FACTOR_IN,
        chunksize=CHUNKSIZE,
        dtype={"securityid": str, factor_key_col: str},
    ):
        if "label_360" in chunk.columns:
            chunk = chunk.drop(columns=["label_360"])

        chunk["date"] = chunk["date"].astype(int)
        chunk["securityid"] = norm_sid(chunk["securityid"])
        chunk["key"] = norm_key(chunk[factor_key_col])

        out_parts = []

        for date, g in chunk.groupby("date", sort=False):
            date = int(date)
            label_path = f"{LABEL_DIR}/label_360_{date}_742.csv"

            if date not in label_cache:
                if not os.path.exists(label_path):
                    raise FileNotFoundError(label_path)

                lab = pd.read_csv(
                    label_path,
                    dtype={"securityid": str, "key": str},
                )
                lab["date"] = lab["date"].astype(int)
                lab["securityid"] = norm_sid(lab["securityid"])
                lab["key"] = norm_key(lab["key"])

                label_cache[date] = lab

                if len(label_cache) > 3:
                    old = sorted(label_cache.keys())[0]
                    label_cache.pop(old, None)

            merged = g.merge(
                label_cache[date],
                on=["date", "securityid", "key"],
                how="left",
            )

            out_parts.append(merged)

        out_chunk = pd.concat(out_parts, ignore_index=True)
        out_chunk = out_chunk.drop(columns=["key"])

        non_na = int(out_chunk["label_360"].notna().sum())
        total_label_non_na += non_na

        out_chunk.to_csv(
            FACTOR_OUT,
            mode="w" if first else "a",
            header=first,
            index=False,
        )

        first = False
        total += len(out_chunk)

        print(
            "[WRITE]",
            "total:",
            total,
            "chunk_rows:",
            len(out_chunk),
            "chunk_label_360_non_na:",
            non_na,
            "chunk_nan_ratio:",
            float(out_chunk["label_360"].isna().mean()),
            flush=True,
        )

    print("[DONE]", FACTOR_OUT, flush=True)
    print("total rows:", total, flush=True)
    print("total label_360 non-na:", total_label_non_na, flush=True)
    print("total label_360 non-na ratio:", total_label_non_na / total if total else 0, flush=True)


def main():
    build_daily_labels()
    add_label_to_factor_file()


if __name__ == "__main__":
    main()
