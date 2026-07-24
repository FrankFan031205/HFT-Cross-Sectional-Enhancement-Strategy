#!/usr/bin/env python3
import os
import glob
import pandas as pd

PROJECT_ROOT = "/mnt/data1/fwz/HFT_010-dev_fwz"

FACTOR_IN = f"{PROJECT_ROOT}/FactorModel/data/raw/factor_features_week2_20_20241022_20250114_csi1000.csv"
FACTOR_OUT = f"{PROJECT_ROOT}/FactorModel/data/raw/factor_features_week2_20_20241022_20250114_csi1000_h360.csv"

MARKET_DIR = f"{PROJECT_ROOT}/PricingModel/data/market_return_20241022_20250114_csi1000_by_date"
LABEL_DIR = f"{PROJECT_ROOT}/FactorModel/data/raw/label_360_csi1000_by_date"

H = 360
CHUNKSIZE = 1_000_000


def norm_sid(s):
    return (
        s.astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(6)
    )


def build_daily_labels():
    os.makedirs(LABEL_DIR, exist_ok=True)

    files = sorted(glob.glob(f"{MARKET_DIR}/market_return_*_csi1000.csv"))
    if not files:
        raise RuntimeError(f"no market files found: {MARKET_DIR}")

    print("num market files:", len(files), flush=True)

    for path in files:
        date = os.path.basename(path).split("_")[2]
        out = f"{LABEL_DIR}/label_360_{date}_csi1000.csv"

        if os.path.exists(out) and os.path.getsize(out) > 0:
            print("[SKIP LABEL]", date, out, flush=True)
            continue

        print("[BUILD LABEL]", date, path, flush=True)

        cols = pd.read_csv(path, nrows=0).columns.tolist()

        sid_col = "securityid" if "securityid" in cols else "SecurityID"
        time_col = "datetime" if "datetime" in cols else "time"

        usecols = ["date", sid_col, time_col, "mid_price"]

        df = pd.read_csv(path, usecols=usecols, dtype={sid_col: str, time_col: str})
        df = df.rename(columns={sid_col: "securityid", time_col: "datetime"})

        df["date"] = df["date"].astype(int)
        df["securityid"] = norm_sid(df["securityid"])
        df["datetime"] = df["datetime"].astype(str)

        df = df.sort_values(["date", "securityid", "datetime"]).reset_index(drop=True)

        future_mid = df.groupby(["date", "securityid"], sort=False)["mid_price"].shift(-H)
        df["label_360"] = ((future_mid - df["mid_price"]) / df["mid_price"]).astype("float32")

        df[["date", "datetime", "securityid", "label_360"]].to_csv(out, index=False)

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

    if os.path.exists(FACTOR_OUT):
        os.remove(FACTOR_OUT)

    first = True
    total = 0

    label_cache = {}

    print("[FACTOR IN]", FACTOR_IN, flush=True)
    print("[FACTOR OUT]", FACTOR_OUT, flush=True)

    for chunk in pd.read_csv(
        FACTOR_IN,
        chunksize=CHUNKSIZE,
        dtype={"securityid": str, "datetime": str},
    ):
        if "label_360" in chunk.columns:
            chunk = chunk.drop(columns=["label_360"])

        chunk["date"] = chunk["date"].astype(int)
        chunk["securityid"] = norm_sid(chunk["securityid"])
        chunk["datetime"] = chunk["datetime"].astype(str)

        out_parts = []

        for date, g in chunk.groupby("date", sort=False):
            date = int(date)
            label_path = f"{LABEL_DIR}/label_360_{date}_csi1000.csv"

            if date not in label_cache:
                if not os.path.exists(label_path):
                    raise FileNotFoundError(label_path)

                lab = pd.read_csv(
                    label_path,
                    dtype={"securityid": str, "datetime": str},
                )
                lab["date"] = lab["date"].astype(int)
                lab["securityid"] = norm_sid(lab["securityid"])
                lab["datetime"] = lab["datetime"].astype(str)

                label_cache[date] = lab

                # 防止缓存太大，只保留最近 3 天
                if len(label_cache) > 3:
                    old = sorted(label_cache.keys())[0]
                    label_cache.pop(old, None)

            merged = g.merge(
                label_cache[date],
                on=["date", "datetime", "securityid"],
                how="left",
            )
            out_parts.append(merged)

        out_chunk = pd.concat(out_parts, ignore_index=True)

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
            "label_360_nan_ratio:",
            float(out_chunk["label_360"].isna().mean()),
            flush=True,
        )

    print("[DONE]", FACTOR_OUT, flush=True)


def main():
    build_daily_labels()
    add_label_to_factor_file()


if __name__ == "__main__":
    main()
