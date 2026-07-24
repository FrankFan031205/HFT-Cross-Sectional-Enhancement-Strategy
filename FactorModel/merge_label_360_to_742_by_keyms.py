#!/usr/bin/env python3
import os
import pandas as pd

ROOT = "/mnt/data1/fwz/HFT_010-dev_fwz"

FACTOR_IN = f"{ROOT}/FactorModel/data/raw/factor_features_20241022_20250114_742.csv"
FACTOR_OUT = f"{ROOT}/FactorModel/data/raw/factor_features_20241022_20250114_742_h360.csv"
LABEL_DIR = f"{ROOT}/FactorModel/data/raw/label_360_742_by_date"

CHUNKSIZE = 1_000_000


def norm_sid(s):
    return (
        s.astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(6)
    )


def make_key_ms(x):
    """
    Fast parser for:
      factor datetime: 20241022_093240000
      label key:       2024-10-22 09:32:40.000
    """
    s = x.astype(str).str.strip()

    out = pd.Series(index=s.index, dtype="int64")

    # label format: 2024-10-22 09:32:40.000
    mask_label = s.str.contains(":", regex=False)

    if mask_label.any():
        dt = pd.to_datetime(
            s.loc[mask_label],
            format="%Y-%m-%d %H:%M:%S.%f",
            errors="coerce",
        )
        out.loc[mask_label] = (
            dt.dt.hour * 3600000
            + dt.dt.minute * 60000
            + dt.dt.second * 1000
            + dt.dt.microsecond // 1000
        ).astype("int64")

    # factor format: 20241022_093240000
    mask_factor = ~mask_label
    if mask_factor.any():
        d = s.loc[mask_factor].str.replace(r"\D", "", regex=True)
        t = d.str[-9:]  # HHMMSSmmm
        hh = t.str.slice(0, 2).astype("int64")
        mm = t.str.slice(2, 4).astype("int64")
        ss = t.str.slice(4, 6).astype("int64")
        ms = t.str.slice(6, 9).astype("int64")
        out.loc[mask_factor] = hh * 3600000 + mm * 60000 + ss * 1000 + ms

    return out.astype("int64")

def load_label(date):
    path = f"{LABEL_DIR}/label_360_{date}_742.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    cols = pd.read_csv(path, nrows=0).columns.tolist()
    key_col = "key" if "key" in cols else "datetime"

    lab = pd.read_csv(
        path,
        usecols=["date", "securityid", key_col, "label_360"],
        dtype={"securityid": str, key_col: str},
    )

    lab["date"] = lab["date"].astype(int)
    lab["securityid"] = norm_sid(lab["securityid"])
    lab["key_ms"] = make_key_ms(lab[key_col])

    return lab[["date", "securityid", "key_ms", "label_360"]]


def main():
    if not os.path.exists(FACTOR_IN):
        raise FileNotFoundError(FACTOR_IN)

    if os.path.exists(FACTOR_OUT):
        os.remove(FACTOR_OUT)

    print("[FACTOR IN]", FACTOR_IN, flush=True)
    print("[FACTOR OUT]", FACTOR_OUT, flush=True)
    print("[LABEL DIR]", LABEL_DIR, flush=True)

    label_cache = {}
    first = True
    total = 0
    total_non_na = 0

    for chunk in pd.read_csv(
        FACTOR_IN,
        chunksize=CHUNKSIZE,
        dtype={"securityid": str, "datetime": str},
    ):
        if "label_360" in chunk.columns:
            chunk = chunk.drop(columns=["label_360"])

        chunk["date"] = chunk["date"].astype(int)
        chunk["securityid"] = norm_sid(chunk["securityid"])
        chunk["key_ms"] = make_key_ms(chunk["datetime"])

        parts = []

        for date, g in chunk.groupby("date", sort=False):
            date = int(date)

            if date not in label_cache:
                label_cache[date] = load_label(date)

                if len(label_cache) > 3:
                    old = sorted(label_cache.keys())[0]
                    label_cache.pop(old, None)

            merged = g.merge(
                label_cache[date],
                on=["date", "securityid", "key_ms"],
                how="left",
            )
            parts.append(merged)

        out_chunk = pd.concat(parts, ignore_index=True)
        out_chunk = out_chunk.drop(columns=["key_ms"])

        non_na = int(out_chunk["label_360"].notna().sum())
        total_non_na += non_na
        total += len(out_chunk)

        print(
            "[WRITE]",
            "total:", total,
            "chunk_rows:", len(out_chunk),
            "chunk_label_360_non_na:", non_na,
            "chunk_nan_ratio:", float(out_chunk["label_360"].isna().mean()),
            flush=True,
        )

        if total <= CHUNKSIZE and non_na == 0:
            raise RuntimeError("first chunk label_360 is all NaN; merge key still not matched")

        out_chunk.to_csv(
            FACTOR_OUT,
            mode="w" if first else "a",
            header=first,
            index=False,
        )
        first = False

    print("[DONE]", FACTOR_OUT, flush=True)
    print("total rows:", total, flush=True)
    print("label_360 non-na ratio:", total_non_na / total if total else 0, flush=True)


if __name__ == "__main__":
    main()
