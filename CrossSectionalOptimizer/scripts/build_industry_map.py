import os
import sys
import glob
import argparse
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from CrossSectionalOptimizer.src.io_utils import (
    load_config,
    resolve_path,
    ensure_parent,
    pick_existing_path,
)


def read_one_csv(path):
    for enc in ["utf-8-sig", "utf-8", "gbk"]:
        try:
            return pd.read_csv(path, dtype=str, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, dtype=str)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="CrossSectionalOptimizer/config/optimizer_1min.yaml")
    args = parser.parse_args()

    cfg = load_config(resolve_path(args.config))
    data_cfg = cfg["data"]

    industry_dir = pick_existing_path(
        data_cfg["industry_dir"],
        data_cfg.get("fallback_industry_dir")
    )

    print("industry_dir:", industry_dir)

    files = sorted(glob.glob(os.path.join(industry_dir, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No industry csv files found in {industry_dir}")

    dfs = []
    for f in files:
        df = read_one_csv(f)
        df["source_file"] = os.path.basename(f)
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    df.columns = [c.strip() for c in df.columns]

    print("raw shape:", df.shape)
    print("columns:", list(df.columns))

    if "ticker" not in df.columns:
        raise ValueError("ticker column not found in industry data")

    df["securityid"] = (
        df["ticker"]
        .astype(str)
        .str.extract(r"(\d+)")[0]
        .str.zfill(6)
    )

    df["intoDate"] = pd.to_datetime(df.get("intoDate"), errors="coerce")
    df["outDate"] = pd.to_datetime(df.get("outDate"), errors="coerce")
    df["outDate"] = df["outDate"].fillna(pd.Timestamp("2099-12-31"))

    start_date = pd.to_datetime(cfg["date"]["start_date"])
    end_date = pd.to_datetime(cfg["date"]["end_date"])

    all_days = []

    for d in pd.date_range(start_date, end_date, freq="D"):
        active = df[(df["intoDate"] <= d) & (df["outDate"] >= d)].copy()

        if active.empty:
            continue

        active = active.sort_values(["securityid", "intoDate"])
        active = active.drop_duplicates(subset=["securityid"], keep="last")
        active["trade_date"] = d.strftime("%Y-%m-%d")

        if "industryID1" not in active.columns:
            active["industryID1"] = active.get("industryID", "UNKNOWN")

        if "industryName1" not in active.columns:
            active["industryName1"] = active.get("industry", "UNKNOWN")

        keep_cols = [
            "trade_date",
            "securityid",
            "ticker",
            "exchangeCD",
            "secShortName",
            "industryVersionCD",
            "industry",
            "industryID",
            "industrySymbol",
            "industryID1",
            "industryName1",
            "industryID2",
            "industryName2",
            "industryID3",
            "industryName3",
            "intoDate",
            "outDate",
            "isNew",
            "source_file",
        ]

        keep_cols = [c for c in keep_cols if c in active.columns]
        all_days.append(active[keep_cols])

    if not all_days:
        raise ValueError("No active industry records found in date range")

    out = pd.concat(all_days, ignore_index=True)
    out["industryID1"] = out["industryID1"].fillna("UNKNOWN")
    out["industryName1"] = out["industryName1"].fillna("UNKNOWN")

    out_path = ensure_parent(data_cfg["industry_map_path"])
    out.to_csv(out_path, index=False)

    print("saved:", out_path)
    print("shape:", out.shape)
    print("num_stocks:", out["securityid"].nunique())
    print("num_industry_level1:", out["industryID1"].nunique())
    print(out[["trade_date", "securityid", "industryID1", "industryName1"]].head())


if __name__ == "__main__":
    main()
