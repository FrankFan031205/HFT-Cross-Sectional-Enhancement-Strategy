from pathlib import Path
import pandas as pd
import numpy as np

market_dir = Path("data/market_return_20241022_20250114_csi1000_by_date")
files = sorted(market_dir.glob("market_return_*_csi1000.csv"))

horizon = 360
price_col = "mid_price"

print("market_dir:", market_dir)
print("num files:", len(files))

if len(files) == 0:
    raise FileNotFoundError(f"no market_return_*_csi1000.csv found in {market_dir}")

for i, p in enumerate(files, 1):
    print(f"[{i}/{len(files)}] processing {p.name}")

    df = pd.read_csv(p, dtype={"securityid": str})
    df["securityid"] = (
        df["securityid"]
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(6)
    )

    if price_col not in df.columns:
        raise KeyError(f"{p.name} missing {price_col}")

    if "datetime" in df.columns:
        sort_cols = ["securityid", "datetime"]
    elif "time" in df.columns:
        sort_cols = ["securityid", "time"]
    else:
        raise KeyError(f"{p.name} need datetime or time column")

    df = df.sort_values(sort_cols).reset_index(drop=True)

    future_price = df.groupby("securityid", sort=False)[price_col].shift(-horizon)

    df["ret_360"] = future_price / df[price_col] - 1.0
    df["label_360"] = df["ret_360"]

    non_na = df["ret_360"].notna().mean()

    df.to_csv(p, index=False)

    print(f"  saved {p.name}, rows={len(df)}, ret_360_non_na={non_na:.6f}")

print("DONE")
