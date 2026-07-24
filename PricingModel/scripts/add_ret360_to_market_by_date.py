from pathlib import Path
import pandas as pd
import numpy as np

market_dir = Path("data/market_return_20241022_20250114_742_by_date")
files = sorted(market_dir.glob("market_return_*_742.csv"))

horizon = 360
price_col = "mid_price"

print("num files:", len(files))

for i, p in enumerate(files, 1):
    print(f"[{i}/{len(files)}] processing {p.name}")

    df = pd.read_csv(p, dtype={"securityid": str})
    df["securityid"] = df["securityid"].astype(str).str.zfill(6)

    sort_cols = ["securityid"]
    if "datetime" in df.columns:
        sort_cols.append("datetime")
    elif "time" in df.columns:
        sort_cols.append("time")
    else:
        raise KeyError("need datetime or time column")

    df = df.sort_values(sort_cols).reset_index(drop=True)

    future_price = df.groupby("securityid")[price_col].shift(-horizon)
    df["ret_360"] = future_price / df[price_col] - 1.0
    df["label_360"] = df["ret_360"]

    df.to_csv(p, index=False)

print("DONE")
