import argparse
import numpy as np
import pandas as pd
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    df = pd.read_csv(in_path)

    need = ["date", "datetime", "securityid", "mid_price"]
    for c in need:
        if c not in df.columns:
            raise KeyError(f"missing column: {c}")

    df["date"] = df["date"].astype(str)
    df["securityid"] = df["securityid"].astype(str).str.zfill(6)

    df = df.sort_values(["date", "securityid", "datetime"]).copy()

    df["mid_ret_1"] = (
        df.groupby(["date", "securityid"])["mid_price"]
        .pct_change()
        .replace([np.inf, -np.inf], np.nan)
    )

    df["volatility_60"] = (
        df.groupby(["date", "securityid"])["mid_ret_1"]
        .rolling(60, min_periods=10)
        .std()
        .reset_index(level=[0, 1], drop=True)
        .fillna(0.0)
    )

    df = df.drop(columns=["mid_ret_1"])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print("saved:", out_path)
    print("shape:", df.shape)
    print(df["volatility_60"].describe())


if __name__ == "__main__":
    main()