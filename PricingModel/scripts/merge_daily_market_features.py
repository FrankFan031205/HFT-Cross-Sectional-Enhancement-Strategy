import argparse
import pandas as pd
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--daily", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    base_path = Path(args.base)
    daily_path = Path(args.daily)
    out_path = Path(args.output)

    base = pd.read_csv(base_path)
    daily = pd.read_csv(daily_path)

    for df in [base, daily]:
        df["date"] = df["date"].astype(str)
        df["securityid"] = df["securityid"].astype(str).str.zfill(6)

    daily = daily[["date", "securityid", "marketValue", "turnoverRate"]]
    daily = daily.drop_duplicates(["date", "securityid"], keep="last")

    out = base.merge(
        daily,
        on=["date", "securityid"],
        how="left",
    )

    cols = ["marketValue", "turnoverRate", "volatility_60"]

    print("shape:", out.shape)
    print("missing ratio:")
    print(out[cols].isna().mean())
    print()
    print(out[["date", "datetime", "securityid"] + cols].head())

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print("saved:", out_path)


if __name__ == "__main__":
    main()