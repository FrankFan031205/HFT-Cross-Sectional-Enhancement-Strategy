import argparse
import pandas as pd
from pathlib import Path


def yyyymmdd_to_dash(x):
    x = str(x)
    return f"{x[:4]}-{x[4:6]}-{x[6:8]}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--daily_dir", default="/home/cross_common_data/daily_market")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    daily_dir = Path(args.daily_dir)
    base_path = Path(args.base)
    out_path = Path(args.output)

    base = pd.read_csv(base_path, usecols=["date", "securityid"])
    base["date"] = base["date"].astype(str)
    base["securityid"] = base["securityid"].astype(str).str.zfill(6)

    dates = sorted(base["date"].unique())
    securities = set(base["securityid"].unique())

    print("num dates:", len(dates))
    print("num securities:", len(securities))
    print("first date:", dates[0])
    print("last date:", dates[-1])

    parts = []

    for d in dates:
        d_dash = yyyymmdd_to_dash(d)
        f = daily_dir / f"daily_market_{d_dash}.csv"

        if not f.exists():
            print("missing file:", f)
            continue

        print("reading:", f)

        daily = pd.read_csv(f)

        need_cols = ["ticker", "tradeDate", "marketValue", "turnoverRate"]
        for c in need_cols:
            if c not in daily.columns:
                raise KeyError(f"missing column {c} in {f}")

        tmp = daily[need_cols].copy()

        tmp["securityid"] = (
            tmp["ticker"]
            .astype(str)
            .str.replace(r"\..*$", "", regex=True)
            .str.zfill(6)
        )

        tmp["date"] = (
            tmp["tradeDate"]
            .astype(str)
            .str.replace("-", "", regex=False)
        )

        tmp = tmp[tmp["securityid"].isin(securities)]
        tmp = tmp[tmp["date"] == d]

        tmp = tmp[["date", "securityid", "marketValue", "turnoverRate"]]
        tmp = tmp.drop_duplicates(["date", "securityid"], keep="last")

        print("matched rows:", len(tmp))

        parts.append(tmp)

    if len(parts) == 0:
        raise ValueError("No daily market features extracted.")

    out = pd.concat(parts, ignore_index=True)
    out = out.drop_duplicates(["date", "securityid"], keep="last")
    out = out.sort_values(["date", "securityid"])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print("saved:", out_path)
    print("shape:", out.shape)
    print("missing ratio:")
    print(out[["marketValue", "turnoverRate"]].isna().mean())


if __name__ == "__main__":
    main()