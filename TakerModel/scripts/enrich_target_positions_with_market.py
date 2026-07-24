# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import pandas as pd


def date_int(s):
    return (
        s.astype(str)
        .str.replace(r"\D", "", regex=True)
        .str.slice(0, 8)
        .astype(int)
    )


def norm_sid_from(df):
    if "sid" in df.columns:
        return pd.to_numeric(df["sid"], errors="coerce").astype("Int64")
    if "securityid" in df.columns:
        return pd.to_numeric(
            df["securityid"].astype(str).str.extract(r"(\d+)")[0],
            errors="coerce",
        ).astype("Int64")
    if "SecurityID" in df.columns:
        return pd.to_numeric(
            df["SecurityID"].astype(str).str.extract(r"(\d+)")[0],
            errors="coerce",
        ).astype("Int64")
    raise KeyError(f"cannot find sid/securityid columns: {df.columns.tolist()}")


def pick_col(df, candidates, name):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"cannot find {name}; candidates={candidates}; columns={df.columns.tolist()}")


def read_market_day(market_dir, d):
    f = Path(market_dir) / f"optimizer_input_{d}.parquet"
    if not f.exists():
        raise FileNotFoundError(f)

    m = pd.read_parquet(f)
    if "date" in m.columns:
        m["date_int"] = date_int(m["date"])
    else:
        m["date_int"] = int(d)

    m["datetime"] = pd.to_datetime(m["datetime"])
    m["sid_norm"] = norm_sid_from(m)

    px_col = pick_col(m, ["mid_price", "tmid", "price", "mark_price", "bid1", "ask1"], "price")
    if "benchmark_weight" not in m.columns:
        raise KeyError(f"{f} has no benchmark_weight")

    out = m[["date_int", "datetime", "sid_norm", px_col, "benchmark_weight"]].copy()
    out = out.rename(columns={"sid_norm": "sid", px_col: "mid_price"})
    out["mid_price"] = pd.to_numeric(out["mid_price"], errors="coerce")
    out["benchmark_weight"] = pd.to_numeric(out["benchmark_weight"], errors="coerce").fillna(0.0)

    out = out.drop_duplicates(["date_int", "datetime", "sid"], keep="last")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--positions", required=True)
    ap.add_argument("--market-input-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pos = pd.read_csv(args.positions)
    pos["datetime"] = pd.to_datetime(pos["datetime"])
    pos["date_int"] = date_int(pos["date"]) if "date" in pos.columns else pos["datetime"].dt.strftime("%Y%m%d").astype(int)
    pos["sid"] = norm_sid_from(pos)

    pieces = []
    for d, pday in pos.groupby("date_int", sort=True):
        d = int(d)
        print(f"===== enrich {d} n_pos={len(pday)} =====")
        mday = read_market_day(args.market_input_dir, d)

        x = pday.merge(
            mday,
            on=["date_int", "datetime", "sid"],
            how="left",
            validate="m:1",
        )

        miss_px = x["mid_price"].isna().mean()
        miss_bw = x["benchmark_weight"].isna().mean()

        print(f"missing mid_price={miss_px:.6f}, missing benchmark_weight={miss_bw:.6f}")

        if miss_px > 0:
            bad = x[x["mid_price"].isna()][["date", "datetime", "securityid", "sid"]].head(10)
            print("missing examples:")
            print(bad.to_string(index=False))
            raise RuntimeError(
                f"too many missing market rows for {d}; "
                f"exact datetime/sid merge failed. Need asof merge."
            )

        pieces.append(x)

    out = pd.concat(pieces, ignore_index=True)

    # 保留原始列，同时追加 sid/date_int/mid_price/benchmark_weight
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print("\nDONE")
    print("out:", out_path)
    print("shape:", out.shape)
    print("columns:", out.columns.tolist())
    print("avg gross from shares*price/capital:",
          (pd.to_numeric(out["actual_shares_after"], errors="coerce").fillna(0.0)
           * pd.to_numeric(out["mid_price"], errors="coerce").fillna(0.0)
          ).groupby(out["datetime"]).sum().mean() / 200000000.0)


if __name__ == "__main__":
    main()
