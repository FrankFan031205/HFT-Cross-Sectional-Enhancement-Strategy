# -*- coding: utf-8 -*-
from pathlib import Path
import argparse
import pandas as pd
import numpy as np


def norm_date_from_series(s):
    return (
        s.astype(str)
        .str.replace(r"\D", "", regex=True)
        .str.slice(0, 8)
        .astype(int)
    )


def norm_sid(df):
    for c in ["sid", "securityid", "SecurityID", "ticker", "instrument"]:
        if c in df.columns:
            return pd.to_numeric(
                df[c].astype(str).str.extract(r"(\d+)")[0],
                errors="coerce",
            ).astype("Int64")
    raise KeyError(f"cannot find sid/securityid columns: {df.columns.tolist()}")


def pick_col(df, candidates, name, required=True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"cannot find {name}; candidates={candidates}; columns={df.columns.tolist()}")
    return None


def normalize_datetime(df):
    if "datetime" in df.columns:
        return pd.to_datetime(df["datetime"])
    if "ts_real" in df.columns:
        return pd.to_datetime(df["ts_real"])
    if "timestamp" in df.columns:
        return pd.to_datetime(df["timestamp"])
    raise KeyError(f"cannot find datetime columns: {df.columns.tolist()}")


def read_pricing(pricing_path: Path, pred_col: str):
    files = []
    if pricing_path.is_dir():
        files = sorted(list(pricing_path.rglob("*priced_dataset.csv")) + list(pricing_path.rglob("*.parquet")))
    else:
        files = [pricing_path]

    if not files:
        raise FileNotFoundError(pricing_path)

    parts = []
    for p in files:
        print(f"===== read pricing {p} =====", flush=True)
        if p.suffix.lower() == ".csv":
            df0 = pd.read_csv(p, nrows=0)
            cols0 = df0.columns.tolist()
            usecols = []
            for c in cols0:
                lc = c.lower()
                if lc in ["date", "datetime", "timestamp", "ts", "ts_real", "sid", "securityid"]:
                    usecols.append(c)
                elif c == pred_col or lc in ["pred_ret", "pred", "score", "alpha", "fair_price", "alpha_price", "alpha_ticks"]:
                    usecols.append(c)
            usecols = list(dict.fromkeys(usecols))
            df = pd.read_csv(p, usecols=usecols)
        else:
            df = pd.read_parquet(p)

        df["datetime"] = normalize_datetime(df)
        if "date" in df.columns:
            df["date"] = norm_date_from_series(df["date"])
        else:
            df["date"] = df["datetime"].dt.strftime("%Y%m%d").astype(int)

        df["sid"] = norm_sid(df)
        if pred_col not in df.columns:
            fallback = pick_col(df, ["pred_ret", "pred", "score", "alpha"], "prediction")
            print(f"[WARN] {pred_col} not found; using {fallback}")
            df[pred_col] = df[fallback]

        df["pred_ret"] = pd.to_numeric(df[pred_col], errors="coerce").fillna(0.0)

        keep = ["date", "datetime", "sid", "pred_ret"]
        df = df[keep].dropna(subset=["datetime", "sid"]).drop_duplicates(["date", "datetime", "sid"], keep="last")
        parts.append(df)

    out = pd.concat(parts, ignore_index=True)
    out = out.drop_duplicates(["date", "datetime", "sid"], keep="last")
    print("pricing rows:", len(out), "dates:", out["date"].min(), out["date"].max())
    return out


def read_market_day(path: Path):
    df = pd.read_csv(path)
    df["datetime"] = normalize_datetime(df)

    if "date" in df.columns:
        df["date"] = norm_date_from_series(df["date"])
    else:
        m = path.name
        import re
        mm = re.search(r"(20\d{6})", m)
        if mm:
            df["date"] = int(mm.group(1))
        else:
            df["date"] = df["datetime"].dt.strftime("%Y%m%d").astype(int)

    df["sid"] = norm_sid(df)
    df["securityid"] = df["sid"].astype(str).str.zfill(6)

    px_col = pick_col(
        df,
        ["mid_price", "tmid", "mid", "price", "lastprice", "close", "vwap"],
        "mid price",
        required=False,
    )
    if px_col is None:
        bid_col = pick_col(df, ["bid1", "bid_price", "bidprice1"], "bid price")
        ask_col = pick_col(df, ["ask1", "ask_price", "askprice1"], "ask price")
        df["mid_price"] = (
            pd.to_numeric(df[bid_col], errors="coerce")
            + pd.to_numeric(df[ask_col], errors="coerce")
        ) / 2.0
    else:
        df["mid_price"] = pd.to_numeric(df[px_col], errors="coerce")

    if "benchmark_weight" in df.columns:
        df["benchmark_weight"] = pd.to_numeric(df["benchmark_weight"], errors="coerce").fillna(0.0)
        s = df.groupby("datetime")["benchmark_weight"].transform("sum")
        df["benchmark_weight"] = np.where(s > 1e-12, df["benchmark_weight"] / s, 0.0)
    else:
        n = df.groupby("datetime")["sid"].transform("count")
        df["benchmark_weight"] = 1.0 / n

    keep = ["date", "datetime", "sid", "securityid", "mid_price", "benchmark_weight"]
    df = df[keep].dropna(subset=["datetime", "sid", "mid_price"]).copy()
    df = df.drop_duplicates(["date", "datetime", "sid"], keep="last")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market-dir", required=True)
    ap.add_argument("--pricing-path", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--pred-col", default="pred_ret")
    ap.add_argument("--h10-mode", choices=["zero", "same"], default="zero")
    args = ap.parse_args()

    market_dir = Path(args.market_dir)
    pricing_path = Path(args.pricing_path)
    out_root = Path(args.out_root)
    h20_out = out_root / "h20_compat"
    h10_out = out_root / "h10_compat"
    h20_out.mkdir(parents=True, exist_ok=True)
    h10_out.mkdir(parents=True, exist_ok=True)

    pricing = read_pricing(pricing_path, args.pred_col)

    market_files = sorted(market_dir.glob("*.csv"))
    if not market_files:
        raise FileNotFoundError(market_dir)

    for mf in market_files:
        m = read_market_day(mf)
        d = int(m["date"].iloc[0])
        p = pricing[pricing["date"] == d].copy()

        x = m.merge(p, on=["date", "datetime", "sid"], how="left")
        miss = x["pred_ret"].isna().mean()
        x["pred_ret"] = x["pred_ret"].fillna(0.0)

        x["pred_ret_h20"] = x["pred_ret"]
        h20 = x.copy()

        h10 = x[["date", "datetime", "sid", "securityid"]].copy()
        if args.h10_mode == "same":
            h10["pred_ret_h10"] = x["pred_ret"]
        else:
            h10["pred_ret_h10"] = 0.0

        out20 = h20_out / f"optimizer_input_{d}.parquet"
        out10 = h10_out / f"optimizer_input_{d}.parquet"

        h20.to_parquet(out20, index=False)
        h10.to_parquet(out10, index=False)

        print(
            f"[DONE] {d} rows={len(x)} pred_missing={miss:.4f} "
            f"bench_sum_first={x.groupby('datetime')['benchmark_weight'].sum().iloc[0]:.6f}"
        )

    print("\nDONE")
    print("H20:", h20_out)
    print("H10:", h10_out)


if __name__ == "__main__":
    main()
