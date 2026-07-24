import argparse
from pathlib import Path
import pandas as pd


def parse_datetime_series(s):
    x = s.astype(str).str.strip()
    out = pd.to_datetime(x, format="%Y%m%d_%H%M%S%f", errors="coerce")

    mask = out.isna()
    if mask.any():
        out.loc[mask] = pd.to_datetime(x[mask], errors="coerce")

    mask = out.isna()
    if mask.any():
        y = x[mask].str.replace(r"\D", "", regex=True)
        parsed = pd.Series(pd.NaT, index=y.index, dtype="datetime64[ns]")

        for length, fmt in [
            (8, "%Y%m%d"),
            (12, "%Y%m%d%H%M"),
            (14, "%Y%m%d%H%M%S"),
            (17, "%Y%m%d%H%M%S%f"),
        ]:
            m = y.str.len() == length
            if m.any():
                parsed.loc[m] = pd.to_datetime(y[m], format=fmt, errors="coerce")

        out.loc[mask] = parsed

    return out


def normalize_symbol(s):
    return s.astype(str).str.extract(r"(\d+)")[0].str.zfill(6)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quotes", default="MarketMakingModel/outputs/quote_decisions/quote_decisions_mlp2_h60_202410_100.csv")
    parser.add_argument("--optimizer", default="CrossSectionalOptimizer/outputs/signals/optimizer_signal_mlp2_h60_202410_100.csv")
    parser.add_argument("--output", default="MarketMakingModel/outputs/quote_decisions/quote_decisions_mlp2_h60_202410_100_with_optimizer.csv")
    parser.add_argument("--chunksize", type=int, default=1000000)
    parser.add_argument("--min_quote_intensity", type=float, default=0.0)
    args = parser.parse_args()

    quotes_path = Path(args.quotes)
    opt_path = Path(args.optimizer)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("reading optimizer:", opt_path)
    opt_cols = [
        "datetime",
        "securityid",
        "target_weight",
        "quote_intensity",
        "bid_aggressiveness",
        "ask_aggressiveness",
        "optimizer_side",
        "optimizer_status",
    ]

    opt = pd.read_csv(opt_path, usecols=opt_cols)
    opt["datetime_key"] = parse_datetime_series(opt["datetime"]).dt.floor("s")
    opt["securityid_key"] = normalize_symbol(opt["securityid"])

    opt = opt.dropna(subset=["datetime_key", "securityid_key"])
    opt = opt.drop_duplicates(["datetime_key", "securityid_key"], keep="last")

    keep = [
        "datetime_key",
        "securityid_key",
        "target_weight",
        "quote_intensity",
        "bid_aggressiveness",
        "ask_aggressiveness",
        "optimizer_side",
        "optimizer_status",
    ]
    opt = opt[keep]

    print("optimizer shape:", opt.shape)
    print("optimizer status:")
    print(opt["optimizer_status"].value_counts().head())

    if out_path.exists():
        out_path.unlink()

    total_rows = 0
    total_matched = 0
    total_bid_before = 0
    total_ask_before = 0
    total_any_before = 0
    total_bid_after = 0
    total_ask_after = 0
    total_any_after = 0

    print("reading quotes by chunks:", quotes_path)

    for i, chunk in enumerate(pd.read_csv(quotes_path, chunksize=args.chunksize), 1):
        chunk.columns = [c.strip() for c in chunk.columns]

        for col in ["datetime", "securityid", "quote_bid", "quote_ask"]:
            if col not in chunk.columns:
                raise ValueError(f"missing quote column: {col}")

        chunk["datetime_key"] = parse_datetime_series(chunk["datetime"]).dt.floor("s")
        chunk["securityid_key"] = normalize_symbol(chunk["securityid"])

        df = chunk.merge(opt, on=["datetime_key", "securityid_key"], how="left")

        for c in ["target_weight", "quote_intensity", "bid_aggressiveness", "ask_aggressiveness"]:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

        df["optimizer_side"] = df["optimizer_side"].fillna("missing")
        df["optimizer_status"] = df["optimizer_status"].fillna("missing")

        df["quote_bid_raw"] = df["quote_bid"].astype(bool)
        df["quote_ask_raw"] = df["quote_ask"].astype(bool)

        bid_active = df["bid_aggressiveness"] > args.min_quote_intensity
        ask_active = df["ask_aggressiveness"] > args.min_quote_intensity

        df["quote_bid"] = df["quote_bid_raw"] & bid_active
        df["quote_ask"] = df["quote_ask_raw"] & ask_active
        df["optimizer_applied"] = df["optimizer_status"].ne("missing")

        total_rows += len(df)
        total_matched += df["optimizer_applied"].sum()

        total_bid_before += df["quote_bid_raw"].sum()
        total_ask_before += df["quote_ask_raw"].sum()
        total_any_before += (df["quote_bid_raw"] | df["quote_ask_raw"]).sum()

        total_bid_after += df["quote_bid"].sum()
        total_ask_after += df["quote_ask"].sum()
        total_any_after += (df["quote_bid"] | df["quote_ask"]).sum()

        df = df.drop(columns=["datetime_key", "securityid_key"])

        df.to_csv(out_path, mode="a", index=False, header=(i == 1))

        print(
            f"chunk={i}, rows={len(df)}, total={total_rows}, "
            f"matched_rate={total_matched / total_rows:.4f}, "
            f"bid_before={total_bid_before / total_rows:.4f}, "
            f"ask_before={total_ask_before / total_rows:.4f}, "
            f"bid_after={total_bid_after / total_rows:.4f}, "
            f"ask_after={total_ask_after / total_rows:.4f}"
        )

    print("saved:", out_path)
    print("total rows:", total_rows)
    print("matched rate:", total_matched / total_rows)
    print("quote rates before:")
    print("bid:", total_bid_before / total_rows)
    print("ask:", total_ask_before / total_rows)
    print("any:", total_any_before / total_rows)
    print("quote rates after:")
    print("bid:", total_bid_after / total_rows)
    print("ask:", total_ask_after / total_rows)
    print("any:", total_any_after / total_rows)


if __name__ == "__main__":
    main()
