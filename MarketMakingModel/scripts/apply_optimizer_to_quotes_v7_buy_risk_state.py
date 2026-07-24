import argparse
from pathlib import Path
import pandas as pd
import numpy as np


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


def to_bool_series(s):
    if s.dtype == bool:
        return s.fillna(False)
    x = s.astype(str).str.lower().str.strip()
    return x.isin(["true", "1", "yes", "y", "t"])


def keep_top_ratio(raw_mask, score, keep_ratio):
    raw_mask = raw_mask.fillna(False)
    n = int(raw_mask.sum())

    if n == 0:
        return pd.Series(False, index=raw_mask.index)

    keep_ratio = max(0.0, min(1.0, float(keep_ratio)))

    if keep_ratio >= 1.0:
        return raw_mask.copy()

    if keep_ratio <= 0.0:
        return pd.Series(False, index=raw_mask.index)

    score_raw = score[raw_mask].fillna(-np.inf)

    kth = max(1, int(np.ceil(n * keep_ratio)))
    threshold = score_raw.nlargest(kth).min()

    return raw_mask & (score >= threshold)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--quotes",
        default="MarketMakingModel/outputs/quote_decisions/quote_decisions_mlp2_h60_202410_100_v5.csv",
    )
    parser.add_argument(
        "--optimizer",
        default="CrossSectionalOptimizer/outputs/signals/optimizer_signal_mlp2_h60_202410_100.csv",
    )
    parser.add_argument(
        "--output",
        default="MarketMakingModel/outputs/quote_decisions/quote_decisions_mlp2_h60_202410_100_v7_buy_risk_state_optimizer.csv",
    )

    parser.add_argument("--chunksize", type=int, default=1000000)

    # v6 core parameters
    parser.add_argument("--bid_keep_strong_alpha", type=float, default=1.00)
    parser.add_argument("--bid_keep_alpha_clipped", type=float, default=0.98)
    parser.add_argument("--bid_keep_high_volatility", type=float, default=0.88)
    parser.add_argument("--bid_keep_normal", type=float, default=0.83)
    parser.add_argument("--bid_keep_weak_alpha", type=float, default=0.68)
    parser.add_argument("--bid_keep_default", type=float, default=0.75)
    parser.add_argument("--ask_keep_ratio", type=float, default=0.55)

    args = parser.parse_args()

    quotes_path = Path(args.quotes)
    opt_path = Path(args.optimizer)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    bid_keep_map = {
        "strong_alpha": args.bid_keep_strong_alpha,
        "alpha_clipped": args.bid_keep_alpha_clipped,
        "high_volatility": args.bid_keep_high_volatility,
        "normal": args.bid_keep_normal,
        "weak_alpha": args.bid_keep_weak_alpha,
        "default": args.bid_keep_default,
    }
    print("v7 bid_keep_map:", bid_keep_map)

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

    opt = opt[
        [
            "datetime_key",
            "securityid_key",
            "target_weight",
            "quote_intensity",
            "bid_aggressiveness",
            "ask_aggressiveness",
            "optimizer_side",
            "optimizer_status",
        ]
    ]

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

    total_bid_protected = 0

    print("reading quotes by chunks:", quotes_path)

    for i, chunk in enumerate(pd.read_csv(quotes_path, chunksize=args.chunksize), 1):
        chunk.columns = [c.strip() for c in chunk.columns]

        for col in ["datetime", "securityid", "quote_bid", "quote_ask"]:
            if col not in chunk.columns:
                raise ValueError(f"missing quote column: {col}")

        chunk["datetime_key"] = parse_datetime_series(chunk["datetime"]).dt.floor("s")
        chunk["securityid_key"] = normalize_symbol(chunk["securityid"])

        df = chunk.merge(opt, on=["datetime_key", "securityid_key"], how="left")

        for c in [
            "target_weight",
            "quote_intensity",
            "bid_aggressiveness",
            "ask_aggressiveness",
        ]:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

        df["optimizer_side"] = df["optimizer_side"].fillna("missing")
        df["optimizer_status"] = df["optimizer_status"].fillna("missing")

        matched = df["optimizer_status"].ne("missing")

        raw_bid = to_bool_series(df["quote_bid"])
        raw_ask = to_bool_series(df["quote_ask"])

        df["quote_bid_raw"] = raw_bid
        df["quote_ask_raw"] = raw_ask

        # bid side: relaxed
        bid_score = df["bid_aggressiveness"].astype(float)

        # give a mild boost to positive target weights
        bid_score = bid_score + (df["target_weight"] > 0).astype(float) * 0.001

        if "risk_state" in df.columns:
            risk_state = df["risk_state"].astype(str).fillna("UNKNOWN")
        else:
            risk_state = pd.Series("UNKNOWN", index=df.index)

        bid_keep = raw_bid & (~matched)

        matched_bid = raw_bid & matched
        for st, ratio in bid_keep_map.items():
            if st == "default":
                continue
            m = matched_bid & (risk_state == st)
            bid_keep = bid_keep | keep_top_ratio(m, bid_score, ratio)

        known_states = set(k for k in bid_keep_map.keys() if k != "default")
        m_other = matched_bid & (~risk_state.isin(known_states))
        bid_keep = bid_keep | keep_top_ratio(m_other, bid_score, bid_keep_map["default"])

        # ask side: stricter
        ask_score = df["ask_aggressiveness"].astype(float)
        ask_score = ask_score + (df["target_weight"] < 0).astype(float) * 0.001

        ask_keep_by_rank = keep_top_ratio(raw_ask & matched, ask_score, args.ask_keep_ratio)
        ask_keep = raw_ask & (~matched | ask_keep_by_rank)

        df["quote_bid"] = bid_keep
        df["quote_ask"] = ask_keep

        df["optimizer_applied"] = matched
        df["v7_policy"] = "buy_risk_state_aware"
        df["v7_bid_score"] = bid_score
        df["v7_ask_score"] = ask_score
        df["v7_risk_state"] = risk_state

        total_rows += len(df)
        total_matched += matched.sum()

        total_bid_before += raw_bid.sum()
        total_ask_before += raw_ask.sum()
        total_any_before += (raw_bid | raw_ask).sum()

        total_bid_after += bid_keep.sum()
        total_ask_after += ask_keep.sum()
        total_any_after += (bid_keep | ask_keep).sum()

        total_bid_protected += 0

        df = df.drop(columns=["datetime_key", "securityid_key"])

        df.to_csv(out_path, mode="a", index=False, header=(i == 1))

        print(
            f"chunk={i}, rows={len(df)}, total={total_rows}, "
            f"matched_rate={total_matched / total_rows:.4f}, "
            f"bid_before={total_bid_before / total_rows:.4f}, "
            f"ask_before={total_ask_before / total_rows:.4f}, "
            f"bid_after={total_bid_after / total_rows:.4f}, "
            f"ask_after={total_ask_after / total_rows:.4f}, "
            f"any_after={total_any_after / total_rows:.4f}"
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

    print("retention:")
    print("bid retention:", total_bid_after / max(total_bid_before, 1))
    print("ask retention:", total_ask_after / max(total_ask_before, 1))
    print("v7 buy risk-state-aware done")


if __name__ == "__main__":
    main()
