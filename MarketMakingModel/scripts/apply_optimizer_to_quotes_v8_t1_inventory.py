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


def clean_state(s):
    return s.where(s.notna(), "UNKNOWN").astype(str)


def keep_top_ratio(raw_mask, score, keep_ratio):
    raw_mask = raw_mask.fillna(False)
    idx = raw_mask[raw_mask].index
    out = pd.Series(False, index=raw_mask.index)

    n = len(idx)
    if n == 0:
        return out

    keep_ratio = max(0.0, min(1.0, float(keep_ratio)))
    if keep_ratio >= 1.0:
        out.loc[idx] = True
        return out
    if keep_ratio <= 0.0:
        return out

    k = max(1, int(np.ceil(n * keep_ratio)))
    score_raw = score.loc[idx].fillna(-np.inf)
    selected = score_raw.sort_values(ascending=False, kind="mergesort").head(k).index
    out.loc[selected] = True
    return out


def build_bid_keep(raw_bid, matched, risk_state, pos_ratio, bid_score, args):
    keep = pd.Series(False, index=raw_bid.index)

    # unmatched rows keep original quote to avoid accidental deletion
    keep |= raw_bid & (~matched)

    matched_bid = raw_bid & matched

    low_pos = pos_ratio < args.mid_pos_threshold
    mid_pos = (pos_ratio >= args.mid_pos_threshold) & (pos_ratio < args.high_pos_threshold)
    high_pos = (pos_ratio >= args.high_pos_threshold) & (pos_ratio < args.max_pos_threshold)
    blocked_pos = pos_ratio >= args.max_pos_threshold

    maps = {
        "low": {
            "strong_alpha": args.low_strong_alpha,
            "alpha_clipped": args.low_alpha_clipped,
            "high_volatility": args.low_high_volatility,
            "normal": args.low_normal,
            "weak_alpha": args.low_weak_alpha,
            "default": args.low_default,
        },
        "mid": {
            "strong_alpha": args.mid_strong_alpha,
            "alpha_clipped": args.mid_alpha_clipped,
            "high_volatility": args.mid_high_volatility,
            "normal": args.mid_normal,
            "weak_alpha": args.mid_weak_alpha,
            "default": args.mid_default,
        },
        "high": {
            "strong_alpha": args.high_strong_alpha,
            "alpha_clipped": args.high_alpha_clipped,
            "high_volatility": args.high_high_volatility,
            "normal": args.high_normal,
            "weak_alpha": args.high_weak_alpha,
            "default": args.high_default,
        },
    }

    bucket_masks = {
        "low": low_pos,
        "mid": mid_pos,
        "high": high_pos,
    }

    for bucket_name, bucket_mask in bucket_masks.items():
        ratio_map = maps[bucket_name]
        base = matched_bid & bucket_mask & (~blocked_pos)

        for st, ratio in ratio_map.items():
            if st == "default":
                continue
            m = base & (risk_state == st)
            keep |= keep_top_ratio(m, bid_score, ratio)

        known = set(k for k in ratio_map.keys() if k != "default")
        other = base & (~risk_state.isin(known))
        keep |= keep_top_ratio(other, bid_score, ratio_map["default"])

    return keep


def build_ask_keep(raw_ask, matched, sellable_ok, pos_ratio, ask_score, args):
    keep = pd.Series(False, index=raw_ask.index)

    # unmatched rows keep original ask only if sellable
    keep |= raw_ask & (~matched) & sellable_ok

    matched_ask = raw_ask & matched & sellable_ok

    high_inventory = pos_ratio >= args.ask_relax_pos_threshold
    normal_inventory = ~high_inventory

    # normal inventory: strict SELL filter
    keep |= keep_top_ratio(
        matched_ask & normal_inventory,
        ask_score,
        args.ask_keep_normal_inventory,
    )

    # high inventory: relax ask to reduce inventory
    keep |= keep_top_ratio(
        matched_ask & high_inventory,
        ask_score,
        args.ask_keep_high_inventory,
    )

    return keep


def get_position_ratio(df, args):
    if args.position_ratio_col in df.columns:
        x = pd.to_numeric(df[args.position_ratio_col], errors="coerce").fillna(0.0)
        return x.clip(lower=0.0)

    if args.position_col in df.columns and args.max_position > 0:
        pos = pd.to_numeric(df[args.position_col], errors="coerce").fillna(0.0)
        return (pos / float(args.max_position)).clip(lower=0.0)

    return pd.Series(0.0, index=df.index)


def get_sellable_position(df, args):
    # If original quote_ask is already feasibility-filtered, do not block ask again here.
    if getattr(args, "trust_raw_ask_feasibility", False):
        return pd.Series(np.inf, index=df.index)

    if args.sellable_position_col in df.columns:
        x = pd.to_numeric(df[args.sellable_position_col], errors="coerce").fillna(0.0)
        # If the column exists but is unusable/all zero, avoid killing all asks unless strict_t1 is requested.
        if (x > 0).mean() == 0 and not args.strict_t1:
            return pd.Series(np.inf, index=df.index)
        return x

    if args.strict_t1:
        raise ValueError(
            f"strict_t1=True but sellable_position column not found: {args.sellable_position_col}"
        )

    if args.position_col in df.columns and args.use_position_as_sellable:
        x = pd.to_numeric(df[args.position_col], errors="coerce").fillna(0.0).clip(lower=0.0)
        if (x > 0).mean() == 0 and not args.strict_t1:
            return pd.Series(np.inf, index=df.index)
        return x

    # if no reliable sellable info, do not block ask
    return pd.Series(np.inf, index=df.index)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--quotes", default="MarketMakingModel/outputs/quote_decisions/quote_decisions_mlp2_h60_202410_100_v5.csv")
    parser.add_argument("--optimizer", default="CrossSectionalOptimizer/outputs/signals/optimizer_signal_mlp2_h60_202410_100.csv")
    parser.add_argument("--output", default="MarketMakingModel/outputs/quote_decisions/quote_decisions_mlp2_h60_202410_100_v8_t1_inventory_optimizer.csv")
    parser.add_argument("--chunksize", type=int, default=1000000)

    # inventory columns
    parser.add_argument("--position_col", default="position")
    parser.add_argument("--position_ratio_col", default="position_ratio")
    parser.add_argument("--sellable_position_col", default="sellable_position")
    parser.add_argument("--max_position", type=float, default=1.0)
    parser.add_argument("--strict_t1", action="store_true")
    parser.add_argument("--use_position_as_sellable", action="store_true")
    parser.add_argument("--trust_raw_ask_feasibility", action="store_true")

    # position buckets
    parser.add_argument("--mid_pos_threshold", type=float, default=0.40)
    parser.add_argument("--high_pos_threshold", type=float, default=0.70)
    parser.add_argument("--max_pos_threshold", type=float, default=0.90)

    # low inventory: close to v7
    parser.add_argument("--low_strong_alpha", type=float, default=1.00)
    parser.add_argument("--low_alpha_clipped", type=float, default=0.98)
    parser.add_argument("--low_high_volatility", type=float, default=0.88)
    parser.add_argument("--low_normal", type=float, default=0.83)
    parser.add_argument("--low_weak_alpha", type=float, default=0.68)
    parser.add_argument("--low_default", type=float, default=0.75)

    # medium inventory: reduce buy
    parser.add_argument("--mid_strong_alpha", type=float, default=1.00)
    parser.add_argument("--mid_alpha_clipped", type=float, default=0.95)
    parser.add_argument("--mid_high_volatility", type=float, default=0.70)
    parser.add_argument("--mid_normal", type=float, default=0.55)
    parser.add_argument("--mid_weak_alpha", type=float, default=0.35)
    parser.add_argument("--mid_default", type=float, default=0.45)

    # high inventory: strongly reduce buy
    parser.add_argument("--high_strong_alpha", type=float, default=0.80)
    parser.add_argument("--high_alpha_clipped", type=float, default=0.60)
    parser.add_argument("--high_high_volatility", type=float, default=0.40)
    parser.add_argument("--high_normal", type=float, default=0.20)
    parser.add_argument("--high_weak_alpha", type=float, default=0.05)
    parser.add_argument("--high_default", type=float, default=0.10)

    # ask side
    parser.add_argument("--ask_keep_normal_inventory", type=float, default=0.55)
    parser.add_argument("--ask_keep_high_inventory", type=float, default=0.75)
    parser.add_argument("--ask_relax_pos_threshold", type=float, default=0.70)

    args = parser.parse_args()

    quotes_path = Path(args.quotes)
    opt_path = Path(args.optimizer)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("v8 policy: T+1 inventory-aware quote controller")
    print("quotes:", quotes_path)
    print("optimizer:", opt_path)
    print("output:", out_path)
    print("strict_t1:", args.strict_t1)
    print("use_position_as_sellable:", args.use_position_as_sellable)

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

    print("reading optimizer")
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

    total_ask_blocked_no_sellable = 0
    total_bid_blocked_high_pos = 0

    print("reading quotes by chunks")

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

        matched = df["optimizer_status"].ne("missing")

        raw_bid = to_bool_series(df["quote_bid"])
        raw_ask = to_bool_series(df["quote_ask"])

        df["quote_bid_raw"] = raw_bid
        df["quote_ask_raw"] = raw_ask

        if "risk_state" in df.columns:
            risk_state = clean_state(df["risk_state"])
        else:
            risk_state = pd.Series("UNKNOWN", index=df.index)

        pos_ratio = get_position_ratio(df, args)
        sellable_position = get_sellable_position(df, args)
        sellable_ok = sellable_position > 0

        bid_score = df["bid_aggressiveness"].astype(float)
        bid_score += (df["target_weight"] > 0).astype(float) * 0.001

        ask_score = df["ask_aggressiveness"].astype(float)
        ask_score += (df["target_weight"] < 0).astype(float) * 0.001
        ask_score += pos_ratio.clip(lower=0.0) * 0.001

        if "alpha_ticks" in df.columns:
            alpha_ticks = pd.to_numeric(df["alpha_ticks"], errors="coerce").fillna(0.0)
            bid_score += alpha_ticks.clip(lower=0.0) * 0.0001
            ask_score += (-alpha_ticks).clip(lower=0.0) * 0.0001

        bid_keep = build_bid_keep(raw_bid, matched, risk_state, pos_ratio, bid_score, args)
        ask_keep = build_ask_keep(raw_ask, matched, sellable_ok, pos_ratio, ask_score, args)

        high_pos_block = raw_bid & matched & (pos_ratio >= args.max_pos_threshold)
        no_sellable_block = raw_ask & matched & (~sellable_ok)

        df["quote_bid"] = bid_keep
        df["quote_ask"] = ask_keep

        df["optimizer_applied"] = matched
        df["v8_policy"] = "t1_inventory_aware"
        df["v8_position_ratio"] = pos_ratio
        df["v8_sellable_position"] = sellable_position.replace(np.inf, np.nan)
        df["v8_sellable_ok"] = sellable_ok
        df["v8_bid_score"] = bid_score
        df["v8_ask_score"] = ask_score
        df["v8_risk_state"] = risk_state
        df["v8_bid_blocked_high_pos"] = high_pos_block
        df["v8_ask_blocked_no_sellable"] = no_sellable_block

        total_rows += len(df)
        total_matched += matched.sum()

        total_bid_before += raw_bid.sum()
        total_ask_before += raw_ask.sum()
        total_any_before += (raw_bid | raw_ask).sum()

        total_bid_after += bid_keep.sum()
        total_ask_after += ask_keep.sum()
        total_any_after += (bid_keep | ask_keep).sum()

        total_bid_blocked_high_pos += high_pos_block.sum()
        total_ask_blocked_no_sellable += no_sellable_block.sum()

        df = df.drop(columns=["datetime_key", "securityid_key"])
        df.to_csv(out_path, mode="a", index=False, header=(i == 1))

        print(
            f"chunk={i}, rows={len(df)}, total={total_rows}, "
            f"matched_rate={total_matched / total_rows:.4f}, "
            f"bid_before={total_bid_before / total_rows:.4f}, "
            f"ask_before={total_ask_before / total_rows:.4f}, "
            f"bid_after={total_bid_after / total_rows:.4f}, "
            f"ask_after={total_ask_after / total_rows:.4f}, "
            f"any_after={total_any_after / total_rows:.4f}, "
            f"bid_high_pos_block={total_bid_blocked_high_pos}, "
            f"ask_no_sellable_block={total_ask_blocked_no_sellable}"
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

    print("inventory blocks:")
    print("bid blocked by high position:", total_bid_blocked_high_pos)
    print("ask blocked by no sellable inventory:", total_ask_blocked_no_sellable)


if __name__ == "__main__":
    main()
