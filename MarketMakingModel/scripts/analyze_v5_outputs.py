import math
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd


FILES = {
    "attention": "outputs/quote_decisions/quote_decisions_attention_h60_202410_100_v5.csv",
    "mlp2": "outputs/quote_decisions/quote_decisions_mlp2_h60_202410_100_v5.csv",
    "mlp60": "outputs/quote_decisions/quote_decisions_mlp60_h60_202410_100_v5.csv",
}

CHUNKSIZE = 500000
SAMPLE_PER_CHUNK = 30000

NUM_COLS = [
    "raw_pred",
    "calibrated_pred",
    "model_alpha_ticks",
    "alpha_ticks",
    "microprice_shift_ticks",
    "book_pressure_shift_ticks",
    "trade_pressure_shift_ticks",
    "cancel_pressure_shift_ticks",
    "microstructure_fair_price",
    "fair_price",
    "quote_fair_price",
    "bid_edge",
    "ask_edge",
    "bid_adverse_buffer_ticks",
    "ask_adverse_buffer_ticks",
    "bid_risk_score",
    "ask_risk_score",
]

KEY_NAN_COLS = [
    "raw_pred",
    "calibrated_pred",
    "alpha_ticks",
    "microstructure_fair_price",
    "fair_price",
    "quote_fair_price",
    "bid_edge",
    "ask_edge",
]

BAD_FILTER_STATES = {
    "before_trading_time",
    "after_trading_time",
    "missing_book",
    "missing_prediction",
    "invalid_price",
    "crossed_or_locked_book",
    "invalid_spread",
    "spread_too_small",
    "spread_too_large",
    "low_bid_liquidity",
    "low_ask_liquidity",
    "near_limit_up",
    "near_limit_down",
}


def init_numeric_stats():
    return {
        "count": 0,
        "sum": 0.0,
        "sumsq": 0.0,
        "min": math.inf,
        "max": -math.inf,
    }


def update_numeric_stats(stats, s):
    x = pd.to_numeric(s, errors="coerce").dropna()
    if len(x) == 0:
        return

    arr = x.to_numpy(dtype=float)

    stats["count"] += len(arr)
    stats["sum"] += float(arr.sum())
    stats["sumsq"] += float((arr * arr).sum())
    stats["min"] = min(stats["min"], float(arr.min()))
    stats["max"] = max(stats["max"], float(arr.max()))


def finalize_numeric_stats(stats, sample=None):
    n = stats["count"]
    if n == 0:
        return {
            "count": 0,
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "25%": np.nan,
            "50%": np.nan,
            "75%": np.nan,
            "max": np.nan,
        }

    mean = stats["sum"] / n
    var = max(stats["sumsq"] / n - mean * mean, 0.0)
    std = var ** 0.5

    if sample is not None and len(sample) > 0:
        q25, q50, q75 = np.nanpercentile(sample, [25, 50, 75])
    else:
        q25 = q50 = q75 = np.nan

    return {
        "count": n,
        "mean": mean,
        "std": std,
        "min": stats["min"],
        "25%": q25,
        "50%": q50,
        "75%": q75,
        "max": stats["max"],
    }


def print_section(title):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def analyze_one(name, path):
    path = Path(path)
    if not path.exists():
        print(f"[{name}] missing file: {path}")
        return None

    print_section(f"Analyzing {name}: {path}")

    header = pd.read_csv(path, nrows=0)
    all_cols = list(header.columns)

    usecols = []
    for c in [
        "datetime",
        "securityid",
        "quote_bid",
        "quote_ask",
        "risk_state",
        "alpha_bucket",
        "volatility_regime",
        "liquidity_state",
        "quote_style",
    ] + NUM_COLS:
        if c in all_cols and c not in usecols:
            usecols.append(c)

    total = 0
    quote_bid_true = 0
    quote_ask_true = 0

    combo_counts = defaultdict(int)
    risk_counts = defaultdict(int)
    alpha_bucket_counts = defaultdict(int)
    vol_regime_counts = defaultdict(int)
    liq_state_counts = defaultdict(int)
    quote_style_counts = defaultdict(int)

    nan_all_counts = defaultdict(int)
    nan_tradable_counts = defaultdict(int)
    nan_quoted_counts = defaultdict(int)

    tradable_count = 0
    quoted_count = 0
    quoted_nan_fair_edge_count = 0

    numeric_stats = {c: init_numeric_stats() for c in NUM_COLS if c in all_cols}

    alpha_bid_stats = init_numeric_stats()
    alpha_ask_stats = init_numeric_stats()

    bid_edge_true_stats = init_numeric_stats()
    bid_edge_false_stats = init_numeric_stats()
    ask_edge_true_stats = init_numeric_stats()
    ask_edge_false_stats = init_numeric_stats()

    sample_store = {c: [] for c in NUM_COLS if c in all_cols}
    alpha_bid_sample = []
    alpha_ask_sample = []
    bid_edge_true_sample = []
    bid_edge_false_sample = []
    ask_edge_true_sample = []
    ask_edge_false_sample = []

    for chunk_id, df in enumerate(pd.read_csv(path, usecols=usecols, chunksize=CHUNKSIZE)):
        n = len(df)
        total += n

        qb = df["quote_bid"].astype(bool) if "quote_bid" in df.columns else pd.Series(False, index=df.index)
        qa = df["quote_ask"].astype(bool) if "quote_ask" in df.columns else pd.Series(False, index=df.index)

        quote_bid_true += int(qb.sum())
        quote_ask_true += int(qa.sum())

        for k, v in pd.crosstab(qb, qa).stack().items():
            combo_counts[(bool(k[0]), bool(k[1]))] += int(v)

        if "risk_state" in df.columns:
            for k, v in df["risk_state"].value_counts(dropna=False).items():
                risk_counts[str(k)] += int(v)

            tradable_mask = ~df["risk_state"].isin(BAD_FILTER_STATES)
        else:
            tradable_mask = pd.Series(True, index=df.index)

        quoted_mask = qb | qa
        tradable_count += int(tradable_mask.sum())
        quoted_count += int(quoted_mask.sum())

        for col, counter in [
            ("alpha_bucket", alpha_bucket_counts),
            ("volatility_regime", vol_regime_counts),
            ("liquidity_state", liq_state_counts),
            ("quote_style", quote_style_counts),
        ]:
            if col in df.columns:
                for k, v in df[col].value_counts(dropna=False).items():
                    counter[str(k)] += int(v)

        for col in KEY_NAN_COLS:
            if col in df.columns:
                nan_all_counts[col] += int(df[col].isna().sum())
                nan_tradable_counts[col] += int(df.loc[tradable_mask, col].isna().sum())
                nan_quoted_counts[col] += int(df.loc[quoted_mask, col].isna().sum())

        fair_edge_cols = [c for c in ["fair_price", "quote_fair_price", "bid_edge", "ask_edge"] if c in df.columns]
        if fair_edge_cols:
            quoted_nan_fair_edge_count += int(df.loc[quoted_mask, fair_edge_cols].isna().any(axis=1).sum())

        for c in numeric_stats:
            update_numeric_stats(numeric_stats[c], df[c])
            s = pd.to_numeric(df[c], errors="coerce").dropna()
            if len(s):
                sample_store[c].append(s.sample(min(len(s), SAMPLE_PER_CHUNK), random_state=chunk_id).to_numpy())

        if "alpha_ticks" in df.columns:
            s_bid = pd.to_numeric(df.loc[qb, "alpha_ticks"], errors="coerce")
            s_ask = pd.to_numeric(df.loc[qa, "alpha_ticks"], errors="coerce")
            update_numeric_stats(alpha_bid_stats, s_bid)
            update_numeric_stats(alpha_ask_stats, s_ask)
            s_bid = s_bid.dropna()
            s_ask = s_ask.dropna()
            if len(s_bid):
                alpha_bid_sample.append(s_bid.sample(min(len(s_bid), SAMPLE_PER_CHUNK), random_state=chunk_id).to_numpy())
            if len(s_ask):
                alpha_ask_sample.append(s_ask.sample(min(len(s_ask), SAMPLE_PER_CHUNK), random_state=chunk_id).to_numpy())

        if "bid_edge" in df.columns:
            update_numeric_stats(bid_edge_true_stats, df.loc[qb, "bid_edge"])
            update_numeric_stats(bid_edge_false_stats, df.loc[~qb, "bid_edge"])
            bt = pd.to_numeric(df.loc[qb, "bid_edge"], errors="coerce").dropna()
            bf = pd.to_numeric(df.loc[~qb, "bid_edge"], errors="coerce").dropna()
            if len(bt):
                bid_edge_true_sample.append(bt.sample(min(len(bt), SAMPLE_PER_CHUNK), random_state=chunk_id).to_numpy())
            if len(bf):
                bid_edge_false_sample.append(bf.sample(min(len(bf), SAMPLE_PER_CHUNK), random_state=chunk_id).to_numpy())

        if "ask_edge" in df.columns:
            update_numeric_stats(ask_edge_true_stats, df.loc[qa, "ask_edge"])
            update_numeric_stats(ask_edge_false_stats, df.loc[~qa, "ask_edge"])
            at = pd.to_numeric(df.loc[qa, "ask_edge"], errors="coerce").dropna()
            af = pd.to_numeric(df.loc[~qa, "ask_edge"], errors="coerce").dropna()
            if len(at):
                ask_edge_true_sample.append(at.sample(min(len(at), SAMPLE_PER_CHUNK), random_state=chunk_id).to_numpy())
            if len(af):
                ask_edge_false_sample.append(af.sample(min(len(af), SAMPLE_PER_CHUNK), random_state=chunk_id).to_numpy())

        if chunk_id % 10 == 0:
            print(f"{name}: chunk={chunk_id}, rows={total}")

    def cat_sample(lst):
        if not lst:
            return np.array([])
        return np.concatenate(lst)

    print("\nBasic:")
    print("rows:", total)
    print("quote_bid_rate:", quote_bid_true / total if total else np.nan)
    print("quote_ask_rate:", quote_ask_true / total if total else np.nan)
    print("tradable_like_rate:", tradable_count / total if total else np.nan)
    print("quoted_any_rate:", quoted_count / total if total else np.nan)
    print("quoted rows with NaN fair/edge:", quoted_nan_fair_edge_count)

    print("\nQuote combination:")
    combo = pd.Series(combo_counts)
    print(combo.sort_index())

    print("\nRisk state top:")
    print(pd.Series(risk_counts).sort_values(ascending=False).head(20))

    print("\nAlpha bucket:")
    print(pd.Series(alpha_bucket_counts).sort_values(ascending=False))

    print("\nVolatility regime:")
    print(pd.Series(vol_regime_counts).sort_values(ascending=False))

    print("\nLiquidity state:")
    print(pd.Series(liq_state_counts).sort_values(ascending=False).head(20))

    print("\nQuote style:")
    print(pd.Series(quote_style_counts).sort_values(ascending=False).head(20))

    print("\nMissing ratio all rows:")
    print(pd.Series({c: nan_all_counts[c] / total for c in KEY_NAN_COLS if c in all_cols}).sort_values(ascending=False))

    print("\nMissing ratio tradable-like rows:")
    print(pd.Series({c: nan_tradable_counts[c] / max(tradable_count, 1) for c in KEY_NAN_COLS if c in all_cols}).sort_values(ascending=False))

    print("\nMissing ratio quoted rows:")
    print(pd.Series({c: nan_quoted_counts[c] / max(quoted_count, 1) for c in KEY_NAN_COLS if c in all_cols}).sort_values(ascending=False))

    numeric_rows = []
    for c, st in numeric_stats.items():
        sample = cat_sample(sample_store[c])
        numeric_rows.append((c, finalize_numeric_stats(st, sample)))

    numeric_df = pd.DataFrame({c: d for c, d in numeric_rows}).T
    print("\nNumeric summary:")
    print(numeric_df)

    print("\nAlpha when quote_bid=True:")
    print(pd.Series(finalize_numeric_stats(alpha_bid_stats, cat_sample(alpha_bid_sample))))

    print("\nAlpha when quote_ask=True:")
    print(pd.Series(finalize_numeric_stats(alpha_ask_stats, cat_sample(alpha_ask_sample))))

    print("\nBid edge quoted=False:")
    print(pd.Series(finalize_numeric_stats(bid_edge_false_stats, cat_sample(bid_edge_false_sample))))

    print("\nBid edge quoted=True:")
    print(pd.Series(finalize_numeric_stats(bid_edge_true_stats, cat_sample(bid_edge_true_sample))))

    print("\nAsk edge quoted=False:")
    print(pd.Series(finalize_numeric_stats(ask_edge_false_stats, cat_sample(ask_edge_false_sample))))

    print("\nAsk edge quoted=True:")
    print(pd.Series(finalize_numeric_stats(ask_edge_true_stats, cat_sample(ask_edge_true_sample))))

    summary = {
        "model": name,
        "rows": total,
        "quote_bid_rate": quote_bid_true / total if total else np.nan,
        "quote_ask_rate": quote_ask_true / total if total else np.nan,
        "quoted_any_rate": quoted_count / total if total else np.nan,
        "tradable_like_rate": tradable_count / total if total else np.nan,
        "quoted_nan_fair_edge_count": quoted_nan_fair_edge_count,
        "alpha_bid_mean": alpha_bid_stats["sum"] / alpha_bid_stats["count"] if alpha_bid_stats["count"] else np.nan,
        "alpha_ask_mean": alpha_ask_stats["sum"] / alpha_ask_stats["count"] if alpha_ask_stats["count"] else np.nan,
        "model_alpha_mean": numeric_df.loc["model_alpha_ticks", "mean"] if "model_alpha_ticks" in numeric_df.index else np.nan,
        "final_alpha_mean": numeric_df.loc["alpha_ticks", "mean"] if "alpha_ticks" in numeric_df.index else np.nan,
        "microprice_shift_mean": numeric_df.loc["microprice_shift_ticks", "mean"] if "microprice_shift_ticks" in numeric_df.index else np.nan,
        "book_shift_mean": numeric_df.loc["book_pressure_shift_ticks", "mean"] if "book_pressure_shift_ticks" in numeric_df.index else np.nan,
        "trade_shift_mean": numeric_df.loc["trade_pressure_shift_ticks", "mean"] if "trade_pressure_shift_ticks" in numeric_df.index else np.nan,
        "cancel_shift_mean": numeric_df.loc["cancel_pressure_shift_ticks", "mean"] if "cancel_pressure_shift_ticks" in numeric_df.index else np.nan,
    }

    return summary


def main():
    summaries = []
    for name, path in FILES.items():
        s = analyze_one(name, path)
        if s is not None:
            summaries.append(s)

    print_section("MODEL COMPARISON SUMMARY")
    summary_df = pd.DataFrame(summaries)
    print(summary_df)

    out = Path("outputs/quote_decisions/v5_model_summary.csv")
    summary_df.to_csv(out, index=False)
    print("\nsaved:", out)


if __name__ == "__main__":
    main()
