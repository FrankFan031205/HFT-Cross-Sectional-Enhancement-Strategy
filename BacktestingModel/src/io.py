from pathlib import Path

import yaml
import pandas as pd


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def parse_datetime_series(x, col_name="datetime"):
    """
    Support:
    1. 20241022_093000000
    2. 2024-10-22 09:30:00.000
    3. normal pandas datetime strings
    """
    if pd.api.types.is_datetime64_any_dtype(x):
        return x

    s = x.astype(str).str.strip()

    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")

    mask_compact = s.str.match(r"^\d{8}_\d{9}$", na=False)
    if mask_compact.any():
        out.loc[mask_compact] = pd.to_datetime(
            s.loc[mask_compact],
            format="%Y%m%d_%H%M%S%f",
            errors="coerce",
        )

    mask_compact_no_us = s.str.match(r"^\d{8}_\d{6}$", na=False)
    if mask_compact_no_us.any():
        out.loc[mask_compact_no_us] = pd.to_datetime(
            s.loc[mask_compact_no_us],
            format="%Y%m%d_%H%M%S",
            errors="coerce",
        )

    remain = out.isna()
    if remain.any():
        out.loc[remain] = pd.to_datetime(s.loc[remain], errors="coerce")

    bad = out.isna()
    if bad.any():
        examples = s.loc[bad].head(10).tolist()
        raise RuntimeError(f"Failed to parse {col_name}. Bad examples: {examples}")

    return out


def _standardize_securityid(df, col="securityid"):
    df[col] = df[col].astype(str).str.replace(".0", "", regex=False).str.zfill(6)
    return df


def _dt_key(s):
    x = parse_datetime_series(s)
    return x.dt.floor("ms").dt.strftime("%Y-%m-%d %H:%M:%S.%f").str[:-3]


def _make_key(df):
    return _dt_key(df["datetime"]) + "|" + df["securityid"].astype(str).str.zfill(6)


def load_quote_decisions(cfg):
    path = cfg["input"]["quote_decision_path"]
    qcfg = cfg["columns"]["quote"]

    df = pd.read_csv(path, low_memory=False)

    rename = {}
    for std_col, raw_col in qcfg.items():
        if std_col.startswith("fallback_"):
            continue
        if raw_col in df.columns:
            rename[raw_col] = std_col

    df = df.rename(columns=rename)

    if "datetime" not in df.columns:
        raise RuntimeError("quote_decision file missing datetime column.")
    if "securityid" not in df.columns:
        raise RuntimeError("quote_decision file missing securityid column.")
    if "quote_bid" not in df.columns:
        raise RuntimeError("quote_decision file missing quote_bid column.")
    if "quote_ask" not in df.columns:
        raise RuntimeError("quote_decision file missing quote_ask column.")

    df["datetime"] = parse_datetime_series(df["datetime"], "quote datetime")
    df = _standardize_securityid(df, "securityid")

    if "bid_quote_price" not in df.columns:
        fallback = qcfg.get("fallback_bid_quote_price", "bid1")
        if fallback not in df.columns:
            raise RuntimeError("Missing bid_quote_price and fallback bid1.")
        df["bid_quote_price"] = df[fallback]

    if "ask_quote_price" not in df.columns:
        fallback = qcfg.get("fallback_ask_quote_price", "ask1")
        if fallback not in df.columns:
            raise RuntimeError("Missing ask_quote_price and fallback ask1.")
        df["ask_quote_price"] = df[fallback]

    df = df.sort_values(["securityid", "datetime"]).reset_index(drop=True)
    return df


def _required_snapshot_cols(max_level=10):
    cols = ["datetime", "securityid"]

    for i in range(1, max_level + 1):
        cols += [
            f"bid{i}",
            f"ask{i}",
            f"bid{i}_volume",
            f"ask{i}_volume",
        ]

    optional = [
        "date",
        "mid_price",
        "spread",
        "spread_ticks",
        "limit_up_price",
        "limit_down_price",
        "hidden_factor_attention_h60",
        "microprice",
        "liquidity_state",
    ]

    return cols, optional


def load_snapshot_state_for_quotes(cfg, quotes):
    path = cfg["input"].get("snapshot_state_path", "")
    if not path:
        return quotes

    path = Path(path)
    if not path.exists():
        raise RuntimeError(f"snapshot_state_path not found: {path}")

    cache_path = cfg["output"].get("snapshot_cache_path", "")
    if cache_path and Path(cache_path).exists():
        print(f"[snapshot] loading small cache: {cache_path}")
        snap = pd.read_csv(cache_path, low_memory=False)
        snap["datetime"] = parse_datetime_series(snap["datetime"], "snapshot cache datetime")
        snap = _standardize_securityid(snap, "securityid")
        return merge_snapshot_into_quotes(quotes, snap)

    max_level = int(cfg["backtest"].get("max_book_level", 10))
    required, optional = _required_snapshot_cols(max_level=max_level)

    header = pd.read_csv(path, nrows=0)
    all_cols = set(header.columns)

    missing_required = [c for c in required if c not in all_cols]
    if missing_required:
        raise RuntimeError(f"snapshot file missing required columns: {missing_required}")

    usecols = required + [c for c in optional if c in all_cols]

    q = quotes[["datetime", "securityid"]].copy()
    q["datetime"] = parse_datetime_series(q["datetime"], "quote datetime")
    q = _standardize_securityid(q, "securityid")
    quote_keys = set(_make_key(q))

    chunksize = int(cfg["backtest"].get("snapshot_chunksize", 1000000))

    print(f"[snapshot] source: {path}")
    print(f"[snapshot] usecols={len(usecols)}")
    print(f"[snapshot] quote keys={len(quote_keys)}")
    print(f"[snapshot] chunksize={chunksize}")

    parts = []
    total_rows = 0
    matched_rows = 0

    for idx, chunk in enumerate(pd.read_csv(path, usecols=usecols, chunksize=chunksize, low_memory=False)):
        total_rows += len(chunk)

        chunk["datetime"] = parse_datetime_series(chunk["datetime"], "snapshot datetime")
        chunk = _standardize_securityid(chunk, "securityid")

        keys = _make_key(chunk)
        m = keys.isin(quote_keys)
        out = chunk.loc[m].copy()

        if len(out) > 0:
            parts.append(out)
            matched_rows += len(out)

        if idx % 10 == 0:
            print(f"[snapshot] chunk={idx}, scanned={total_rows}, matched={matched_rows}")

    if parts:
        snap = pd.concat(parts, ignore_index=True)
        snap = snap.drop_duplicates(["datetime", "securityid"])
        snap = snap.sort_values(["securityid", "datetime"]).reset_index(drop=True)
    else:
        snap = pd.DataFrame(columns=usecols)

    print(f"[snapshot] final matched rows={len(snap)}")

    if cache_path:
        save_csv(snap, cache_path)

    return merge_snapshot_into_quotes(quotes, snap)


def merge_snapshot_into_quotes(quotes, snapshot):
    q = quotes.copy()
    s = snapshot.copy()

    q["datetime"] = parse_datetime_series(q["datetime"], "quote datetime")
    s["datetime"] = parse_datetime_series(s["datetime"], "snapshot datetime")

    q = _standardize_securityid(q, "securityid")
    s = _standardize_securityid(s, "securityid")

    book_cols = []
    for i in range(1, 11):
        book_cols += [f"bid{i}", f"ask{i}", f"bid{i}_volume", f"ask{i}_volume"]

    missing_in_quotes = [c for c in book_cols if c not in q.columns]

    if not missing_in_quotes:
        print("[snapshot] quote file already has all book columns; no merge needed.")
        return q

    keep_cols = ["datetime", "securityid"] + [
        c for c in s.columns
        if c not in ["datetime", "securityid"] and c not in q.columns
    ]

    merged = q.merge(
        s[keep_cols],
        on=["datetime", "securityid"],
        how="left",
    )

    still_missing = [c for c in book_cols if c not in merged.columns]
    if still_missing:
        raise RuntimeError(f"Still missing book columns after snapshot merge: {still_missing}")

    null_ratio = merged[book_cols].isna().mean().mean()
    print(f"[snapshot] merged quotes shape={merged.shape}, avg book null ratio={null_ratio:.6f}")

    if null_ratio > 0.5:
        print("[warning] More than 50% book values are missing after exact datetime merge.")
        print("[warning] Check datetime precision between quote_decision and snapshot csv.")

    return merged


def load_trade_replay_from_csv(path):
    df = pd.read_csv(path, low_memory=False)
    df["datetime"] = parse_datetime_series(df["datetime"], "trade datetime")
    df = _standardize_securityid(df, "securityid")
    return df.sort_values(["securityid", "datetime"]).reset_index(drop=True)


def standardize_trades(df, cfg):
    tcfg = cfg["columns"]["trade"]

    rename = {}
    for std_col, raw_col in tcfg.items():
        if raw_col in df.columns:
            rename[raw_col] = std_col

    df = df.rename(columns=rename)

    required = ["datetime", "securityid", "price", "qty", "side"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"trade data missing columns: {missing}")

    df["datetime"] = parse_datetime_series(df["datetime"], "trade datetime")
    df = _standardize_securityid(df, "securityid")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    # Trade DB stores price as integer cents, e.g. 220 -> 2.20.
    # But cached trade csv may already be scaled. Avoid double scaling.
    trade_price_scale = float(cfg.get("data_scale", {}).get("trade_price_scale", 1.0))
    if trade_price_scale != 1.0:
        med_price = df["price"].dropna().median()
        if pd.notna(med_price) and med_price > 50:
            df["price"] = df["price"] / trade_price_scale

    df["qty"] = pd.to_numeric(df["qty"], errors="coerce")
    df = df.dropna(subset=["datetime", "securityid", "price", "qty", "side"])
    return df.sort_values(["securityid", "datetime"]).reset_index(drop=True)


def save_csv(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"saved: {path}, rows={len(df)}")
