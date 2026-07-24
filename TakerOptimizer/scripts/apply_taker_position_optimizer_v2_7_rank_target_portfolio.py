import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import yaml


def read_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def mkdir_parent(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def log_print(msg, log_path=None):
    print(msg)
    if log_path:
        mkdir_parent(log_path)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(str(msg) + "\n")


def unique_list(xs):
    ans = []
    for x in xs:
        if x is not None and x not in ans:
            ans.append(x)
    return ans


def normalize_date_series(s):
    raw = s.astype("string").str.strip()
    ans = pd.Series(index=s.index, dtype="float64")

    m = raw.str.match(r"^\d{8}$", na=False)
    if m.any():
        ans.loc[m] = pd.to_numeric(raw.loc[m], errors="coerce")

    remain = ans.isna()
    if remain.any():
        dt = pd.to_datetime(raw.loc[remain], errors="coerce")
        good = dt.notna()
        if good.any():
            ans.loc[dt.loc[good].index] = dt.loc[good].dt.strftime("%Y%m%d").astype("int64")

    remain = ans.isna()
    if remain.any():
        num = pd.to_numeric(raw.loc[remain], errors="coerce")
        ans.loc[remain] = num

    if ans.isna().any():
        bad = raw.loc[ans.isna()].head(5).tolist()
        raise ValueError(f"failed to parse date values: {bad}")

    return ans.astype("int64")


def parse_datetime_series(s):
    raw = s.astype("string").str.strip()
    dt = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")

    m1 = raw.str.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+$", na=False)
    if m1.any():
        dt.loc[m1] = pd.to_datetime(raw.loc[m1], format="%Y-%m-%d %H:%M:%S.%f", errors="coerce")

    m2 = raw.str.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$", na=False)
    if m2.any():
        dt.loc[m2] = pd.to_datetime(raw.loc[m2], format="%Y-%m-%d %H:%M:%S", errors="coerce")

    m3 = raw.str.match(r"^\d{8}_\d{6,12}$", na=False)
    if m3.any():
        dt.loc[m3] = pd.to_datetime(raw.loc[m3], format="%Y%m%d_%H%M%S%f", errors="coerce")

    remain = dt.isna()
    if remain.any():
        compact = (
            raw.loc[remain]
            .str.replace("_", "", regex=False)
            .str.replace("-", "", regex=False)
            .str.replace(":", "", regex=False)
            .str.replace(" ", "", regex=False)
            .str.replace(".", "", regex=False)
        )

        parsed = pd.to_datetime(compact, format="%Y%m%d%H%M%S%f", errors="coerce")
        mask2 = parsed.isna()
        if mask2.any():
            parsed2 = pd.to_datetime(compact.loc[mask2], format="%Y%m%d%H%M%S", errors="coerce")
            parsed.loc[mask2] = parsed2

        dt.loc[remain] = parsed

    if dt.isna().any():
        bad = raw.loc[dt.isna()].head(5).tolist()
        raise ValueError(f"failed to parse datetime values: {bad}")

    return dt


def normalize_symbol_series(s):
    raw = s.astype("string").str.strip()
    extracted = raw.str.extract(r"(\d+)")[0]
    num = pd.to_numeric(extracted, errors="coerce")
    if num.notna().all():
        return num.astype("int64")
    return raw


def floor_qty_to_lot(qty, lot_size):
    if not np.isfinite(qty) or qty <= 0:
        return 0
    if lot_size > 1:
        return int(np.floor(qty / lot_size) * lot_size)
    return int(np.floor(qty))


def is_trade_minute(minute, ocfg):
    t = pd.Timestamp(minute).strftime("%H:%M:%S")
    before = ocfg.get("block_before_time")
    after = ocfg.get("block_after_time")

    if before and t < str(before):
        return False
    if after and t > str(after):
        return False
    return True


def is_rebalance_minute(minute, ocfg):
    interval = int(ocfg.get("rebalance_interval_min", 1))
    if interval <= 1:
        return True
    t = pd.Timestamp(minute)
    return (t.hour * 60 + t.minute) % interval == 0


def load_minute_market(cfg):
    data_cfg = cfg["data"]
    c = cfg["columns"]
    ocfg = cfg["optimizer"]

    path = Path(data_cfg["market_data_path"])
    if not path.exists():
        raise FileNotFoundError(f"market_data_path not found: {path}")

    header = pd.read_csv(path, nrows=0)
    cols = header.columns.tolist()

    needed = unique_list([
        c["date_col"],
        c["datetime_col"],
        c["symbol_col"],
        c["price_col"],
        c.get("bid_col"),
        c.get("ask_col"),
        c.get("spread_col"),
        c.get("bid_volume_col"),
        c.get("ask_volume_col"),
        c.get("limit_up_col"),
        c.get("limit_down_col"),
    ])

    usecols = [x for x in needed if x in cols]

    missing = [x for x in [c["date_col"], c["datetime_col"], c["symbol_col"], c["price_col"]] if x not in cols]
    if missing:
        raise ValueError(f"market file missing required columns: {missing}")

    chunksize = int(ocfg.get("chunksize", 1000000))
    parts = []

    for i, chunk in enumerate(pd.read_csv(path, usecols=usecols, chunksize=chunksize), 1):
        chunk["date"] = normalize_date_series(chunk[c["date_col"]])
        chunk["__dt"] = parse_datetime_series(chunk[c["datetime_col"]])
        chunk["minute"] = chunk["__dt"].dt.floor("min")
        chunk["securityid"] = normalize_symbol_series(chunk[c["symbol_col"]])

        chunk = chunk.sort_values(["date", "minute", "securityid", "__dt"])
        chunk = chunk.groupby(["date", "minute", "securityid"], as_index=False).tail(1)
        parts.append(chunk)

        if i % 10 == 0:
            print(f"market chunks loaded: {i}")

    df = pd.concat(parts, ignore_index=True)
    df = df.sort_values(["date", "minute", "securityid", "__dt"])
    df = df.groupby(["date", "minute", "securityid"], as_index=False).tail(1)
    return df.reset_index(drop=True)


def load_rank_alpha(cfg):
    data_cfg = cfg["data"]
    c = cfg["columns"]
    ocfg = cfg["optimizer"]

    path = Path(data_cfg["rank_alpha_path"])
    if not path.exists():
        raise FileNotFoundError(f"rank_alpha_path not found: {path}")

    rank_col = c["rank_alpha_col"]

    header = pd.read_csv(path, nrows=0)
    cols = header.columns.tolist()

    needed = unique_list([
        c["date_col"],
        c["datetime_col"],
        c["symbol_col"],
        rank_col,
    ])

    usecols = [x for x in needed if x in cols]

    missing = [x for x in [c["datetime_col"], c["symbol_col"], rank_col] if x not in cols]
    if missing:
        raise ValueError(f"rank alpha file missing required columns: {missing}")

    chunksize = int(ocfg.get("chunksize", 1000000))
    parts = []

    for i, chunk in enumerate(pd.read_csv(path, usecols=usecols, chunksize=chunksize), 1):
        chunk["__dt"] = parse_datetime_series(chunk[c["datetime_col"]])
        chunk["minute"] = chunk["__dt"].dt.floor("min")
        chunk["securityid"] = normalize_symbol_series(chunk[c["symbol_col"]])

        if c.get("date_col") in chunk.columns:
            chunk["date"] = normalize_date_series(chunk[c["date_col"]])
        else:
            chunk["date"] = chunk["__dt"].dt.strftime("%Y%m%d").astype("int64")

        chunk[rank_col] = pd.to_numeric(chunk[rank_col], errors="coerce")
        chunk = chunk.dropna(subset=[rank_col])

        chunk = chunk.sort_values(["date", "minute", "securityid", "__dt"])
        chunk = chunk.groupby(["date", "minute", "securityid"], as_index=False).tail(1)
        parts.append(chunk[["date", "minute", "securityid", rank_col]])

        if i % 10 == 0:
            print(f"rank alpha chunks loaded: {i}")

    df = pd.concat(parts, ignore_index=True)
    df = df.drop_duplicates(["date", "minute", "securityid"], keep="last")
    return df


def load_pricing_optional(cfg):
    data_cfg = cfg["data"]
    c = cfg["columns"]
    ocfg = cfg["optimizer"]

    path = data_cfg.get("pricing_path")
    if not path:
        return None

    path = Path(path)
    if not path.exists():
        print(f"pricing_path not found, skip: {path}")
        return None

    header = pd.read_csv(path, nrows=0)
    cols = header.columns.tolist()

    pricing_pred_col = c.get("pricing_pred_col")
    fair_price_col = c.get("fair_price_col")

    needed = unique_list([
        c["date_col"],
        c["datetime_col"],
        c["symbol_col"],
        pricing_pred_col,
        fair_price_col,
    ])

    usecols = [x for x in needed if x in cols]

    if c["datetime_col"] not in cols or c["symbol_col"] not in cols:
        print("pricing file missing datetime/securityid, skip pricing")
        return None

    chunksize = int(ocfg.get("chunksize", 1000000))
    parts = []

    for i, chunk in enumerate(pd.read_csv(path, usecols=usecols, chunksize=chunksize), 1):
        chunk["__dt"] = parse_datetime_series(chunk[c["datetime_col"]])
        chunk["minute"] = chunk["__dt"].dt.floor("min")
        chunk["securityid"] = normalize_symbol_series(chunk[c["symbol_col"]])

        if c.get("date_col") in chunk.columns:
            chunk["date"] = normalize_date_series(chunk[c["date_col"]])
        else:
            chunk["date"] = chunk["__dt"].dt.strftime("%Y%m%d").astype("int64")

        keep = ["date", "minute", "securityid"]

        if pricing_pred_col and pricing_pred_col in chunk.columns:
            chunk[pricing_pred_col] = pd.to_numeric(chunk[pricing_pred_col], errors="coerce")
            keep.append(pricing_pred_col)

        if fair_price_col and fair_price_col in chunk.columns:
            chunk[fair_price_col] = pd.to_numeric(chunk[fair_price_col], errors="coerce")
            keep.append(fair_price_col)

        chunk = chunk.sort_values(["date", "minute", "securityid", "__dt"])
        chunk = chunk.groupby(["date", "minute", "securityid"], as_index=False).tail(1)
        parts.append(chunk[keep])

        if i % 10 == 0:
            print(f"pricing chunks loaded: {i}")

    if not parts:
        return None

    df = pd.concat(parts, ignore_index=True)
    df = df.drop_duplicates(["date", "minute", "securityid"], keep="last")
    return df


def build_input(cfg):
    market = load_minute_market(cfg)
    rank_alpha = load_rank_alpha(cfg)
    pricing = load_pricing_optional(cfg)

    df = market.merge(rank_alpha, on=["date", "minute", "securityid"], how="left")

    if pricing is not None:
        df = df.merge(pricing, on=["date", "minute", "securityid"], how="left")

    return df.sort_values(["minute", "securityid"]).reset_index(drop=True)


def add_market_features(df, cfg):
    c = cfg["columns"]
    ocfg = cfg["optimizer"]

    mid = pd.to_numeric(df[c["price_col"]], errors="coerce")
    bid = pd.to_numeric(df[c["bid_col"]], errors="coerce")
    ask = pd.to_numeric(df[c["ask_col"]], errors="coerce")

    if c.get("spread_col") in df.columns:
        spread = pd.to_numeric(df[c["spread_col"]], errors="coerce")
    else:
        spread = ask - bid

    max_spread = float(ocfg.get("max_spread_bps", 1e9))

    df["spread_bps"] = spread / mid * 10000.0
    df["valid_market"] = (
        np.isfinite(mid)
        & (mid > 0)
        & np.isfinite(bid)
        & np.isfinite(ask)
        & (ask >= bid)
        & np.isfinite(df["spread_bps"])
        & (df["spread_bps"] >= 0)
        & (df["spread_bps"] <= max_spread)
    )

    rank_col = c["rank_alpha_col"]
    df["score_for_rank"] = pd.to_numeric(df[rank_col], errors="coerce")
    df["has_rank_signal"] = df["score_for_rank"].notna()

    fair_col = c.get("fair_price_col")
    if fair_col and fair_col in df.columns:
        fair = pd.to_numeric(df[fair_col], errors="coerce")
        df["fair_price"] = fair
        df["buy_edge_bps"] = (fair - ask) / mid * 10000.0
        df["sell_edge_bps"] = (bid - fair) / mid * 10000.0
    else:
        df["fair_price"] = np.nan
        df["buy_edge_bps"] = np.nan
        df["sell_edge_bps"] = np.nan

    return df


def compute_desired_weights(df_t, state_qty, state_weight, state_entry_i, state_trade_i, minute_i, cfg):
    ocfg = cfg["optimizer"]

    top_n = int(ocfg.get("top_n", 50))
    exit_rank_n = int(ocfg.get("exit_rank_n", 80))

    target_gross = float(ocfg.get("target_gross_limit", 0.50))
    max_gross = float(ocfg.get("max_gross_limit", target_gross))
    single_limit = float(ocfg.get("single_name_limit", 0.02))

    min_hold = int(ocfg.get("min_hold_minutes", 5))
    cooldown = int(ocfg.get("trade_cooldown_minutes", 2))

    trade_allowed = is_trade_minute(df_t["minute"].iloc[0], ocfg)
    rebalance_allowed = trade_allowed and is_rebalance_minute(df_t["minute"].iloc[0], ocfg)

    df_t = df_t.copy()
    df_t["alpha_rank"] = pd.to_numeric(df_t["score_for_rank"], errors="coerce").rank(method="first", ascending=False)

    df_t["current_qty"] = df_t["securityid"].astype(str).map(state_qty).fillna(0).astype(int)
    df_t["current_weight"] = df_t["securityid"].astype(str).map(state_weight).fillna(0.0).astype(float)

    tradable_rank = df_t["has_rank_signal"].fillna(False) & df_t["valid_market"].fillna(False)
    top_mask = tradable_rank & (df_t["alpha_rank"] <= top_n)

    n_top = int(top_mask.sum())

    if n_top > 0:
        per_name_weight = min(single_limit, min(target_gross, max_gross) / n_top)
    else:
        per_name_weight = 0.0

    desired = []
    states = []
    hold_age_list = []
    cooldown_left_list = []

    for _, row in df_t.iterrows():
        sym = str(row["securityid"])
        current_w = float(row["current_weight"])
        current_qty = int(row["current_qty"])

        rank = row.get("alpha_rank", np.nan)
        has_rank = bool(row.get("has_rank_signal", False))
        valid_market = bool(row.get("valid_market", False))

        entry_i = state_entry_i.get(sym)
        last_trade_i = state_trade_i.get(sym)

        hold_age = 999999 if entry_i is None else max(0, minute_i - entry_i)
        cooldown_left = 0 if last_trade_i is None else max(0, cooldown - (minute_i - last_trade_i))

        if not trade_allowed:
            dw = current_w
            st = "HOLD" if current_qty > 0 else "FLAT"

        elif not rebalance_allowed:
            dw = current_w
            st = "HOLD" if current_qty > 0 else "FLAT"

        else:
            is_top = bool(has_rank and valid_market and np.isfinite(rank) and rank <= top_n)
            is_hold_rank = bool(has_rank and valid_market and np.isfinite(rank) and rank <= exit_rank_n)

            if current_qty <= 0:
                if is_top and cooldown_left == 0:
                    dw = per_name_weight
                    st = "ENTRY"
                else:
                    dw = 0.0
                    st = "FLAT"
            else:
                if hold_age < min_hold:
                    dw = current_w
                    st = "HOLD"
                elif is_top and cooldown_left == 0:
                    dw = max(current_w, per_name_weight)
                    st = "ADD" if dw > current_w else "HOLD"
                elif is_hold_rank:
                    dw = current_w
                    st = "HOLD"
                else:
                    dw = 0.0
                    st = "EXIT"

        desired.append(float(dw))
        states.append(st)
        hold_age_list.append(hold_age)
        cooldown_left_list.append(cooldown_left)

    df_t["desired_weight"] = desired
    df_t["state"] = states
    df_t["hold_age_min"] = hold_age_list
    df_t["cooldown_left_min"] = cooldown_left_list
    df_t["target_gross_t"] = min(target_gross, max_gross)

    return df_t


def apply_execution(df_t, state_qty, state_weight, state_entry_i, state_trade_i, minute_i, cfg):
    c = cfg["columns"]
    ocfg = cfg["optimizer"]

    capital = float(ocfg.get("capital", 1.0))
    lot = int(ocfg.get("lot_size", 100))
    smoothing = float(ocfg.get("rebalance_smoothing", 0.50))
    exit_smoothing = float(ocfg.get("exit_smoothing", smoothing))
    band = float(ocfg.get("target_weight_band", 0.00005))

    min_notional = float(ocfg.get("min_trade_notional", 20000))
    volume_cap_ratio = float(ocfg.get("volume_cap_ratio", 0.20))
    turnover_limit = float(ocfg.get("turnover_limit", 0.05))
    remaining_turnover_notional = turnover_limit * capital

    df_t = df_t.copy()

    price = pd.to_numeric(df_t[c["price_col"]], errors="coerce")
    bid_vol = pd.to_numeric(df_t[c["bid_volume_col"]], errors="coerce")
    ask_vol = pd.to_numeric(df_t[c["ask_volume_col"]], errors="coerce")

    target_weights = []
    target_qtys = []
    delta_raws = []

    for idx, row in df_t.iterrows():
        current_w = float(row["current_weight"])
        desired_w = float(row["desired_weight"])
        st = str(row["state"])

        if abs(desired_w - current_w) < band:
            target_w = current_w
        else:
            sm = exit_smoothing if st == "EXIT" else smoothing
            target_w = current_w + sm * (desired_w - current_w)

        target_w = max(0.0, target_w)

        px = price.loc[idx]
        qty = floor_qty_to_lot(target_w * capital / px, lot) if np.isfinite(px) and px > 0 else 0

        current_qty = int(row["current_qty"])
        delta_raw = int(qty - current_qty)

        target_weights.append(target_w)
        target_qtys.append(qty)
        delta_raws.append(delta_raw)

    df_t["target_weight"] = target_weights
    df_t["target_qty"] = target_qtys
    df_t["delta_qty_raw"] = delta_raws

    df_t["delta_qty_executable"] = 0
    df_t["blocked_reason"] = "no_delta"
    df_t["side"] = "NONE"
    df_t["abs_delta_notional"] = 0.0

    def priority_key(row):
        d = int(row["delta_qty_raw"])
        if d < 0:
            return (0, 0)
        if d > 0:
            r = row.get("alpha_rank", np.inf)
            return (1, r if np.isfinite(r) else 999999)
        return (2, 999999)

    order_rows = []
    for idx, row in df_t.iterrows():
        d = int(row["delta_qty_raw"])
        if d != 0:
            order_rows.append((priority_key(row), idx))

    order_rows.sort(key=lambda x: x[0])

    exec_delta = {}
    reason = {}

    for _, idx in order_rows:
        row = df_t.loc[idx]
        d = int(row["delta_qty_raw"])
        px = price.loc[idx]

        if not np.isfinite(px) or px <= 0:
            exec_delta[idx] = 0
            reason[idx] = "bad_price"
            continue

        notional = abs(d) * px

        if not bool(row.get("valid_market", False)):
            exec_delta[idx] = 0
            reason[idx] = "invalid_market"
            continue

        if notional < min_notional:
            exec_delta[idx] = 0
            reason[idx] = "below_min_trade_notional"
            continue

        if remaining_turnover_notional <= 0:
            exec_delta[idx] = 0
            reason[idx] = "blocked_by_turnover_limit"
            continue

        max_qty_turnover = floor_qty_to_lot(remaining_turnover_notional / px, lot)

        if d > 0:
            max_qty_volume = floor_qty_to_lot(ask_vol.loc[idx] * volume_cap_ratio, lot) if np.isfinite(ask_vol.loc[idx]) else abs(d)
            q = min(abs(d), max_qty_volume, max_qty_turnover)
            q = floor_qty_to_lot(q, lot)
            ex = int(q)
            if ex <= 0:
                rs = "blocked_by_ask_volume_or_turnover"
            elif ex < abs(d):
                rs = "clipped_by_ask_volume_or_turnover"
            else:
                rs = "none"
        else:
            current_qty = int(row["current_qty"])
            max_qty_volume = floor_qty_to_lot(bid_vol.loc[idx] * volume_cap_ratio, lot) if np.isfinite(bid_vol.loc[idx]) else abs(d)
            q = min(abs(d), current_qty, max_qty_volume, max_qty_turnover)
            q = floor_qty_to_lot(q, lot)
            ex = -int(q)
            if q <= 0:
                rs = "blocked_by_sellable_or_bid_volume_or_turnover"
            elif q < abs(d):
                rs = "clipped_by_sellable_or_bid_volume_or_turnover"
            else:
                rs = "none"

        exec_delta[idx] = ex
        reason[idx] = rs

        remaining_turnover_notional -= abs(ex) * px

    for idx in df_t.index:
        ex = int(exec_delta.get(idx, 0))
        df_t.at[idx, "delta_qty_executable"] = ex
        df_t.at[idx, "blocked_reason"] = reason.get(idx, "no_delta")
        df_t.at[idx, "side"] = "BUY" if ex > 0 else ("SELL" if ex < 0 else "NONE")
        px = price.loc[idx]
        df_t.at[idx, "abs_delta_notional"] = abs(ex) * px if np.isfinite(px) else 0.0

    effective_qty = []
    effective_weight = []

    for idx, row in df_t.iterrows():
        sym = str(row["securityid"])
        px = price.loc[idx]

        current_qty = int(row["current_qty"])
        ex = int(row["delta_qty_executable"])
        q = int(current_qty + ex)
        w = q * px / capital if np.isfinite(px) and capital > 0 else 0.0

        effective_qty.append(q)
        effective_weight.append(w)

        state_qty[sym] = q
        state_weight[sym] = w

        if ex != 0:
            state_trade_i[sym] = minute_i

        if current_qty <= 0 and q > 0:
            state_entry_i[sym] = minute_i

        if q <= 0:
            state_entry_i.pop(sym, None)

    df_t["effective_target_qty"] = effective_qty
    df_t["effective_target_weight"] = effective_weight
    df_t["gross_weight"] = np.abs(df_t["effective_target_weight"])
    df_t["selected"] = df_t["effective_target_qty"] != 0

    return df_t


def apply_optimizer(cfg):
    data_cfg = cfg["data"]
    c = cfg["columns"]

    output_path = data_cfg["output_path"]
    log_path = data_cfg.get("log_path")

    mkdir_parent(output_path)
    if log_path:
        mkdir_parent(log_path)
        Path(log_path).write_text("", encoding="utf-8")

    df = build_input(cfg)
    df = add_market_features(df, cfg)

    log_print("===== input summary =====", log_path)
    log_print(f"rows: {len(df)}", log_path)
    log_print(f"date range: {df['date'].min()} -> {df['date'].max()}", log_path)
    log_print(f"minutes: {df['minute'].nunique()}", log_path)
    log_print(f"symbols: {df['securityid'].nunique()}", log_path)
    log_print(f"rank signal non-null rate: {df['has_rank_signal'].mean():.6f}", log_path)

    state_qty = defaultdict(int)
    state_weight = defaultdict(float)
    state_entry_i = {}
    state_trade_i = {}

    all_out = []
    minutes = sorted(df["minute"].dropna().unique())

    for i, minute in enumerate(minutes, 1):
        df_t = df[df["minute"] == minute].copy()

        df_t = compute_desired_weights(
            df_t,
            state_qty,
            state_weight,
            state_entry_i,
            state_trade_i,
            i,
            cfg,
        )

        out = apply_execution(
            df_t,
            state_qty,
            state_weight,
            state_entry_i,
            state_trade_i,
            i,
            cfg,
        )

        out["optimizer_status"] = "rank_target_portfolio"

        keep_cols = unique_list([
            "date",
            "minute",
            c["datetime_col"],
            "securityid",
            c["rank_alpha_col"],
            "score_for_rank",
            "alpha_rank",
            c.get("pricing_pred_col"),
            "fair_price",
            "buy_edge_bps",
            "sell_edge_bps",
            c["price_col"],
            c.get("bid_col"),
            c.get("ask_col"),
            c.get("spread_col"),
            c.get("bid_volume_col"),
            c.get("ask_volume_col"),
            "spread_bps",
            "valid_market",
            "has_rank_signal",
            "state",
            "hold_age_min",
            "cooldown_left_min",
            "target_gross_t",
            "desired_weight",
            "current_weight",
            "current_qty",
            "target_weight",
            "target_qty",
            "delta_qty_raw",
            "delta_qty_executable",
            "effective_target_qty",
            "effective_target_weight",
            "gross_weight",
            "selected",
            "side",
            "blocked_reason",
            "abs_delta_notional",
            "optimizer_status",
        ])

        keep_cols = [x for x in keep_cols if x in out.columns]
        all_out.append(out[keep_cols])

        if i % 50 == 0 or i == len(minutes):
            log_print(
                f"minute {i}/{len(minutes)} {minute}, "
                f"rows={len(out)}, rank_signal={int(out['has_rank_signal'].sum())}, "
                f"selected={int(out['selected'].sum())}, "
                f"gross={float(out['gross_weight'].sum()):.6f}, "
                f"delta_notional={float(out['abs_delta_notional'].sum()):.2f}, "
                f"states={dict(out['state'].value_counts().head())}",
                log_path,
            )

    final = pd.concat(all_out, ignore_index=True)
    final.to_csv(output_path, index=False)

    log_print("===== final summary =====", log_path)
    log_print(f"output_path: {output_path}", log_path)
    log_print(f"rows: {len(final)}", log_path)
    log_print(f"date range: {final['date'].min()} -> {final['date'].max()}", log_path)
    log_print(f"minutes: {final['minute'].nunique()}", log_path)
    log_print(f"avg selected per minute: {final.groupby('minute')['selected'].sum().mean():.4f}", log_path)
    log_print(f"avg gross weight: {final.groupby('minute')['gross_weight'].sum().mean():.6f}", log_path)
    log_print(f"avg abs executable delta notional per minute: {final.groupby('minute')['abs_delta_notional'].sum().mean():.2f}", log_path)
    log_print("side counts:", log_path)
    log_print(str(final["side"].value_counts()), log_path)
    log_print("state counts:", log_path)
    log_print(str(final["state"].value_counts()), log_path)
    log_print("blocked reason counts:", log_path)
    log_print(str(final["blocked_reason"].value_counts().head(20)), log_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = read_yaml(args.config)
    apply_optimizer(cfg)


if __name__ == "__main__":
    main()
