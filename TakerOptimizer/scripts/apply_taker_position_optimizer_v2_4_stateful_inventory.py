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


def floor_signed_to_lot(notional, price, lot_size):
    if not np.isfinite(notional) or not np.isfinite(price) or price <= 0:
        return 0
    sign = 1 if notional >= 0 else -1
    qty = floor_qty_to_lot(abs(notional) / price, lot_size)
    return int(sign * qty)


def minute_index(ts):
    t = pd.Timestamp(ts)
    return int(t.hour * 60 + t.minute)


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


def load_market_minute(cfg):
    data_cfg = cfg["data"]
    c = cfg["columns"]
    ocfg = cfg["optimizer"]

    path = Path(data_cfg["market_data_path"])
    if not path.exists():
        raise FileNotFoundError(f"market_data_path not found: {path}")

    header = pd.read_csv(path, nrows=0)
    cols = header.columns.tolist()

    needed = unique_list([
        c["date_col"], c["datetime_col"], c["symbol_col"], c["price_col"],
        c.get("bid_col"), c.get("ask_col"), c.get("spread_col"),
        c.get("bid_volume_col"), c.get("ask_volume_col"),
        c.get("limit_up_col"), c.get("limit_down_col"),
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
    df = df.reset_index(drop=True)
    return df



def load_rank_alpha_minute(cfg):
    data_cfg = cfg["data"]
    c = cfg["columns"]
    ocfg = cfg["optimizer"]

    path = data_cfg.get("rank_alpha_path")
    rank_col = c.get("rank_alpha_col")

    if not path or not rank_col:
        return None

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"rank_alpha_path not found: {path}")

    header = pd.read_csv(path, nrows=0)
    cols = header.columns.tolist()

    needed = unique_list([
        c["date_col"], c["datetime_col"], c["symbol_col"], rank_col
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

    if not parts:
        return None

    df = pd.concat(parts, ignore_index=True)
    df = df.drop_duplicates(["date", "minute", "securityid"], keep="last")
    return df


def load_pricing_minute(cfg):
    data_cfg = cfg["data"]
    c = cfg["columns"]
    ocfg = cfg["optimizer"]

    path = Path(data_cfg["pricing_path"])
    if not path.exists():
        raise FileNotFoundError(f"pricing_path not found: {path}")

    header = pd.read_csv(path, nrows=0)
    cols = header.columns.tolist()

    needed = unique_list([
        c["date_col"], c["datetime_col"], c["symbol_col"],
        c["alpha_col"], c["fair_price_col"],
    ])
    usecols = [x for x in needed if x in cols]

    missing = [x for x in [c["datetime_col"], c["symbol_col"], c["alpha_col"], c["fair_price_col"]] if x not in cols]
    if missing:
        raise ValueError(f"pricing file missing required columns: {missing}")

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

        chunk[c["alpha_col"]] = pd.to_numeric(chunk[c["alpha_col"]], errors="coerce")
        chunk[c["fair_price_col"]] = pd.to_numeric(chunk[c["fair_price_col"]], errors="coerce")

        chunk = chunk.sort_values(["date", "minute", "securityid", "__dt"])
        chunk = chunk.groupby(["date", "minute", "securityid"], as_index=False).tail(1)
        parts.append(chunk[["date", "minute", "securityid", c["alpha_col"], c["fair_price_col"]]])

        if i % 10 == 0:
            print(f"pricing chunks loaded: {i}")

    df = pd.concat(parts, ignore_index=True)
    df = df.drop_duplicates(["date", "minute", "securityid"], keep="last")
    return df


def load_barra_proxy(cfg):
    path = Path(cfg["data"]["barra_proxy_path"])
    if not path.exists():
        raise FileNotFoundError(f"barra_proxy_path not found: {path}")

    c = cfg["columns"]
    style_cols = c.get("barra_style_cols", [])
    industry_col = c.get("industry_col")

    header = pd.read_csv(path, nrows=0)
    cols = header.columns.tolist()

    needed = unique_list(["date", "securityid", industry_col] + style_cols)
    usecols = [x for x in needed if x in cols]

    barra = pd.read_csv(path, usecols=usecols, low_memory=False)
    barra["date"] = normalize_date_series(barra["date"])
    barra["securityid"] = normalize_symbol_series(barra["securityid"])

    for col in style_cols:
        if col in barra.columns:
            barra[col] = pd.to_numeric(barra[col], errors="coerce").fillna(0.0)

    if industry_col and industry_col in barra.columns:
        barra[industry_col] = barra[industry_col].fillna("UNKNOWN").astype(str)

    return barra.drop_duplicates(["date", "securityid"], keep="last")


def build_input(cfg):
    c = cfg["columns"]

    market = load_market_minute(cfg)
    rank_alpha = load_rank_alpha_minute(cfg)
    pricing = load_pricing_minute(cfg)
    barra = load_barra_proxy(cfg)

    df = market.copy()

    if rank_alpha is not None:
        df = df.merge(rank_alpha, on=["date", "minute", "securityid"], how="left")

    df = df.merge(pricing, on=["date", "minute", "securityid"], how="left")
    df = df.merge(barra, on=["date", "securityid"], how="left")

    for col in c.get("barra_style_cols", []):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    industry_col = c.get("industry_col")
    if industry_col and industry_col in df.columns:
        df[industry_col] = df[industry_col].fillna("UNKNOWN").astype(str)

    return df.sort_values(["minute", "securityid"]).reset_index(drop=True)

def add_features(df, cfg):
    c = cfg["columns"]
    ocfg = cfg["optimizer"]

    mid = pd.to_numeric(df[c["price_col"]], errors="coerce")
    bid = pd.to_numeric(df[c["bid_col"]], errors="coerce")
    ask = pd.to_numeric(df[c["ask_col"]], errors="coerce")

    if c.get("spread_col") in df.columns:
        spread = pd.to_numeric(df[c["spread_col"]], errors="coerce")
    else:
        spread = ask - bid

    fair = pd.to_numeric(df[c["fair_price_col"]], errors="coerce")
    alpha = pd.to_numeric(df[c["alpha_col"]], errors="coerce")

    fee = float(ocfg.get("taker_fee_bps", 0.0))
    impact = float(ocfg.get("impact_bps", 0.0))

    df["fair_price"] = fair
    df["alpha_bps"] = alpha * 10000.0
    df["spread_bps"] = spread / mid * 10000.0
    df["buy_edge_bps"] = (fair - ask) / mid * 10000.0
    df["sell_edge_bps"] = (bid - fair) / mid * 10000.0
    df["net_buy_edge_bps"] = df["buy_edge_bps"] - fee - impact
    df["net_sell_edge_bps"] = df["sell_edge_bps"] - fee - impact

    max_spread = float(ocfg.get("max_spread_bps", 1e9))
    df["has_signal"] = df["fair_price"].notna()
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

    df["pricing_valid"] = df["valid_market"] & df["has_signal"]

    # Rank signal for portfolio construction.
    # Prefer full-universe rank alpha. Pricing fair_price can be sparse.
    rank_col = c.get("rank_alpha_col")
    if rank_col and rank_col in df.columns:
        df["score_for_rank"] = pd.to_numeric(df[rank_col], errors="coerce")
    else:
        df["score_for_rank"] = alpha

    df["has_rank_signal"] = df["score_for_rank"].notna()

    return df


def dynamic_gross_limit(df_t, cfg):
    ocfg = cfg["optimizer"]
    base_gross = float(ocfg.get("base_gross_limit", 0.03))
    max_gross = float(ocfg.get("max_gross_limit", base_gross))
    full_edge = float(ocfg.get("net_alpha_bps_for_max_gross", 6.0))

    x = pd.to_numeric(df_t.loc[df_t["state"].isin(["ENTRY", "ADD"]), "net_buy_edge_bps"], errors="coerce")
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return base_gross

    avg_edge = float(x.sort_values(ascending=False).head(10).mean())
    ratio = np.clip(avg_edge / full_edge, 0.0, 1.0) if full_edge > 0 else 1.0
    return float(base_gross + (max_gross - base_gross) * ratio)


def solve_stateful(df_t, prev_weight_map, cfg, gross_limit_t):
    c = cfg["columns"]
    ocfg = cfg["optimizer"]

    n = len(df_t)
    if n < int(ocfg.get("min_names_per_timestamp", 20)):
        out = df_t[["securityid"]].copy()
        out["raw_target_weight"] = df_t["current_weight"].to_numpy()
        out["optimizer_status"] = "too_few_names_hold"
        return out

    try:
        import cvxpy as cp
    except Exception:
        out = df_t[["securityid"]].copy()
        out["raw_target_weight"] = df_t["current_weight"].to_numpy()
        out["optimizer_status"] = "no_cvxpy_hold"
        return out

    symbols = df_t["securityid"].astype(str).tolist()
    w_prev = np.array([prev_weight_map.get(s, 0.0) for s in symbols], dtype=float)

    current_w = df_t["current_weight"].to_numpy(dtype=float)

    lb = np.zeros(n)
    ub = np.zeros(n)

    single = float(ocfg.get("single_name_limit", 0.003))

    state = df_t["state"].astype(str).to_numpy()

    for i, st in enumerate(state):
        if st == "ENTRY":
            lb[i] = 0.0
            ub[i] = single
        elif st == "ADD":
            lb[i] = current_w[i]
            ub[i] = single
        elif st == "HOLD":
            lb[i] = current_w[i]
            ub[i] = current_w[i]
        elif st == "EXIT":
            lb[i] = 0.0
            ub[i] = 0.0
        else:
            lb[i] = 0.0
            ub[i] = 0.0

    if ub.sum() <= 0:
        out = df_t[["securityid"]].copy()
        out["raw_target_weight"] = 0.0
        out.loc[df_t["state"].astype(str).eq("HOLD"), "raw_target_weight"] = df_t.loc[df_t["state"].astype(str).eq("HOLD"), "current_weight"]
        out["optimizer_status"] = "no_active_state"
        return out

    alpha = pd.to_numeric(df_t["score_for_rank"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    alpha = alpha - np.nanmean(alpha)
    std = np.nanstd(alpha)
    if std > 0:
        alpha = alpha / std

    risk_diag = np.ones(n)
    if "volatility_z" in df_t.columns:
        vol = pd.to_numeric(df_t["volatility_z"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        risk_diag = 1.0 + np.square(vol)

    w = cp.Variable(n)

    objective_terms = [
        alpha @ w,
        -float(ocfg.get("risk_aversion", 1.0)) * cp.sum(cp.multiply(risk_diag, cp.square(w))),
        -float(ocfg.get("turnover_penalty", 0.05)) * cp.norm1(w - w_prev),
    ]

    style_cols = [x for x in c.get("barra_style_cols", []) if x in df_t.columns]
    X_style = None
    if style_cols:
        xs = []
        for col in style_cols:
            xs.append(pd.to_numeric(df_t[col], errors="coerce").fillna(0.0).to_numpy(dtype=float))
        X_style = np.vstack(xs).T
        objective_terms.append(
            -float(ocfg.get("barra_risk_aversion", 0.2)) * cp.sum_squares(X_style.T @ w)
        )

    industry_col = c.get("industry_col", "industryID1")
    X_industry = None
    if industry_col in df_t.columns:
        ind = df_t[industry_col].where(df_t[industry_col].notna(), "UNKNOWN").map(str)
        X_industry = pd.get_dummies(ind).to_numpy(dtype=float)
        objective_terms.append(
            -float(ocfg.get("industry_risk_aversion", 0.1)) * cp.sum_squares(X_industry.T @ w)
        )

    constraints = [
        w >= lb,
        w <= ub,
        cp.sum(w) <= gross_limit_t,
        cp.norm1(w - w_prev) <= float(ocfg.get("turnover_limit", 0.005)),
    ]

    style_limit = float(ocfg.get("style_limit", 0.03))
    if X_style is not None:
        for j in range(X_style.shape[1]):
            x = X_style[:, j]
            constraints += [x @ w <= style_limit, x @ w >= -style_limit]

    industry_limit = float(ocfg.get("industry_limit", 0.03))
    if X_industry is not None:
        constraints += [X_industry.T @ w <= industry_limit, X_industry.T @ w >= -industry_limit]

    prob = cp.Problem(cp.Maximize(sum(objective_terms)), constraints)

    try:
        solver = ocfg.get("solver", "OSQP")
        prob.solve(solver=solver, warm_start=True, verbose=False)
        status = str(prob.status)
    except Exception:
        try:
            prob.solve(warm_start=True, verbose=False)
            status = str(prob.status)
        except Exception as e:
            out = df_t[["securityid"]].copy()
            out["raw_target_weight"] = current_w
            out["optimizer_status"] = "solve_error_hold:" + str(e)[:80]
            return out

    if w.value is None:
        out = df_t[["securityid"]].copy()
        out["raw_target_weight"] = current_w
        out["optimizer_status"] = "solve_none_hold"
        return out

    res = np.asarray(w.value).reshape(-1)
    res[np.abs(res) < 1e-10] = 0.0

    out = df_t[["securityid"]].copy()
    out["raw_target_weight"] = res
    out["optimizer_status"] = status
    return out


def apply_caps_and_update(out, state_maps, cfg):
    c = cfg["columns"]
    ocfg = cfg["optimizer"]

    capital = float(ocfg.get("capital", 1.0))
    lot = int(ocfg.get("lot_size", 100))
    smoothing = float(ocfg.get("rebalance_smoothing", 0.1))
    band = float(ocfg.get("target_weight_band", 0.001))
    min_trade_notional = float(ocfg.get("min_trade_notional", 50000))
    volume_cap_ratio = float(ocfg.get("volume_cap_ratio", 0.20))
    enforce_sellable = bool(ocfg.get("enforce_sellable_inventory", True))

    rows = []

    for _, row in out.iterrows():
        sym = str(row["securityid"])

        px = pd.to_numeric(pd.Series([row.get(c["price_col"], np.nan)]), errors="coerce").iloc[0]
        bid = pd.to_numeric(pd.Series([row.get(c["bid_col"], np.nan)]), errors="coerce").iloc[0]
        ask = pd.to_numeric(pd.Series([row.get(c["ask_col"], np.nan)]), errors="coerce").iloc[0]
        bid_vol = pd.to_numeric(pd.Series([row.get(c["bid_volume_col"], np.nan)]), errors="coerce").iloc[0]
        ask_vol = pd.to_numeric(pd.Series([row.get(c["ask_volume_col"], np.nan)]), errors="coerce").iloc[0]
        limit_up = pd.to_numeric(pd.Series([row.get(c["limit_up_col"], np.nan)]), errors="coerce").iloc[0]
        limit_down = pd.to_numeric(pd.Series([row.get(c["limit_down_col"], np.nan)]), errors="coerce").iloc[0]

        current_qty = int(state_maps["qty"].get(sym, 0))
        current_weight = float(state_maps["weight"].get(sym, 0.0))
        sellable_qty = max(0, current_qty)

        raw_w = float(row.get("raw_target_weight", current_weight))

        state = str(row.get("state", ""))

        if state == "EXIT":
            target_w = 0.0
        elif state == "ENTRY":
            target_w = current_weight + max(smoothing, 0.3) * (raw_w - current_weight)
        elif abs(raw_w - current_weight) < band:
            target_w = current_weight
        else:
            target_w = current_weight + smoothing * (raw_w - current_weight)

        target_w = max(0.0, target_w)
        target_qty = floor_signed_to_lot(target_w * capital, px, lot)
        delta_raw = int(target_qty - current_qty)

        trade_notional = abs(delta_raw) * px if np.isfinite(px) else 0.0

        max_buy = floor_qty_to_lot(ask_vol * volume_cap_ratio, lot) if np.isfinite(ask_vol) else abs(delta_raw)
        max_sell = floor_qty_to_lot(bid_vol * volume_cap_ratio, lot) if np.isfinite(bid_vol) else abs(delta_raw)
        if enforce_sellable:
            max_sell = min(max_sell, sellable_qty)

        delta_exec = delta_raw
        reason = "none"

        if delta_raw == 0:
            delta_exec = 0
            reason = "no_delta"
        elif not bool(row.get("valid_market", False)):
            delta_exec = 0
            reason = "invalid_market"
        elif trade_notional < min_trade_notional:
            delta_exec = 0
            reason = "below_min_trade_notional"
        elif delta_raw > 0:
            if np.isfinite(limit_up) and np.isfinite(ask) and ask >= limit_up:
                delta_exec = 0
                reason = "limit_up_block"
            else:
                delta_exec = min(delta_raw, max_buy)
                if delta_exec <= 0:
                    reason = "blocked_by_ask_volume"
                elif delta_exec < delta_raw:
                    reason = "clipped_by_ask_volume"
        else:
            # Do not block sell only because bid is near limit_down at optimizer layer.
            # Actual fill feasibility will be handled by TakerBacktestingModel.
            sell_exec = min(abs(delta_raw), max_sell)
            delta_exec = -int(sell_exec)
            if sell_exec <= 0:
                reason = "blocked_by_no_sellable_or_bid_volume"
            elif sell_exec < abs(delta_raw):
                reason = "clipped_by_sellable_or_bid_volume"

        effective_qty = int(current_qty + delta_exec)
        effective_weight = (effective_qty * px / capital) if np.isfinite(px) and capital > 0 else 0.0

        side = "BUY" if delta_exec > 0 else ("SELL" if delta_exec < 0 else "NONE")

        row = row.copy()
        row["current_qty"] = current_qty
        row["sellable_qty"] = sellable_qty
        row["raw_target_weight"] = raw_w
        row["target_weight"] = target_w
        row["target_qty"] = target_qty
        row["delta_qty_raw"] = delta_raw
        row["delta_qty_executable"] = int(delta_exec)
        row["effective_target_qty"] = effective_qty
        row["effective_target_weight"] = effective_weight
        row["side"] = side
        row["blocked_reason"] = reason
        row["max_buy_qty_by_ask_volume"] = int(max_buy) if np.isfinite(max_buy) else 0
        row["max_sell_qty_by_bid_volume"] = int(max_sell) if np.isfinite(max_sell) else 0
        row["abs_delta_notional"] = abs(delta_exec) * px if np.isfinite(px) else 0.0
        row["gross_weight"] = abs(effective_weight)
        row["selected"] = effective_qty != 0
        rows.append(row)

        state_maps["qty"][sym] = effective_qty
        state_maps["weight"][sym] = effective_weight

        if side != "NONE":
            state_maps["last_trade_minute"][sym] = row["minute"]

        if current_qty == 0 and effective_qty > 0:
            state_maps["entry_minute"][sym] = row["minute"]
        if effective_qty == 0:
            state_maps["entry_minute"].pop(sym, None)

    return pd.DataFrame(rows)


def build_state(df_t, state_maps, cfg, minute, minute_no):
    ocfg = cfg["optimizer"]

    entry_edge = float(ocfg.get("entry_edge_bps", 0.0))
    add_edge = float(ocfg.get("add_edge_bps", 1.0))
    hold_edge = float(ocfg.get("hold_edge_bps", -1.0))
    exit_edge = float(ocfg.get("exit_edge_bps", 1.0))
    min_hold = int(ocfg.get("min_hold_minutes", 5))
    cooldown = int(ocfg.get("trade_cooldown_minutes", 3))
    max_stale = int(ocfg.get("max_stale_hold_minutes", 10))

    can_rebalance = is_rebalance_minute(minute, ocfg)

    # Cross-sectional rank: smaller rank means stronger predicted return.
    df_t["alpha_rank"] = pd.to_numeric(df_t.get("score_for_rank", np.nan), errors="coerce").rank(
        method="first",
        ascending=False,
    )

    entry_top_n = int(ocfg.get("entry_top_n", 30))
    add_top_n = int(ocfg.get("add_top_n", entry_top_n))
    hold_top_n = int(ocfg.get("hold_top_n", 50))
    exit_rank_n = int(ocfg.get("exit_rank_n", hold_top_n))

    states = []
    hold_ages = []
    cooldown_lefts = []
    stale_counts = []

    for _, row in df_t.iterrows():
        sym = str(row["securityid"])
        qty = int(state_maps["qty"].get(sym, 0))
        w = float(state_maps["weight"].get(sym, 0.0))

        entry_minute = state_maps["entry_minute"].get(sym)
        if entry_minute is None:
            hold_age = 999999
        else:
            hold_age = max(0, minute_index(minute) - minute_index(entry_minute))

        last_trade = state_maps["last_trade_minute"].get(sym)
        if last_trade is None:
            cooldown_left = 0
        else:
            cooldown_left = max(0, cooldown - (minute_index(minute) - minute_index(last_trade)))

        stale = int(state_maps["stale"].get(sym, 0))

        has_signal = bool(row.get("has_signal", False))
        has_rank_signal = bool(row.get("has_rank_signal", False))
        pricing_valid = bool(row.get("pricing_valid", False))
        valid_market = bool(row.get("valid_market", False))

        buy_edge = row.get("net_buy_edge_bps", np.nan)
        sell_edge = row.get("net_sell_edge_bps", np.nan)

        if has_signal:
            stale = 0
        elif qty > 0:
            stale += 1
        else:
            stale = 0

        rank = row.get("alpha_rank", np.nan)
        rank_ok_entry = np.isfinite(rank) and rank <= entry_top_n
        rank_ok_add = np.isfinite(rank) and rank <= add_top_n
        rank_ok_hold = np.isfinite(rank) and rank <= hold_top_n
        rank_bad_exit = np.isfinite(rank) and rank > exit_rank_n

        if qty <= 0:
            # Empty inventory: allow entry by cross-sectional rank.
            # Edge is not the only entry condition; otherwise gross can never build up.
            if can_rebalance and pricing_valid and cooldown_left == 0 and rank_ok_entry:
                st = "ENTRY"
            elif can_rebalance and pricing_valid and cooldown_left == 0 and np.isfinite(buy_edge) and buy_edge >= entry_edge:
                st = "ENTRY"
            else:
                st = "FLAT"

        else:
            if not has_signal:
                if stale <= max_stale:
                    st = "HOLD"
                else:
                    st = "EXIT"

            elif hold_age < min_hold:
                st = "HOLD"

            elif cooldown_left > 0:
                st = "HOLD"

            elif pricing_valid and np.isfinite(sell_edge) and sell_edge >= exit_edge:
                st = "EXIT"

            elif rank_bad_exit:
                st = "EXIT"

            elif can_rebalance and pricing_valid and rank_ok_add:
                st = "ADD"

            elif can_rebalance and pricing_valid and np.isfinite(buy_edge) and buy_edge >= add_edge:
                st = "ADD"

            elif rank_ok_hold:
                st = "HOLD"

            elif pricing_valid and np.isfinite(buy_edge) and buy_edge >= hold_edge:
                st = "HOLD"

            else:
                st = "EXIT"

        states.append(st)
        hold_ages.append(hold_age)
        cooldown_lefts.append(cooldown_left)
        stale_counts.append(stale)
        state_maps["stale"][sym] = stale

    df_t["state"] = states
    df_t["hold_age_min"] = hold_ages
    df_t["cooldown_left_min"] = cooldown_lefts
    df_t["stale_signal_count"] = stale_counts

    df_t["current_qty"] = df_t["securityid"].astype(str).map(state_maps["qty"]).fillna(0).astype(int)
    df_t["current_weight"] = df_t["securityid"].astype(str).map(state_maps["weight"]).fillna(0.0).astype(float)

    return df_t


def apply_optimizer(cfg):
    data_cfg = cfg["data"]
    log_path = data_cfg.get("log_path")
    output_path = data_cfg["output_path"]

    mkdir_parent(output_path)
    if log_path:
        mkdir_parent(log_path)
        Path(log_path).write_text("", encoding="utf-8")

    df = build_input(cfg)
    df = add_features(df, cfg)

    log_print("===== input summary =====", log_path)
    log_print(f"rows: {len(df)}", log_path)
    log_print(f"date range: {df['date'].min()} -> {df['date'].max()}", log_path)
    log_print(f"minutes: {df['minute'].nunique()}", log_path)
    log_print(f"symbols: {df['securityid'].nunique()}", log_path)
    log_print(f"pricing signal non-null rate: {df['has_signal'].mean():.6f}", log_path)

    state_maps = {
        "qty": defaultdict(int),
        "weight": defaultdict(float),
        "entry_minute": {},
        "last_trade_minute": {},
        "stale": defaultdict(int),
    }

    all_out = []
    minutes = sorted(df["minute"].dropna().unique())

    for i, minute in enumerate(minutes, 1):
        df_t = df[df["minute"] == minute].copy()

        if not is_trade_minute(minute, cfg["optimizer"]):
            df_t["state"] = np.where(
                df_t["securityid"].astype(str).map(state_maps["qty"]).fillna(0).astype(int) > 0,
                "HOLD",
                "FLAT",
            )
            df_t["hold_age_min"] = 0
            df_t["cooldown_left_min"] = 0
            df_t["stale_signal_count"] = 0
            df_t["current_qty"] = df_t["securityid"].astype(str).map(state_maps["qty"]).fillna(0).astype(int)
            df_t["current_weight"] = df_t["securityid"].astype(str).map(state_maps["weight"]).fillna(0.0).astype(float)
            df_t["raw_target_weight"] = df_t["current_weight"]
            df_t["optimizer_status"] = "blocked_time_hold"
            gross_limit_t = float(df_t["current_weight"].abs().sum())
        else:
            df_t = build_state(df_t, state_maps, cfg, minute, i)
            gross_limit_t = dynamic_gross_limit(df_t, cfg)
            opt_out = solve_stateful(df_t, state_maps["weight"], cfg, gross_limit_t)
            df_t = df_t.merge(opt_out, on="securityid", how="left")
            df_t["raw_target_weight"] = df_t["raw_target_weight"].fillna(df_t["current_weight"])
            df_t["optimizer_status"] = df_t["optimizer_status"].fillna("missing_optimizer_output")

        df_t["gross_limit_t"] = gross_limit_t
        out = apply_caps_and_update(df_t, state_maps, cfg)

        keep_cols = unique_list([
            "date", "minute", cfg["columns"]["datetime_col"], "securityid",
            cfg["columns"]["alpha_col"], "fair_price",
            "mid_price", "bid1", "ask1", "spread", "bid1_volume", "ask1_volume",
            cfg["columns"].get("industry_col"),
        ] + cfg["columns"].get("barra_style_cols", []) + [
            "alpha_bps", "score_for_rank", "has_rank_signal",
            "spread_bps", "buy_edge_bps", "sell_edge_bps",
            "net_buy_edge_bps", "net_sell_edge_bps",
            "valid_market", "has_signal", "pricing_valid",
            "state", "hold_age_min", "cooldown_left_min", "stale_signal_count",
            "gross_limit_t", "current_weight", "current_qty", "sellable_qty",
            "raw_target_weight", "target_weight", "target_qty",
            "delta_qty_raw", "delta_qty_executable",
            "effective_target_qty", "effective_target_weight",
            "side", "selected", "blocked_reason",
            "max_buy_qty_by_ask_volume", "max_sell_qty_by_bid_volume",
            "abs_delta_notional", "gross_weight", "optimizer_status",
        ])

        keep_cols = [x for x in keep_cols if x in out.columns]
        all_out.append(out[keep_cols])

        if i % 50 == 0 or i == len(minutes):
            log_print(
                f"minute {i}/{len(minutes)} {minute}, "
                f"rows={len(out)}, selected={int(out['selected'].sum())}, "
                f"gross={float(out['gross_weight'].sum()):.6f}, "
                f"delta_notional={float(out['abs_delta_notional'].sum()):.2f}, "
                f"states={dict(out['state'].value_counts().head())}, "
                f"status={out['optimizer_status'].iloc[0]}",
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
    log_print("optimizer status counts:", log_path)
    log_print(str(final["optimizer_status"].value_counts().head(20)), log_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = read_yaml(args.config)
    apply_optimizer(cfg)


if __name__ == "__main__":
    main()
