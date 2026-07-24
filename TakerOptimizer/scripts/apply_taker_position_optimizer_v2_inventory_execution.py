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
            ans.loc[remain[remain].index[good]] = dt.loc[good].dt.strftime("%Y%m%d").astype("int64")

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

    m4 = raw.str.match(r"^\d{14,20}$", na=False)
    if m4.any():
        dt.loc[m4] = pd.to_datetime(raw.loc[m4], format="%Y%m%d%H%M%S%f", errors="coerce")

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
    qty = abs(notional) / price
    qty = floor_qty_to_lot(qty, lot_size)
    return int(sign * qty)


def is_trade_minute(minute, ocfg):
    t = pd.Timestamp(minute).strftime("%H:%M:%S")
    before = ocfg.get("block_before_time")
    after = ocfg.get("block_after_time")

    if before and t < str(before):
        return False
    if after and t > str(after):
        return False
    return True


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
        c.get("alpha_col"),
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


def load_minute_prediction(cfg):
    data_cfg = cfg["data"]
    c = cfg["columns"]
    ocfg = cfg["optimizer"]

    path = data_cfg.get("prediction_path")
    alpha_col = c["alpha_col"]

    if not path:
        return None

    path = Path(path)
    if not path.exists():
        print(f"prediction_path not found: {path}, skip external prediction")
        return None

    header = pd.read_csv(path, nrows=0)
    cols = header.columns.tolist()

    if alpha_col not in cols:
        raise ValueError(f"alpha_col {alpha_col} not found in prediction file columns: {cols}")

    needed = unique_list([
        c.get("date_col"),
        c["datetime_col"],
        c["symbol_col"],
        alpha_col,
    ])

    usecols = [x for x in needed if x in cols]

    missing = [x for x in [c["datetime_col"], c["symbol_col"], alpha_col] if x not in cols]
    if missing:
        raise ValueError(f"prediction file missing required columns: {missing}")

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

        chunk[alpha_col] = pd.to_numeric(chunk[alpha_col], errors="coerce")
        chunk = chunk.dropna(subset=[alpha_col])

        chunk = chunk.sort_values(["date", "minute", "securityid", "__dt"])
        chunk = chunk.groupby(["date", "minute", "securityid"], as_index=False).tail(1)
        parts.append(chunk[["date", "minute", "securityid", alpha_col]])

        if i % 10 == 0:
            print(f"prediction chunks loaded: {i}")

    pred = pd.concat(parts, ignore_index=True)
    pred = pred.sort_values(["date", "minute", "securityid"])
    pred = pred.drop_duplicates(["date", "minute", "securityid"], keep="last")

    return pred


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

    barra = barra.drop_duplicates(["date", "securityid"], keep="last")
    return barra


def build_optimizer_input(cfg):
    c = cfg["columns"]
    alpha_col = c["alpha_col"]

    market = load_minute_market(cfg)

    if alpha_col not in market.columns:
        pred = load_minute_prediction(cfg)
        if pred is None:
            raise ValueError(f"alpha_col {alpha_col} not found in market and prediction cannot be loaded")
        market = market.merge(pred, on=["date", "minute", "securityid"], how="left")

    barra = load_barra_proxy(cfg)
    df = market.merge(barra, on=["date", "securityid"], how="left")

    for col in c.get("barra_style_cols", []):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    industry_col = c.get("industry_col")
    if industry_col and industry_col in df.columns:
        df[industry_col] = df[industry_col].fillna("UNKNOWN").astype(str)

    df[alpha_col] = pd.to_numeric(df[alpha_col], errors="coerce")

    return df.sort_values(["minute", "securityid"]).reset_index(drop=True)


def add_execution_features(df, cfg):
    c = cfg["columns"]
    ocfg = cfg["optimizer"]

    alpha_col = c["alpha_col"]
    price_col = c["price_col"]

    mid = pd.to_numeric(df[price_col], errors="coerce")
    bid = pd.to_numeric(df[c.get("bid_col")], errors="coerce") if c.get("bid_col") in df.columns else np.nan
    ask = pd.to_numeric(df[c.get("ask_col")], errors="coerce") if c.get("ask_col") in df.columns else np.nan

    if c.get("spread_col") in df.columns:
        spread = pd.to_numeric(df[c.get("spread_col")], errors="coerce")
    else:
        spread = ask - bid

    df["alpha_bps"] = pd.to_numeric(df[alpha_col], errors="coerce") * float(ocfg.get("alpha_scale_bps", 10000.0))
    df["spread_bps"] = spread / mid * 10000.0

    df["cost_bps"] = (
        df["spread_bps"].fillna(np.inf) / 2.0
        + float(ocfg.get("taker_fee_bps", 0.0))
        + float(ocfg.get("impact_bps", 0.0))
    )

    df["net_alpha_bps"] = df["alpha_bps"] - df["cost_bps"]

    max_spread_bps = float(ocfg.get("max_spread_bps", 1e9))
    min_net_alpha_bps = float(ocfg.get("min_net_alpha_bps", 0.0))

    valid_market = (
        np.isfinite(mid)
        & (mid > 0)
        & np.isfinite(df["spread_bps"])
        & (df["spread_bps"] >= 0)
        & (df["spread_bps"] <= max_spread_bps)
        & df[alpha_col].notna()
    )

    df["valid_market"] = valid_market
    df["eligible_buy"] = valid_market & (df["net_alpha_bps"] >= min_net_alpha_bps)
    df["opt_alpha_bps"] = df["net_alpha_bps"].where(df["eligible_buy"], -abs(min_net_alpha_bps))

    return df


def dynamic_gross_limit(df_t, cfg):
    ocfg = cfg["optimizer"]

    base_gross = float(ocfg.get("base_gross_limit", ocfg.get("gross_limit", 0.05)))
    max_gross = float(ocfg.get("max_gross_limit", ocfg.get("gross_limit", base_gross)))

    full_edge = float(ocfg.get("net_alpha_bps_for_max_gross", 6.0))

    x = pd.to_numeric(df_t.loc[df_t["eligible_buy"], "net_alpha_bps"], errors="coerce")
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return 0.0

    top_n = int(ocfg.get("gross_signal_top_n", 10))
    avg_edge = float(x.sort_values(ascending=False).head(top_n).mean())

    if full_edge <= 0:
        ratio = 1.0
    else:
        ratio = np.clip(avg_edge / full_edge, 0.0, 1.0)

    return float(base_gross + (max_gross - base_gross) * ratio)


def fallback_alpha_weights(df_t, prev_weight_map, cfg, gross_limit_t):
    ocfg = cfg["optimizer"]

    alpha = pd.to_numeric(df_t["opt_alpha_bps"], errors="coerce").fillna(-1e9)
    eligible = df_t["eligible_buy"].fillna(False).to_numpy(dtype=bool)

    w = np.zeros(len(df_t))

    if eligible.sum() > 0 and gross_limit_t > 0:
        a = alpha.clip(lower=0.0)
        a = a.where(df_t["eligible_buy"], 0.0)

        if a.sum() > 0:
            w = (a / a.sum() * gross_limit_t).to_numpy(dtype=float)

    single = float(ocfg.get("single_name_limit", 0.005))
    w = np.clip(w, 0.0, single)

    gross = np.abs(w).sum()
    if gross > gross_limit_t and gross > 0:
        w = w / gross * gross_limit_t

    out = df_t[["securityid"]].copy()
    out["raw_target_weight"] = w
    out["optimizer_status"] = "fallback_alpha_rescale"
    return out


def solve_one_timestamp(df_t, prev_weight_map, cfg, gross_limit_t):
    c = cfg["columns"]
    ocfg = cfg["optimizer"]

    industry_col = c.get("industry_col", "industryID1")
    style_cols = [x for x in c.get("barra_style_cols", []) if x in df_t.columns]

    df_t = df_t.copy()
    n = len(df_t)

    if n < int(ocfg.get("min_names_per_timestamp", 20)):
        out = df_t[["securityid"]].copy()
        out["raw_target_weight"] = 0.0
        out["optimizer_status"] = "too_few_names"
        return out

    if df_t["eligible_buy"].sum() == 0 or gross_limit_t <= 0:
        out = df_t[["securityid"]].copy()
        out["raw_target_weight"] = 0.0
        out["optimizer_status"] = "no_eligible_buy"
        return out

    try:
        import cvxpy as cp
    except Exception:
        return fallback_alpha_weights(df_t, prev_weight_map, cfg, gross_limit_t)

    symbols = df_t["securityid"].astype(str).tolist()

    alpha = pd.to_numeric(df_t["opt_alpha_bps"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    alpha = alpha - np.nanmean(alpha)
    alpha_std = np.nanstd(alpha)

    if alpha_std > 0:
        alpha = alpha / alpha_std

    eligible = df_t["eligible_buy"].fillna(False).to_numpy(dtype=bool)
    w_prev = np.array([prev_weight_map.get(s, 0.0) for s in symbols], dtype=float)

    single_limit = float(ocfg.get("single_name_limit", 0.005))
    turnover_limit = float(ocfg.get("turnover_limit", 0.015))

    risk_aversion = float(ocfg.get("risk_aversion", 1.0))
    barra_risk_aversion = float(ocfg.get("barra_risk_aversion", 0.2))
    industry_risk_aversion = float(ocfg.get("industry_risk_aversion", 0.1))
    turnover_penalty = float(ocfg.get("turnover_penalty", 0.05))

    risk_diag = np.ones(n)

    if "volatility_z" in df_t.columns:
        vol = pd.to_numeric(df_t["volatility_z"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        risk_diag = 1.0 + np.square(vol)

    w = cp.Variable(n)

    objective_terms = [
        alpha @ w,
        -risk_aversion * cp.sum(cp.multiply(risk_diag, cp.square(w))),
        -turnover_penalty * cp.norm1(w - w_prev),
    ]

    X_style = None
    if len(style_cols) > 0:
        xs = []
        for col in style_cols:
            x = pd.to_numeric(df_t[col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            xs.append(x)
        X_style = np.vstack(xs).T
        objective_terms.append(-barra_risk_aversion * cp.sum_squares(X_style.T @ w))

    X_industry = None
    if industry_col in df_t.columns:
        industry = df_t[industry_col].where(df_t[industry_col].notna(), "UNKNOWN").map(str)
        X_industry = pd.get_dummies(industry).to_numpy(dtype=float)
        objective_terms.append(-industry_risk_aversion * cp.sum_squares(X_industry.T @ w))

    constraints = [
        w >= 0.0,
        w <= single_limit,
        cp.sum(w) <= gross_limit_t,
        cp.norm1(w - w_prev) <= turnover_limit,
    ]

    zero_idx = np.where(~eligible)[0]
    if len(zero_idx) > 0:
        constraints.append(w[zero_idx] == 0.0)

    style_limit = float(ocfg.get("style_limit", 0.03))
    if X_style is not None:
        for j in range(X_style.shape[1]):
            x = X_style[:, j]
            constraints += [
                x @ w <= style_limit,
                x @ w >= -style_limit,
            ]

    industry_limit = float(ocfg.get("industry_limit", 0.03))
    if X_industry is not None:
        constraints += [
            X_industry.T @ w <= industry_limit,
            X_industry.T @ w >= -industry_limit,
        ]

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
            out["raw_target_weight"] = 0.0
            out["optimizer_status"] = "solve_error:" + str(e)[:100]
            return out

    if w.value is None:
        return fallback_alpha_weights(df_t, prev_weight_map, cfg, gross_limit_t)

    raw_target_weight = np.asarray(w.value).reshape(-1)
    raw_target_weight[np.abs(raw_target_weight) < 1e-10] = 0.0

    out = df_t[["securityid"]].copy()
    out["raw_target_weight"] = raw_target_weight
    out["optimizer_status"] = status
    return out


def apply_execution_caps(out, prev_qty_map, prev_weight_map, cfg):
    c = cfg["columns"]
    ocfg = cfg["optimizer"]

    capital = float(ocfg.get("capital", 1.0))
    lot_size = int(ocfg.get("lot_size", 100))
    smoothing = float(ocfg.get("rebalance_smoothing", 1.0))
    volume_cap_ratio = float(ocfg.get("volume_cap_ratio", 0.05))
    min_trade_notional = float(ocfg.get("min_trade_notional", 0.0))
    enforce_sellable = bool(ocfg.get("enforce_sellable_inventory", True))

    price_col = c["price_col"]
    bid_col = c.get("bid_col")
    ask_col = c.get("ask_col")
    bid_volume_col = c.get("bid_volume_col")
    ask_volume_col = c.get("ask_volume_col")
    limit_up_col = c.get("limit_up_col")
    limit_down_col = c.get("limit_down_col")

    rows = []

    for _, row in out.iterrows():
        sym = str(row["securityid"])

        px = pd.to_numeric(pd.Series([row.get(price_col, np.nan)]), errors="coerce").iloc[0]
        bid = pd.to_numeric(pd.Series([row.get(bid_col, np.nan)]), errors="coerce").iloc[0] if bid_col else np.nan
        ask = pd.to_numeric(pd.Series([row.get(ask_col, np.nan)]), errors="coerce").iloc[0] if ask_col else np.nan

        bid_vol = pd.to_numeric(pd.Series([row.get(bid_volume_col, np.nan)]), errors="coerce").iloc[0] if bid_volume_col else np.nan
        ask_vol = pd.to_numeric(pd.Series([row.get(ask_volume_col, np.nan)]), errors="coerce").iloc[0] if ask_volume_col else np.nan

        limit_up = pd.to_numeric(pd.Series([row.get(limit_up_col, np.nan)]), errors="coerce").iloc[0] if limit_up_col else np.nan
        limit_down = pd.to_numeric(pd.Series([row.get(limit_down_col, np.nan)]), errors="coerce").iloc[0] if limit_down_col else np.nan

        prev_w = float(prev_weight_map.get(sym, 0.0))
        current_qty = int(prev_qty_map.get(sym, 0))
        sellable_qty = max(0, current_qty)

        raw_w = float(row.get("raw_target_weight", 0.0))
        target_w = (1.0 - smoothing) * prev_w + smoothing * raw_w
        target_w = max(0.0, target_w)

        target_notional = target_w * capital
        target_qty = floor_signed_to_lot(target_notional, px, lot_size)
        delta_qty_raw = int(target_qty - current_qty)

        max_buy_qty_by_ask_volume = floor_qty_to_lot(ask_vol * volume_cap_ratio, lot_size) if np.isfinite(ask_vol) else abs(delta_qty_raw)
        max_sell_qty_by_bid_volume = floor_qty_to_lot(bid_vol * volume_cap_ratio, lot_size) if np.isfinite(bid_vol) else abs(delta_qty_raw)

        delta_qty_executable = delta_qty_raw
        blocked_reason = "none"

        trade_notional = abs(delta_qty_raw) * px if np.isfinite(px) else 0.0

        if delta_qty_raw == 0:
            delta_qty_executable = 0
            blocked_reason = "no_delta"

        elif not bool(row.get("valid_market", False)):
            delta_qty_executable = 0
            blocked_reason = "invalid_market_or_spread"

        elif trade_notional < min_trade_notional:
            delta_qty_executable = 0
            blocked_reason = "below_min_trade_notional"

        elif delta_qty_raw > 0:
            if np.isfinite(limit_up) and np.isfinite(ask) and ask >= limit_up:
                delta_qty_executable = 0
                blocked_reason = "limit_up_block"
            else:
                delta_qty_executable = min(delta_qty_raw, max_buy_qty_by_ask_volume)
                if delta_qty_executable <= 0:
                    blocked_reason = "blocked_by_ask_volume"
                elif delta_qty_executable < delta_qty_raw:
                    blocked_reason = "clipped_by_ask_volume"

        else:
            raw_sell_qty = abs(delta_qty_raw)

            if np.isfinite(limit_down) and np.isfinite(bid) and bid <= limit_down:
                delta_qty_executable = 0
                blocked_reason = "limit_down_block"
            else:
                max_sell = max_sell_qty_by_bid_volume
                if enforce_sellable:
                    max_sell = min(max_sell, sellable_qty)

                sell_exec = min(raw_sell_qty, max_sell)
                delta_qty_executable = -int(sell_exec)

                if sell_exec <= 0:
                    if enforce_sellable and sellable_qty <= 0:
                        blocked_reason = "blocked_by_no_sellable_inventory"
                    else:
                        blocked_reason = "blocked_by_bid_volume"
                elif sell_exec < raw_sell_qty:
                    blocked_reason = "clipped_by_sellable_or_bid_volume"

        effective_target_qty = int(current_qty + delta_qty_executable)
        effective_target_notional = effective_target_qty * px if np.isfinite(px) else 0.0
        effective_target_weight = effective_target_notional / capital if capital > 0 else 0.0

        if delta_qty_executable > 0:
            side = "BUY"
        elif delta_qty_executable < 0:
            side = "SELL"
        else:
            side = "NONE"

        row = row.copy()
        row["raw_target_weight"] = raw_w
        row["target_weight"] = target_w
        row["target_notional"] = target_notional
        row["current_qty"] = current_qty
        row["sellable_qty"] = sellable_qty
        row["target_qty"] = target_qty
        row["delta_qty_raw"] = delta_qty_raw
        row["delta_qty_executable"] = int(delta_qty_executable)
        row["effective_target_qty"] = effective_target_qty
        row["effective_target_weight"] = effective_target_weight
        row["side"] = side
        row["max_buy_qty_by_ask_volume"] = int(max_buy_qty_by_ask_volume) if np.isfinite(max_buy_qty_by_ask_volume) else 0
        row["max_sell_qty_by_bid_volume"] = int(max_sell_qty_by_bid_volume) if np.isfinite(max_sell_qty_by_bid_volume) else 0
        row["blocked_reason"] = blocked_reason
        row["position_source"] = "theoretical"
        row["abs_delta_notional"] = abs(delta_qty_executable) * px if np.isfinite(px) else 0.0
        row["gross_weight"] = abs(effective_target_weight)
        row["selected"] = effective_target_qty != 0

        rows.append(row)

    final = pd.DataFrame(rows)

    for _, row in final.iterrows():
        sym = str(row["securityid"])
        prev_qty_map[sym] = int(row["effective_target_qty"])
        prev_weight_map[sym] = float(row["effective_target_weight"])

    return final


def apply_optimizer(cfg):
    data_cfg = cfg["data"]
    c = cfg["columns"]

    output_path = data_cfg["output_path"]
    log_path = data_cfg.get("log_path")

    mkdir_parent(output_path)
    if log_path:
        mkdir_parent(log_path)
        Path(log_path).write_text("", encoding="utf-8")

    df = build_optimizer_input(cfg)
    df = add_execution_features(df, cfg)

    alpha_col = c["alpha_col"]

    log_print("===== input summary =====", log_path)
    log_print(f"rows: {len(df)}", log_path)
    log_print(f"date range: {df['date'].min()} -> {df['date'].max()}", log_path)
    log_print(f"minute range: {df['minute'].min()} -> {df['minute'].max()}", log_path)
    log_print(f"num minutes: {df['minute'].nunique()}", log_path)
    log_print(f"num symbols: {df['securityid'].nunique()}", log_path)
    log_print(f"alpha non-null rate: {df[alpha_col].notna().mean():.6f}", log_path)
    log_print(f"eligible buy rate: {df['eligible_buy'].mean():.6f}", log_path)

    prev_weight_map = defaultdict(float)
    prev_qty_map = defaultdict(int)

    all_out = []
    minutes = sorted(df["minute"].dropna().unique())

    for i, minute in enumerate(minutes, 1):
        df_t = df[df["minute"] == minute].copy()

        if not is_trade_minute(minute, cfg["optimizer"]):
            df_t["raw_target_weight"] = df_t["securityid"].astype(str).map(prev_weight_map).fillna(0.0).astype(float)
            df_t["optimizer_status"] = "blocked_time"
            df_t["gross_limit_t"] = float(np.abs(df_t["raw_target_weight"]).sum())
        else:
            gross_limit_t = dynamic_gross_limit(df_t, cfg)
            opt_out = solve_one_timestamp(df_t, prev_weight_map, cfg, gross_limit_t)

            df_t = df_t.merge(opt_out, on="securityid", how="left")
            df_t["raw_target_weight"] = df_t["raw_target_weight"].fillna(0.0)
            df_t["optimizer_status"] = df_t["optimizer_status"].fillna("missing_optimizer_output")
            df_t["gross_limit_t"] = gross_limit_t

        out = apply_execution_caps(df_t, prev_qty_map, prev_weight_map, cfg)

        out["alpha_rank"] = pd.to_numeric(out[alpha_col], errors="coerce").rank(method="first", ascending=False)

        keep_cols = unique_list([
            "date",
            "minute",
            c["datetime_col"],
            "securityid",
            alpha_col,
            "alpha_bps",
            "spread_bps",
            "cost_bps",
            "net_alpha_bps",
            "fair_price",
            "buy_edge_bps",
            "sell_edge_bps",
            "eligible_buy",
            c["price_col"],
            c.get("bid_col"),
            c.get("ask_col"),
            c.get("spread_col"),
            c.get("bid_volume_col"),
            c.get("ask_volume_col"),
            c.get("limit_up_col"),
            c.get("limit_down_col"),
            c.get("industry_col"),
        ] + c.get("barra_style_cols", []) + [
            "alpha_rank",
            "gross_limit_t",
            "raw_target_weight",
            "target_weight",
            "effective_target_weight",
            "gross_weight",
            "target_notional",
            "current_qty",
            "sellable_qty",
            "target_qty",
            "delta_qty_raw",
            "delta_qty_executable",
            "effective_target_qty",
            "side",
            "selected",
            "max_buy_qty_by_ask_volume",
            "max_sell_qty_by_bid_volume",
            "blocked_reason",
            "position_source",
            "abs_delta_notional",
            "valid_market",
            "optimizer_status",
        ])

        keep_cols = [x for x in keep_cols if x in out.columns]
        all_out.append(out[keep_cols])

        if i % 50 == 0 or i == len(minutes):
            log_print(
                f"minute {i}/{len(minutes)} {minute}, "
                f"rows={len(out)}, selected={int(out['selected'].sum())}, "
                f"gross={float(out['gross_weight'].sum()):.6f}, "
                f"delta_notional={float(out['abs_delta_notional'].sum()):.2f}, "
                f"eligible={int(out['eligible_buy'].sum())}, "
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
