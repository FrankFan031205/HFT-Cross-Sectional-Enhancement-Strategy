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

def resolve_csv_files(path_like):
    p = Path(path_like)
    if p.is_file():
        return [p]
    if p.is_dir():
        files = sorted(p.glob("*.csv"))
        if not files:
            raise FileNotFoundError(f"no csv files found in directory: {p}")
        return files

    files = sorted(Path(".").glob(str(path_like)))
    if not files:
        raise FileNotFoundError(f"path not found: {path_like}")
    return files


def get_date_bounds(cfg):
    data_cfg = cfg.get("data", {})
    start = data_cfg.get("start_date")
    end = data_cfg.get("end_date")
    start = int(start) if start not in [None, ""] else None
    end = int(end) if end not in [None, ""] else None
    return start, end


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
    interval = int(ocfg.get("portfolio_rebalance_interval_min", 5))
    if interval <= 1:
        return True
    t = pd.Timestamp(minute)
    return (t.hour * 60 + t.minute) % interval == 0


def load_minute_market(cfg):
    data_cfg = cfg["data"]
    c = cfg["columns"]
    ocfg = cfg["optimizer"]

    files = resolve_csv_files(data_cfg["market_data_path"])
    start_date, end_date = get_date_bounds(cfg)

    header = pd.read_csv(files[0], nrows=0)
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

    for fp_i, fp in enumerate(files, 1):
        for i, chunk in enumerate(pd.read_csv(fp, usecols=usecols, chunksize=chunksize), 1):
            chunk["date"] = normalize_date_series(chunk[c["date_col"]])

            if start_date is not None:
                chunk = chunk[chunk["date"] >= start_date]
            if end_date is not None:
                chunk = chunk[chunk["date"] <= end_date]
            if len(chunk) == 0:
                continue

            chunk["__dt"] = parse_datetime_series(chunk[c["datetime_col"]])
            chunk["minute"] = chunk["__dt"].dt.floor("min")
            chunk["securityid"] = normalize_symbol_series(chunk[c["symbol_col"]])

            chunk = chunk.sort_values(["date", "minute", "securityid", "__dt"])
            chunk = chunk.groupby(["date", "minute", "securityid"], as_index=False).tail(1)
            parts.append(chunk)

        if fp_i % 5 == 0 or fp_i == len(files):
            print(f"market files loaded: {fp_i}/{len(files)} {fp}")

    if not parts:
        raise RuntimeError("no market rows loaded after date filtering")

    df = pd.concat(parts, ignore_index=True)
    df = df.sort_values(["date", "minute", "securityid", "__dt"])
    df = df.groupby(["date", "minute", "securityid"], as_index=False).tail(1)

    return df.reset_index(drop=True)

def load_alpha_file(cfg, path_key, col_key, out_col):
    data_cfg = cfg["data"]
    c = cfg["columns"]
    ocfg = cfg["optimizer"]

    path = data_cfg.get(path_key)
    alpha_col = c.get(col_key)

    if not path or not alpha_col:
        return None

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path_key} not found: {path}")

    header = pd.read_csv(path, nrows=0)
    cols = header.columns.tolist()

    needed = unique_list([
        c["date_col"],
        c["datetime_col"],
        c["symbol_col"],
        alpha_col,
    ])

    usecols = [x for x in needed if x in cols]
    missing = [x for x in [c["datetime_col"], c["symbol_col"], alpha_col] if x not in cols]

    if missing:
        raise ValueError(f"{path_key} missing required columns: {missing}")

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

        chunk[out_col] = pd.to_numeric(chunk[alpha_col], errors="coerce")
        chunk = chunk.dropna(subset=[out_col])
        chunk = chunk.sort_values(["date", "minute", "securityid", "__dt"])
        chunk = chunk.groupby(["date", "minute", "securityid"], as_index=False).tail(1)

        parts.append(chunk[["date", "minute", "securityid", out_col]])

        if i % 10 == 0:
            print(f"{path_key} chunks loaded: {i}")

    if not parts:
        return None

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


def add_cs_zscore(df, value_col, out_col):
    x = pd.to_numeric(df[value_col], errors="coerce")

    mean = x.groupby(df["minute"]).transform("mean")
    std = x.groupby(df["minute"]).transform("std")

    z = (x - mean) / std.replace(0, np.nan)
    df[out_col] = z.fillna(0.0)
    return df


def build_input(cfg):
    market = load_minute_market(cfg)
    global_alpha = load_alpha_file(cfg, "global_alpha_path", "global_alpha_col", "global_alpha_raw")
    local_alpha = load_alpha_file(cfg, "local_alpha_path", "local_alpha_col", "local_alpha_raw")
    pricing = load_pricing_optional(cfg)

    df = market.copy()

    if global_alpha is not None:
        df = df.merge(global_alpha, on=["date", "minute", "securityid"], how="left")

    if local_alpha is not None:
        df = df.merge(local_alpha, on=["date", "minute", "securityid"], how="left")

    if pricing is not None:
        df = df.merge(pricing, on=["date", "minute", "securityid"], how="left")

    df = df.sort_values(["securityid", "minute"]).reset_index(drop=True)

    if "global_alpha_raw" in df.columns:
        df = add_cs_zscore(df, "global_alpha_raw", "global_alpha_z")
    else:
        df["global_alpha_raw"] = np.nan
        df["global_alpha_z"] = np.nan

    if "local_alpha_raw" in df.columns:
        df = add_cs_zscore(df, "local_alpha_raw", "local_alpha_z")
    else:
        df["local_alpha_raw"] = df["global_alpha_raw"]
        df["local_alpha_z"] = df["global_alpha_z"]

    halflife = float(cfg["optimizer"].get("global_ewma_halflife_min", 15))
    df["global_score"] = (
        df.groupby("securityid")["global_alpha_z"]
        .transform(lambda s: s.ewm(halflife=halflife, adjust=False, min_periods=1).mean())
    )

    return df.sort_values(["minute", "securityid"]).reset_index(drop=True)


def add_risk_overlay_features(df, cfg):
    """
    v2.9 risk overlay:
    1) regime_aware_target_gross:
       - base_target_gross
       - previous-day daily loss derisk
       - intraday market regime derisk
    2) optional alpha quality derisk from old risk_overlay config
    """
    ocfg = cfg["optimizer"]
    c = cfg["columns"]

    regime_cfg = ocfg.get("regime_aware_target_gross", {})
    old_risk_cfg = ocfg.get("risk_overlay", {})

    enabled = bool(regime_cfg) or bool(old_risk_cfg.get("enabled", False))

    if not enabled:
        df["market_ret_1m"] = 0.0
        df["up_ratio"] = 0.5
        df["market_volatility"] = 0.0
        df["market_cum_ret"] = 0.0
        df["market_cum_up_ratio"] = 0.5
        df["minutes_since_open"] = 0
        df["alpha_spread"] = 1.0

        df["daily_loss_multiplier"] = 1.0
        df["intraday_market_multiplier"] = 1.0
        df["alpha_quality_multiplier"] = 1.0
        df["market_regime_multiplier"] = 1.0
        df["risk_multiplier"] = 1.0

        df["daily_loss_block_new_buy"] = False
        df["intraday_market_block_new_buy"] = False
        df["block_new_entry_by_risk"] = False
        return df

    price_col = c["price_col"]
    date_col = c.get("date_col", "date")
    symbol_col = c.get("symbol_col", "securityid")

    df = df.sort_values([symbol_col, "minute"]).copy()

    px = pd.to_numeric(df[price_col], errors="coerce")

    # 1-minute symbol return
    df["symbol_ret_1m"] = px.groupby(df[symbol_col]).pct_change().replace([np.inf, -np.inf], np.nan)

    g_min = df.groupby("minute")
    market_ret = g_min["symbol_ret_1m"].mean().rename("market_ret_1m")
    up_ratio = g_min["symbol_ret_1m"].apply(lambda x: (x > 0).mean()).rename("up_ratio")
    market_vol = g_min["symbol_ret_1m"].std().fillna(0.0).rename("market_volatility")

    tmp = pd.concat([market_ret, up_ratio, market_vol], axis=1).reset_index()
    df = df.merge(tmp, on="minute", how="left")

    # intraday cumulative return from today's first available price
    first_px = df.groupby([date_col, symbol_col])[price_col].transform("first")
    df["symbol_ret_from_open"] = pd.to_numeric(df[price_col], errors="coerce") / pd.to_numeric(first_px, errors="coerce") - 1.0

    g_dm = df.groupby([date_col, "minute"])
    intraday = g_dm["symbol_ret_from_open"].agg(
        market_cum_ret="mean",
        market_cum_up_ratio=lambda x: (x > 0).mean(),
    ).reset_index()

    # minute index within each date
    minute_table = intraday[[date_col, "minute"]].drop_duplicates().sort_values([date_col, "minute"])
    minute_table["minutes_since_open"] = minute_table.groupby(date_col).cumcount()
    intraday = intraday.merge(minute_table, on=[date_col, "minute"], how="left")

    df = df.merge(intraday, on=[date_col, "minute"], how="left")

    # alpha quality: cross-sectional top-bottom spread
    if "global_score" in df.columns:
        def qspread(x):
            x = pd.to_numeric(x, errors="coerce").dropna()
            if len(x) < 20:
                return 0.0
            return float(x.quantile(0.90) - x.quantile(0.10))

        alpha_spread = df.groupby("minute")["global_score"].apply(qspread).rename("alpha_spread").reset_index()
        df = df.merge(alpha_spread, on="minute", how="left")
    else:
        df["alpha_spread"] = 1.0

    # ------------------------------------------------------------
    # A. intraday market regime derisk
    # ------------------------------------------------------------
    df["intraday_market_multiplier"] = 1.0
    df["intraday_market_block_new_buy"] = False

    intraday_cfg = regime_cfg.get("intraday_market_regime", {})
    if intraday_cfg.get("enabled", False):
        min_minutes = int(intraday_cfg.get("market_regime_min_minutes", 30))
        market_ret_stop = float(intraday_cfg.get("market_ret_stop", -0.015))
        up_ratio_stop = float(intraday_cfg.get("market_up_ratio_stop", 0.20))
        scale = float(intraday_cfg.get("market_target_scale", 0.20))
        block_new_buy = bool(intraday_cfg.get("block_new_buy", True))

        cond_intraday = (
            (df["minutes_since_open"].fillna(0) >= min_minutes)
            & (
                (df["market_cum_ret"].fillna(0.0) <= market_ret_stop)
                | (df["market_cum_up_ratio"].fillna(0.5) <= up_ratio_stop)
            )
        )

        df.loc[cond_intraday, "intraday_market_multiplier"] = scale
        if block_new_buy:
            df.loc[cond_intraday, "intraday_market_block_new_buy"] = True

    # ------------------------------------------------------------
    # B. previous daily loss derisk
    # ------------------------------------------------------------
    df["daily_loss_multiplier"] = 1.0
    df["daily_loss_block_new_buy"] = False

    daily_cfg = regime_cfg.get("daily_loss_derisk", {})
    daily_pnl_path = cfg.get("data", {}).get("daily_pnl_path", None)

    if daily_cfg.get("enabled", False) and daily_pnl_path:
        from pathlib import Path as _Path

        pnl_path = _Path(daily_pnl_path)
        if pnl_path.exists():
            pnl = pd.read_csv(pnl_path)

            # accepted column names
            if "date" not in pnl.columns:
                raise ValueError(f"daily_pnl_path must contain date column: {daily_pnl_path}")

            pnl_col = None
            for cand in ["net_pnl", "total_net_pnl", "daily_net_pnl", "pnl"]:
                if cand in pnl.columns:
                    pnl_col = cand
                    break

            if pnl_col is None:
                raise ValueError(
                    f"daily_pnl_path must contain one of net_pnl/total_net_pnl/daily_net_pnl/pnl: {daily_pnl_path}"
                )

            pnl["date"] = pnl["date"].astype(int)
            pnl[pnl_col] = pd.to_numeric(pnl[pnl_col], errors="coerce")

            stop = float(daily_cfg.get("daily_loss_stop", -500000))
            cooldown_days = int(daily_cfg.get("cooldown_days", 2))
            scale = float(daily_cfg.get("risk_off_target_scale", 0.30))
            block_new_buy = bool(daily_cfg.get("block_new_buy", True))

            # Sequential-safe trading calendar:
            # In live daily mode, df may only contain today's date.
            # Therefore we must build calendar from both current df dates and historical pnl dates.
            df_dates = sorted(pd.Series(df[date_col].dropna().astype(int).unique()).tolist())
            pnl_dates = sorted(pd.Series(pnl["date"].dropna().astype(int).unique()).tolist())
            all_dates = sorted(set(df_dates) | set(pnl_dates))

            bad_dates = set(pnl.loc[pnl[pnl_col] <= stop, "date"].astype(int).tolist())

            risk_off_dates = set()
            for bd in bad_dates:
                if bd not in all_dates:
                    continue
                idx = all_dates.index(bd)
                for j in range(1, cooldown_days + 1):
                    if idx + j < len(all_dates):
                        risk_off_dates.add(all_dates[idx + j])

            cond_daily = df[date_col].astype(int).isin(risk_off_dates)
            df.loc[cond_daily, "daily_loss_multiplier"] = scale
            if block_new_buy:
                df.loc[cond_daily, "daily_loss_block_new_buy"] = True

    # ------------------------------------------------------------
    # C. optional old alpha quality derisk
    # ------------------------------------------------------------
    df["alpha_quality_multiplier"] = 1.0

    if old_risk_cfg.get("enabled", False):
        min_alpha_spread = float(old_risk_cfg.get("min_alpha_spread", 0.0))
        weak_alpha_multiplier = float(old_risk_cfg.get("weak_alpha_multiplier", 1.0))
        weak_alpha = df["alpha_spread"].fillna(0.0) < min_alpha_spread
        df.loc[weak_alpha, "alpha_quality_multiplier"] = weak_alpha_multiplier

    # Combined multiplier
    df["market_regime_multiplier"] = (
        df["intraday_market_multiplier"].astype(float)
        * df["daily_loss_multiplier"].astype(float)
    )

    df["risk_multiplier"] = (
        df["market_regime_multiplier"].astype(float)
        * df["alpha_quality_multiplier"].astype(float)
    ).clip(lower=0.0, upper=1.0)

    # block new buy
    df["block_new_entry_by_risk"] = (
        df["intraday_market_block_new_buy"].astype(bool)
        | df["daily_loss_block_new_buy"].astype(bool)
    )

    # also support old risk_overlay threshold
    if old_risk_cfg.get("enabled", False):
        block_entry_threshold = float(old_risk_cfg.get("block_entry_multiplier_threshold", -1.0))
        if block_entry_threshold >= 0:
            df["block_new_entry_by_risk"] = (
                df["block_new_entry_by_risk"]
                | (df["risk_multiplier"] <= block_entry_threshold)
            )

    return df

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

    df["has_global_signal"] = df["global_score"].notna()
    df["has_local_signal"] = df["local_alpha_z"].notna()

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

    global_top_n = int(ocfg.get("global_top_n", 50))
    global_exit_rank_n = int(ocfg.get("global_exit_rank_n", 70))

    regime_gross_cfg = ocfg.get("regime_aware_target_gross", {})
    base_target_gross = float(regime_gross_cfg.get("base_target_gross", ocfg.get("target_gross_limit", 0.60)))
    max_gross = float(ocfg.get("max_gross_limit", base_target_gross))
    single_limit = float(ocfg.get("single_name_limit", 0.025))

    # risk overlay multiplier is minute-level; use median in current minute
    if "risk_multiplier" in df_t.columns:
        risk_multiplier_t = float(pd.to_numeric(df_t["risk_multiplier"], errors="coerce").median())
        if not np.isfinite(risk_multiplier_t):
            risk_multiplier_t = 1.0
    else:
        risk_multiplier_t = 1.0

    target_gross = base_target_gross * risk_multiplier_t

    min_hold = int(ocfg.get("min_hold_minutes", 10))
    cooldown = int(ocfg.get("trade_cooldown_minutes", 2))

    minute = df_t["minute"].iloc[0]
    trade_allowed = is_trade_minute(minute, ocfg)
    rebalance_allowed = trade_allowed and is_rebalance_minute(minute, ocfg)

    df_t = df_t.copy()
    df_t["global_rank"] = pd.to_numeric(df_t["global_score"], errors="coerce").rank(method="first", ascending=False)
    df_t["local_rank"] = pd.to_numeric(df_t["local_alpha_z"], errors="coerce").rank(method="first", ascending=False)

    df_t["current_qty"] = df_t["securityid"].astype(str).map(state_qty).fillna(0).astype(int)
    df_t["current_weight"] = df_t["securityid"].astype(str).map(state_weight).fillna(0.0).astype(float)

    eligible_global = df_t["has_global_signal"].fillna(False) & df_t["valid_market"].fillna(False)
    top_mask = eligible_global & (df_t["global_rank"] <= global_top_n)

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

        grank = row.get("global_rank", np.nan)
        has_global = bool(row.get("has_global_signal", False))
        valid_market = bool(row.get("valid_market", False))

        entry_i = state_entry_i.get(sym)
        last_trade_i = state_trade_i.get(sym)

        hold_age = 999999 if entry_i is None else max(0, minute_i - entry_i)
        cooldown_left = 0 if last_trade_i is None else max(0, cooldown - (minute_i - last_trade_i))

        if not trade_allowed or not rebalance_allowed:
            dw = current_w
            st = "HOLD" if current_qty > 0 else "FLAT"
        else:
            is_global_top = bool(has_global and valid_market and np.isfinite(grank) and grank <= global_top_n)
            is_global_hold = bool(has_global and valid_market and np.isfinite(grank) and grank <= global_exit_rank_n)

            block_entry_by_risk = bool(row.get("block_new_entry_by_risk", False))

            if current_qty <= 0:
                if is_global_top and cooldown_left == 0 and not block_entry_by_risk:
                    dw = per_name_weight
                    st = "ENTRY"
                else:
                    dw = 0.0
                    st = "FLAT_RISK_BLOCK" if block_entry_by_risk and is_global_top else "FLAT"
            else:
                if hold_age < min_hold:
                    dw = current_w
                    st = "HOLD"
                elif is_global_top and cooldown_left == 0:
                    dw = max(current_w, per_name_weight)
                    st = "ADD" if dw > current_w else "HOLD"
                elif is_global_hold:
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
    df_t["base_target_gross_t"] = base_target_gross
    df_t["risk_multiplier_t"] = risk_multiplier_t
    df_t["per_name_weight_t"] = per_name_weight

    return df_t


def local_buy_allowed(row, cfg):
    ocfg = cfg["optimizer"]

    if not bool(ocfg.get("use_local_buy_gate", True)):
        return True

    local_buy_top_n = int(ocfg.get("local_buy_top_n", 70))
    edge_min = float(ocfg.get("local_buy_edge_min_bps", -5.0))

    local_rank = row.get("local_rank", np.nan)
    buy_edge = row.get("buy_edge_bps", np.nan)

    if np.isfinite(local_rank) and local_rank <= local_buy_top_n:
        return True

    if np.isfinite(buy_edge) and buy_edge >= edge_min:
        return True

    return False


def apply_execution(df_t, state_qty, state_weight, state_entry_i, state_trade_i, minute_i, cfg):
    c = cfg["columns"]
    ocfg = cfg["optimizer"]

    capital = float(ocfg.get("capital", 1.0))
    lot = int(ocfg.get("lot_size", 100))
    smoothing = float(ocfg.get("rebalance_smoothing", 0.80))

    # Align with best TakerModel execution params
    entry_smoothing = float(ocfg.get("entry_rebalance_ratio", ocfg.get("entry_smoothing", smoothing)))
    exit_smoothing = float(ocfg.get("exit_rebalance_ratio", ocfg.get("exit_smoothing", smoothing)))
    reduce_when_target_smaller = bool(ocfg.get("reduce_when_target_smaller", False))

    band = float(ocfg.get("target_weight_band", 0.00003))

    min_notional = float(ocfg.get("min_trade_notional", 10000))
    entry_min_notional = float(ocfg.get("entry_min_abs_delta_notional", min_notional))
    exit_min_notional = float(ocfg.get("exit_min_abs_delta_notional", 0.0))

    entry_max_spread_bps = float(ocfg.get("entry_max_spread_bps", ocfg.get("max_spread_bps", 1e9)))
    exit_max_spread_bps = float(ocfg.get("exit_max_spread_bps", ocfg.get("max_spread_bps", 1e9)))

    volume_cap_ratio = float(ocfg.get("volume_cap_ratio", 0.35))
    turnover_limit = float(ocfg.get("turnover_limit", 0.08))
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

        # Do not churn just because target becomes slightly smaller.
        # Only EXIT / risk-driven target reduction should actively reduce.
        if desired_w < current_w and st != "EXIT" and not reduce_when_target_smaller:
            desired_w = current_w

        if abs(desired_w - current_w) < band:
            target_w = current_w
        else:
            if st == "EXIT":
                sm = exit_smoothing
            elif desired_w > current_w:
                sm = entry_smoothing
            else:
                sm = smoothing
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
            r = row.get("global_rank", np.inf)
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

        if not bool(row.get("valid_market", False)):
            exec_delta[idx] = 0
            reason[idx] = "invalid_market"
            continue

        if d > 0 and not local_buy_allowed(row, cfg):
            exec_delta[idx] = 0
            reason[idx] = "blocked_by_local_buy_gate"
            continue

        spread_bps = row.get("spread_bps", np.nan)

        if d > 0 and np.isfinite(spread_bps) and spread_bps > entry_max_spread_bps:
            exec_delta[idx] = 0
            reason[idx] = "entry_wide_spread"
            continue

        if d < 0 and np.isfinite(spread_bps) and spread_bps > exit_max_spread_bps:
            exec_delta[idx] = 0
            reason[idx] = "exit_wide_spread"
            continue

        notional = abs(d) * px
        side_min_notional = entry_min_notional if d > 0 else exit_min_notional

        if notional < side_min_notional:
            exec_delta[idx] = 0
            reason[idx] = "below_entry_min_notional" if d > 0 else "below_exit_min_notional"
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
    df = add_risk_overlay_features(df, cfg)

    log_print("===== input summary =====", log_path)
    log_print(f"rows: {len(df)}", log_path)
    log_print(f"date range: {df['date'].min()} -> {df['date'].max()}", log_path)
    log_print(f"minutes: {df['minute'].nunique()}", log_path)
    log_print(f"symbols: {df['securityid'].nunique()}", log_path)
    log_print(f"global signal non-null rate: {df['has_global_signal'].mean():.6f}", log_path)
    log_print(f"local signal non-null rate: {df['has_local_signal'].mean():.6f}", log_path)

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

        out["optimizer_status"] = "local_global_rank_target"

        keep_cols = unique_list([
            "date",
            "minute",
            c["datetime_col"],
            "securityid",
            "global_alpha_raw",
            "global_alpha_z",
            "global_score",
            "global_rank",
            "local_alpha_raw",
            "local_alpha_z",
            "local_rank",
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
            "has_global_signal",
            "has_local_signal",
            "state",
            "hold_age_min",
            "cooldown_left_min",
            "target_gross_t",
            "base_target_gross_t",
            "risk_multiplier_t",
            "market_ret_1m",
            "up_ratio",
            "market_volatility",
            "market_cum_ret",
            "market_cum_up_ratio",
            "minutes_since_open",
            "daily_loss_multiplier",
            "intraday_market_multiplier",
            "daily_loss_block_new_buy",
            "intraday_market_block_new_buy",
            "alpha_spread",
            "market_regime_multiplier",
            "alpha_quality_multiplier",
            "risk_multiplier",
            "block_new_entry_by_risk",
            "per_name_weight_t",
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
                f"rows={len(out)}, global_signal={int(out['has_global_signal'].sum())}, "
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

    if "risk_multiplier_t" in final.columns:
        log_print("risk multiplier per minute:", log_path)
        log_print(str(final.groupby("minute")["risk_multiplier_t"].median().describe()), log_path)

    if "market_regime_multiplier" in final.columns:
        log_print("market regime multiplier counts:", log_path)
        log_print(str(final.groupby("minute")["market_regime_multiplier"].median().value_counts().sort_index()), log_path)

    if "alpha_quality_multiplier" in final.columns:
        log_print("alpha quality multiplier counts:", log_path)
        log_print(str(final.groupby("minute")["alpha_quality_multiplier"].median().value_counts().sort_index()), log_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = read_yaml(args.config)
    apply_optimizer(cfg)


if __name__ == "__main__":
    main()