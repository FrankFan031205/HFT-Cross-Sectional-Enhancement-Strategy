import numpy as np
import pandas as pd


def safe_numeric(df, col, default=np.nan):
    if col not in df.columns:
        return pd.Series(default, index=df.index)
    return pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)


def add_trade_date(df, datetime_col):
    df[datetime_col] = pd.to_datetime(df[datetime_col], errors="coerce")
    df["trade_date"] = df[datetime_col].dt.strftime("%Y-%m-%d")
    return df


def fast_group_zscore(df, group_col, raw_col, out_col):
    x = pd.to_numeric(df[raw_col], errors="coerce").replace([np.inf, -np.inf], np.nan)

    # Fill within timestamp first, then global, then 0
    group_median = x.groupby(df[group_col]).transform("median")
    global_median = x.median()
    if pd.isna(global_median):
        global_median = 0.0

    x = x.fillna(group_median).fillna(global_median).fillna(0.0)

    mean = x.groupby(df[group_col]).transform("mean")
    std = x.groupby(df[group_col]).transform("std").replace(0, np.nan)

    z = (x - mean) / std
    df[out_col] = z.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return df


def add_style_exposures(df, cfg):
    c = cfg["columns"]
    dt_col = c["datetime_col"]

    mv_col = c.get("market_value_col", "marketValue")
    turnover_col = c.get("turnover_col", "turnoverRate")
    vol_col = c.get("volatility_col", "volatility_60")
    spread_col = c.get("spread_col", "spread")

    mv = safe_numeric(df, mv_col)

    if mv.isna().all() and "negMarketValue" in df.columns:
        mv = safe_numeric(df, "negMarketValue")

    turnover = safe_numeric(df, turnover_col)
    spread = safe_numeric(df, spread_col)
    vol = safe_numeric(df, vol_col)

    if mv.isna().all():
        df["size_raw"] = 0.0
    else:
        df["size_raw"] = np.log(mv.clip(lower=1.0))

    if turnover.isna().all():
        if spread.isna().all():
            df["liquidity_raw"] = 0.0
        else:
            df["liquidity_raw"] = -spread
    else:
        df["liquidity_raw"] = turnover

    if vol.isna().all():
        if "ATR14" in df.columns and not safe_numeric(df, "ATR14").isna().all():
            df["volatility_raw"] = safe_numeric(df, "ATR14")
        elif "ATR6" in df.columns and not safe_numeric(df, "ATR6").isna().all():
            df["volatility_raw"] = safe_numeric(df, "ATR6")
        else:
            df["volatility_raw"] = 0.0
    else:
        df["volatility_raw"] = vol

    df = fast_group_zscore(df, dt_col, "size_raw", "size_z")
    df = fast_group_zscore(df, dt_col, "liquidity_raw", "liquidity_z")
    df = fast_group_zscore(df, dt_col, "volatility_raw", "volatility_z")

    return df
