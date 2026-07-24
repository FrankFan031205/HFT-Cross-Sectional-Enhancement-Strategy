import argparse
import os
import yaml
import numpy as np
import pandas as pd


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def ensure_dir(path):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def lower_columns(df):
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def find_col_from_columns(columns, candidates, required=True, name="column"):
    cols = {str(c).strip().lower(): c for c in columns}
    for c in candidates:
        k = str(c).strip().lower()
        if k in cols:
            return k
    if required:
        raise ValueError(f"cannot find {name}, candidates={candidates}, existing={list(cols.keys())[:80]}")
    return None


def find_col(df, candidates, required=True, name="column"):
    return find_col_from_columns(df.columns, candidates, required, name)


def normalize_symbol(s):
    x = pd.to_numeric(s, errors="coerce")
    if x.notna().mean() > 0.99:
        return x.astype("Int64")
    return s.astype(str).str.strip()


def standardize_side(x):
    if pd.isna(x):
        return "NONE"
    s = str(x).strip().upper()
    if s in ["BUY", "B", "LONG", "1", "+1"]:
        return "BUY"
    if s in ["SELL", "S", "SHORT", "-1"]:
        return "SELL"
    if s in ["NONE", "NO", "NAN", "0", "FLAT", "HOLD", ""]:
        return "NONE"
    return s


def to_bool_series(s):
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").fillna(0).astype(float) != 0

    x = s.astype(str).str.strip().str.lower()
    return x.isin(["1", "true", "yes", "y", "selected", "valid", "t"])


def auto_price_cols_from_columns(columns):
    bid_col = find_col_from_columns(
        columns,
        ["bid1", "bid_price", "bidprice1", "bid_price1", "bid1_price", "best_bid", "bid"],
        required=False,
        name="bid_col",
    )
    ask_col = find_col_from_columns(
        columns,
        ["ask1", "ask_price", "askprice1", "ask_price1", "ask1_price", "best_ask", "ask"],
        required=False,
        name="ask_col",
    )
    mid_col = find_col_from_columns(
        columns,
        ["mid_price", "midprice", "mid", "wap_mid"],
        required=False,
        name="mid_col",
    )
    return bid_col, ask_col, mid_col


def get_optional_numeric(df, col, default=np.nan):
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(default, index=df.index)


def prepare_optimizer(opt_raw, cfg):
    opt = lower_columns(opt_raw)

    cols = cfg["columns"]
    filt = cfg.get("filters", {})
    ta = cfg.get("time_alignment", {})

    datetime_col = str(cols.get("datetime_col", "datetime")).lower()
    symbol_col = str(cols.get("symbol_col", "securityid")).lower()
    side_col = str(cols.get("side_col", "side")).lower()
    status_col = str(cols.get("optimizer_status_col", "optimizer_status")).lower()
    weight_col = str(cols.get("weight_col", "effective_target_weight")).lower()

    if datetime_col not in opt.columns:
        raise ValueError(f"optimizer missing datetime_col={datetime_col}")
    if symbol_col not in opt.columns:
        raise ValueError(f"optimizer missing symbol_col={symbol_col}")
    if weight_col not in opt.columns:
        raise ValueError(f"optimizer missing weight_col={weight_col}")
    if side_col not in opt.columns:
        raise ValueError(f"optimizer missing side_col={side_col}")

    opt[datetime_col] = pd.to_datetime(opt[datetime_col], errors="coerce")
    opt[symbol_col] = normalize_symbol(opt[symbol_col])
    opt[weight_col] = pd.to_numeric(opt[weight_col], errors="coerce").fillna(0.0)

    raw_side = opt[side_col].map(standardize_side)

    if status_col in opt.columns:
        optimizer_status = opt[status_col].astype(str)
    else:
        optimizer_status = "unknown"

    role = str(ta.get("optimizer_datetime_role", "decision_time")).lower()
    lag_minutes = int(ta.get("execution_lag_minutes", 1))

    if role == "decision_time":
        decision_datetime = opt[datetime_col]
        execution_datetime = decision_datetime + pd.to_timedelta(lag_minutes, unit="m")
    elif role == "execution_time":
        execution_datetime = opt[datetime_col]
        decision_datetime = execution_datetime - pd.to_timedelta(lag_minutes, unit="m")
    else:
        raise ValueError("optimizer_datetime_role must be decision_time or execution_time")

    raw_abs_weight = opt[weight_col].abs().astype(float)

    raw_signed_weight = np.where(
        raw_side.eq("BUY"),
        raw_abs_weight,
        np.where(raw_side.eq("SELL"), -raw_abs_weight, 0.0),
    )

    out = pd.DataFrame(
        {
            "decision_datetime": decision_datetime,
            "execution_datetime": execution_datetime,
            "execution_minute": execution_datetime.dt.floor("min"),
            "securityid": opt[symbol_col],
            "raw_side": raw_side,
            "raw_target_weight": raw_signed_weight.astype(float),
            "raw_abs_target_weight": np.abs(raw_signed_weight.astype(float)),
            "optimizer_status": optimizer_status,
        }
    )

    extra_cols = [
        "pred_ret_cs_dl_cnn_gru_h30",
        "alpha_bps",
        "spread_bps",
        "cost_bps",
        "net_alpha_bps",
        "eligible_buy",
        "mid_price",
        "bid1",
        "ask1",
        "spread",
        "industryid1",
        "size_z",
        "liquidity_z",
        "value_z",
        "momentum_z",
        "volatility_z",
        "alpha_rank",
        "gross_limit_t",
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
        "selected",
        "max_buy_qty_by_ask_volume",
        "max_sell_qty_by_bid_volume",
        "blocked_reason",
        "position_source",
        "abs_delta_notional",
        "valid_market",
    ]

    for c in extra_cols:
        if c in opt.columns and c not in out.columns:
            out[c] = opt[c].values

    out["decision_date"] = out["decision_datetime"].dt.strftime("%Y%m%d")
    out["execution_date"] = out["execution_datetime"].dt.strftime("%Y%m%d")

    require_optimal = bool(filt.get("require_optimal", True))
    require_selected = bool(filt.get("require_selected", True))
    require_valid_market = bool(filt.get("require_valid_market", True))

    min_net_alpha_bps = filt.get("min_net_alpha_bps", None)
    min_abs_delta_notional = float(filt.get("min_abs_delta_notional", 0.0))
    min_abs_target_weight = float(filt.get("min_abs_target_weight", 0.0))
    max_alpha_rank = filt.get("max_alpha_rank", None)
    allowed_blocked_reasons = filt.get("allowed_blocked_reasons", None)

    if require_optimal:
        out["filter_optimal"] = out["optimizer_status"].astype(str).str.lower().eq("optimal")
    else:
        out["filter_optimal"] = True

    if require_selected:
        if "selected" not in out.columns:
            raise ValueError("filters.require_selected=true but optimizer output has no selected column")
        out["filter_selected"] = to_bool_series(out["selected"])
    else:
        out["filter_selected"] = True

    if require_valid_market:
        if "valid_market" not in out.columns:
            raise ValueError("filters.require_valid_market=true but optimizer output has no valid_market column")
        out["filter_valid_market"] = to_bool_series(out["valid_market"])
    else:
        out["filter_valid_market"] = True

    if min_net_alpha_bps is not None:
        if "net_alpha_bps" not in out.columns:
            raise ValueError("filters.min_net_alpha_bps is set but optimizer output has no net_alpha_bps column")
        out["net_alpha_bps"] = pd.to_numeric(out["net_alpha_bps"], errors="coerce")
        out["filter_net_alpha"] = out["net_alpha_bps"] >= float(min_net_alpha_bps)
    else:
        out["filter_net_alpha"] = True

    if min_abs_delta_notional > 0:
        if "abs_delta_notional" in out.columns:
            out["abs_delta_notional"] = pd.to_numeric(out["abs_delta_notional"], errors="coerce").fillna(0.0)
            out["filter_delta_notional"] = out["abs_delta_notional"] >= min_abs_delta_notional
        else:
            out["filter_delta_notional"] = True
    else:
        out["filter_delta_notional"] = True

    if min_abs_target_weight > 0:
        out["filter_target_weight"] = out["raw_abs_target_weight"] >= min_abs_target_weight
    else:
        out["filter_target_weight"] = True

    if max_alpha_rank is not None:
        if "alpha_rank" not in out.columns:
            raise ValueError("filters.max_alpha_rank is set but optimizer output has no alpha_rank column")
        out["alpha_rank"] = pd.to_numeric(out["alpha_rank"], errors="coerce")
        out["filter_alpha_rank"] = out["alpha_rank"] <= float(max_alpha_rank)
    else:
        out["filter_alpha_rank"] = True

    if allowed_blocked_reasons:
        if "blocked_reason" not in out.columns:
            raise ValueError("filters.allowed_blocked_reasons is set but optimizer output has no blocked_reason column")
        allowed = set(str(x).strip().lower() for x in allowed_blocked_reasons)
        out["blocked_reason_norm"] = out["blocked_reason"].astype(str).str.strip().str.lower()
        out["filter_blocked_reason"] = out["blocked_reason_norm"].isin(allowed)
    else:
        out["filter_blocked_reason"] = True

    out["filter_side"] = out["raw_side"].isin(["BUY", "SELL"])
    out["filter_nonzero_target"] = out["raw_abs_target_weight"] > 0

    filter_cols = [
        "filter_optimal",
        "filter_selected",
        "filter_valid_market",
        "filter_net_alpha",
        "filter_delta_notional",
        "filter_target_weight",
        "filter_alpha_rank",
        "filter_blocked_reason",
        "filter_side",
        "filter_nonzero_target",
    ]

    out["passed_filter"] = True
    for c in filter_cols:
        out["passed_filter"] = out["passed_filter"] & out[c].astype(bool)

    out["target_weight"] = np.where(out["passed_filter"], out["raw_target_weight"], 0.0).astype(float)

    out["target_side"] = np.where(
        out["target_weight"] > 0,
        "BUY",
        np.where(out["target_weight"] < 0, "SELL", "NONE"),
    )

    out = out.dropna(subset=["execution_datetime", "execution_minute", "securityid"])
    out = out.sort_values(["securityid", "execution_datetime"]).reset_index(drop=True)

    return out


def load_market_first_snapshot(market_path, opt, cfg):
    cols = cfg["columns"]
    runtime = cfg.get("runtime", {})

    datetime_col = str(cols.get("datetime_col", "datetime")).lower()
    symbol_col = str(cols.get("symbol_col", "securityid")).lower()
    label_col = str(cols.get("label_col", "label_120")).lower()

    chunksize = int(runtime.get("market_chunksize", 2000000))

    header = pd.read_csv(market_path, nrows=0)
    header_lower = [str(c).strip().lower() for c in header.columns]
    lower_to_orig = {str(c).strip().lower(): c for c in header.columns}

    bid_cfg = str(cols.get("bid_col", "auto")).lower()
    ask_cfg = str(cols.get("ask_col", "auto")).lower()
    mid_cfg = str(cols.get("mid_col", "auto")).lower()

    auto_bid, auto_ask, auto_mid = auto_price_cols_from_columns(header.columns)

    bid_col = auto_bid if bid_cfg == "auto" else bid_cfg
    ask_col = auto_ask if ask_cfg == "auto" else ask_cfg
    mid_col = auto_mid if mid_cfg == "auto" else mid_cfg

    need_lower = [datetime_col, symbol_col, label_col, bid_col, ask_col, mid_col]
    for c in need_lower:
        if c not in header_lower:
            raise ValueError(f"market file missing required column={c}")

    usecols_orig = [lower_to_orig[c] for c in dict.fromkeys(need_lower)]

    needed_symbols = set(opt["securityid"].dropna().astype(int).unique())
    needed_minutes = set(pd.to_datetime(opt["execution_minute"].dropna().unique()))

    parts = []

    for i, chunk in enumerate(pd.read_csv(market_path, usecols=usecols_orig, chunksize=chunksize, low_memory=False), start=1):
        chunk = lower_columns(chunk)

        chunk[datetime_col] = pd.to_datetime(chunk[datetime_col], errors="coerce")
        chunk[symbol_col] = normalize_symbol(chunk[symbol_col])

        chunk = chunk[chunk[symbol_col].notna()].copy()
        chunk["_symbol_int"] = chunk[symbol_col].astype(int)

        chunk = chunk[chunk["_symbol_int"].isin(needed_symbols)].copy()

        if chunk.empty:
            continue

        chunk["execution_minute"] = chunk[datetime_col].dt.floor("min")
        chunk = chunk[chunk["execution_minute"].isin(needed_minutes)].copy()

        if chunk.empty:
            continue

        chunk = chunk.rename(
            columns={
                datetime_col: "market_datetime",
                symbol_col: "securityid",
                bid_col: "bid_price",
                ask_col: "ask_price",
                mid_col: "mid_price",
                label_col: "label",
            }
        )

        keep = [
            "securityid",
            "execution_minute",
            "market_datetime",
            "bid_price",
            "ask_price",
            "mid_price",
            "label",
        ]

        chunk = chunk[keep].copy()

        for c in ["bid_price", "ask_price", "mid_price", "label"]:
            chunk[c] = pd.to_numeric(chunk[c], errors="coerce")

        chunk = chunk.sort_values(["securityid", "execution_minute", "market_datetime"])
        chunk = chunk.drop_duplicates(["securityid", "execution_minute"], keep="first")

        parts.append(chunk)

        if i % 10 == 0:
            print(f"  loaded market chunks: {i}, matched parts: {len(parts)}")

    if not parts:
        return pd.DataFrame(
            columns=[
                "securityid",
                "execution_minute",
                "market_datetime",
                "bid_price",
                "ask_price",
                "mid_price",
                "label",
            ]
        )

    mkt = pd.concat(parts, ignore_index=True)
    mkt = mkt.sort_values(["securityid", "execution_minute", "market_datetime"])
    mkt = mkt.drop_duplicates(["securityid", "execution_minute"], keep="first")
    mkt["securityid"] = normalize_symbol(mkt["securityid"])

    return mkt.reset_index(drop=True)


def merge_optimizer_market(opt, mkt):
    return opt.merge(
        mkt,
        on=["securityid", "execution_minute"],
        how="left",
        suffixes=("", "_mkt"),
    )


def compute_positions_and_pnl(df, cfg):
    exe = cfg.get("execution", {})

    capital = float(exe.get("capital", 200000000))
    taker_fee_bps = float(exe.get("taker_fee_bps", 1.5))
    slippage_bps = float(exe.get("slippage_bps", 0.0))
    allow_sell = bool(exe.get("allow_sell", True))

    df = df.copy()

    if not allow_sell:
        df.loc[df["target_weight"] < 0, "target_weight"] = 0.0
        df["target_side"] = np.where(df["target_weight"] > 0, "BUY", "NONE")

    df = df.sort_values(["securityid", "execution_datetime"]).reset_index(drop=True)

    df["prev_target_weight"] = df.groupby("securityid")["target_weight"].shift(1).fillna(0.0)
    df["trade_weight"] = df["target_weight"] - df["prev_target_weight"]

    df["trade_side"] = np.where(
        df["trade_weight"] > 0,
        "BUY",
        np.where(df["trade_weight"] < 0, "SELL", "NONE"),
    )

    df["abs_trade_weight"] = df["trade_weight"].abs()
    df["abs_target_weight"] = df["target_weight"].abs()

    for c in ["mid_price", "bid_price", "ask_price", "label"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["effective_bid"] = df["bid_price"].where(
        df["bid_price"].notna() & (df["bid_price"] > 0),
        df["mid_price"],
    )

    df["effective_ask"] = df["ask_price"].where(
        df["ask_price"].notna() & (df["ask_price"] > 0),
        df["mid_price"],
    )

    df["exec_price"] = np.where(
        df["trade_side"].eq("BUY"),
        df["effective_ask"],
        np.where(df["trade_side"].eq("SELL"), df["effective_bid"], np.nan),
    )

    buy_spread_cost = df["effective_ask"] / df["mid_price"] - 1.0
    sell_spread_cost = (df["mid_price"] - df["effective_bid"]) / df["mid_price"]

    df["spread_cost_ret"] = np.where(
        df["trade_side"].eq("BUY"),
        buy_spread_cost,
        np.where(df["trade_side"].eq("SELL"), sell_spread_cost, 0.0),
    )

    df["spread_cost_ret"] = (
        df["spread_cost_ret"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .clip(lower=0.0)
    )

    df["gross_exposure"] = capital * df["abs_target_weight"]
    df["net_exposure"] = capital * df["target_weight"]
    df["trade_notional"] = capital * df["abs_trade_weight"]

    df["gross_pnl"] = capital * df["target_weight"] * df["label"]

    df["spread_cost"] = df["trade_notional"] * df["spread_cost_ret"]
    df["fee"] = df["trade_notional"] * taker_fee_bps / 10000.0
    df["slippage"] = df["trade_notional"] * slippage_bps / 10000.0
    df["cost"] = df["spread_cost"] + df["fee"] + df["slippage"]

    df["net_pnl"] = df["gross_pnl"] - df["cost"]

    df["gross_pnl_bps_on_trade_turnover"] = df["gross_pnl"] / df["trade_notional"].replace(0, np.nan) * 10000.0
    df["net_pnl_bps_on_trade_turnover"] = df["net_pnl"] / df["trade_notional"].replace(0, np.nan) * 10000.0
    df["pnl_bps_on_trade_turnover"] = df["net_pnl_bps_on_trade_turnover"]

    df["gross_pnl_bps_on_gross_exposure"] = df["gross_pnl"] / df["gross_exposure"].replace(0, np.nan) * 10000.0
    df["net_pnl_bps_on_gross_exposure"] = df["net_pnl"] / df["gross_exposure"].replace(0, np.nan) * 10000.0

    keep = (
        (df["abs_target_weight"] > 0)
        | (df["abs_trade_weight"] > 0)
        | (df["gross_pnl"].abs() > 0)
        | (df["net_pnl"].abs() > 0)
    )

    return df[keep].copy()


def make_minute_metrics(x):
    if x.empty:
        return pd.DataFrame()

    g = x.groupby(["execution_date", "execution_minute"], dropna=False)

    out = g.agg(
        num_position_rows=("securityid", "count"),
        num_trade_events=("trade_side", lambda s: (s != "NONE").sum()),
        num_buy_trades=("trade_side", lambda s: (s == "BUY").sum()),
        num_sell_trades=("trade_side", lambda s: (s == "SELL").sum()),
        num_passed_filter=("passed_filter", "sum"),
        gross_weight=("abs_target_weight", "sum"),
        net_weight=("target_weight", "sum"),
        trade_weight=("abs_trade_weight", "sum"),
        gross_exposure=("gross_exposure", "sum"),
        net_exposure=("net_exposure", "sum"),
        turnover=("trade_notional", "sum"),
        gross_pnl=("gross_pnl", "sum"),
        spread_cost=("spread_cost", "sum"),
        fee=("fee", "sum"),
        slippage=("slippage", "sum"),
        cost=("cost", "sum"),
        net_pnl=("net_pnl", "sum"),
    ).reset_index()

    out["gross_pnl_bps_on_turnover"] = out["gross_pnl"] / out["turnover"].replace(0, np.nan) * 10000.0
    out["cost_bps_on_turnover"] = out["cost"] / out["turnover"].replace(0, np.nan) * 10000.0
    out["net_pnl_bps_on_turnover"] = out["net_pnl"] / out["turnover"].replace(0, np.nan) * 10000.0

    out = out.sort_values(["execution_date", "execution_minute"]).reset_index(drop=True)

    out["cum_net_pnl"] = out["net_pnl"].cumsum()
    out["cum_turnover"] = out["turnover"].cumsum()
    out["drawdown"] = out["cum_net_pnl"] - out["cum_net_pnl"].cummax()

    return out


def make_daily_metrics(x):
    if x.empty:
        return pd.DataFrame()

    g = x.groupby("execution_date", dropna=False)

    out = g.agg(
        num_position_rows=("securityid", "count"),
        num_trade_events=("trade_side", lambda s: (s != "NONE").sum()),
        num_buy_trades=("trade_side", lambda s: (s == "BUY").sum()),
        num_sell_trades=("trade_side", lambda s: (s == "SELL").sum()),
        num_symbols=("securityid", "nunique"),
        num_passed_filter=("passed_filter", "sum"),
        gross_weight_sum=("abs_target_weight", "sum"),
        trade_weight_sum=("abs_trade_weight", "sum"),
        turnover=("trade_notional", "sum"),
        gross_pnl=("gross_pnl", "sum"),
        spread_cost=("spread_cost", "sum"),
        fee=("fee", "sum"),
        slippage=("slippage", "sum"),
        cost=("cost", "sum"),
        net_pnl=("net_pnl", "sum"),
        win_rate_rows=("net_pnl", lambda s: (s > 0).mean()),
    ).reset_index()

    out["gross_pnl_bps_on_turnover"] = out["gross_pnl"] / out["turnover"].replace(0, np.nan) * 10000.0
    out["cost_bps_on_turnover"] = out["cost"] / out["turnover"].replace(0, np.nan) * 10000.0
    out["net_pnl_bps_on_turnover"] = out["net_pnl"] / out["turnover"].replace(0, np.nan) * 10000.0

    out = out.sort_values("execution_date").reset_index(drop=True)

    out["cum_net_pnl"] = out["net_pnl"].cumsum()
    out["cum_turnover"] = out["turnover"].cumsum()

    return out


def make_bucket_metrics(x):
    if x.empty or "net_alpha_bps" not in x.columns:
        return pd.DataFrame()

    df = x[x["trade_notional"] > 0].copy()

    if df.empty:
        return pd.DataFrame()

    df["net_alpha_bps"] = pd.to_numeric(df["net_alpha_bps"], errors="coerce")
    df = df[df["net_alpha_bps"].notna()].copy()

    if df.empty:
        return pd.DataFrame()

    n_unique = df["net_alpha_bps"].nunique()

    if n_unique < 2:
        df["net_alpha_bucket"] = "all"
    else:
        q = min(5, n_unique)
        df["net_alpha_bucket"] = pd.qcut(
            df["net_alpha_bps"].rank(method="first"),
            q,
            labels=[f"Q{i+1}" for i in range(q)],
        )

    g = df.groupby("net_alpha_bucket", dropna=False).agg(
        num_trades=("securityid", "count"),
        turnover=("trade_notional", "sum"),
        gross_exposure=("gross_exposure", "sum"),
        gross_pnl=("gross_pnl", "sum"),
        spread_cost=("spread_cost", "sum"),
        fee=("fee", "sum"),
        cost=("cost", "sum"),
        net_pnl=("net_pnl", "sum"),
        avg_net_alpha_bps=("net_alpha_bps", "mean"),
        min_net_alpha_bps=("net_alpha_bps", "min"),
        max_net_alpha_bps=("net_alpha_bps", "max"),
    ).reset_index()

    g["gross_pnl_bps_on_turnover"] = g["gross_pnl"] / g["turnover"].replace(0, np.nan) * 10000.0
    g["cost_bps_on_turnover"] = g["cost"] / g["turnover"].replace(0, np.nan) * 10000.0
    g["net_pnl_bps_on_turnover"] = g["net_pnl"] / g["turnover"].replace(0, np.nan) * 10000.0

    return g


def make_filter_summary(opt):
    rows = []

    filter_cols = [
        "filter_optimal",
        "filter_selected",
        "filter_valid_market",
        "filter_net_alpha",
        "filter_delta_notional",
        "filter_target_weight",
        "filter_alpha_rank",
        "filter_blocked_reason",
        "filter_side",
        "filter_nonzero_target",
        "passed_filter",
    ]

    n = len(opt)

    for c in filter_cols:
        if c in opt.columns:
            ok = int(opt[c].astype(bool).sum())
            rows.append(
                {
                    "filter": c,
                    "pass_rows": ok,
                    "total_rows": n,
                    "pass_rate": ok / n if n > 0 else np.nan,
                }
            )

    return pd.DataFrame(rows)


def make_summary(x, minute_metrics, opt, cfg):
    capital = float(cfg.get("execution", {}).get("capital", 200000000))

    if x.empty:
        return pd.DataFrame(
            [
                {
                    "capital": capital,
                    "num_position_rows": 0,
                    "num_trade_events": 0,
                    "total_net_pnl": 0.0,
                }
            ]
        )

    total_turnover = x["trade_notional"].sum()
    total_gross_exposure = x["gross_exposure"].sum()
    total_gross_pnl = x["gross_pnl"].sum()
    total_spread_cost = x["spread_cost"].sum()
    total_fee = x["fee"].sum()
    total_slippage = x["slippage"].sum()
    total_cost = x["cost"].sum()
    total_net_pnl = x["net_pnl"].sum()

    max_drawdown = np.nan
    if minute_metrics is not None and not minute_metrics.empty:
        max_drawdown = minute_metrics["drawdown"].min()

    summary = {
        "start_date": x["execution_date"].min(),
        "end_date": x["execution_date"].max(),
        "capital": capital,
        "optimizer_rows": len(opt),
        "optimizer_passed_filter_rows": int(opt["passed_filter"].sum()) if "passed_filter" in opt.columns else np.nan,
        "optimizer_pass_rate": float(opt["passed_filter"].mean()) if "passed_filter" in opt.columns else np.nan,
        "num_position_rows": len(x),
        "num_trade_events": int((x["trade_side"] != "NONE").sum()),
        "num_buy_trades": int((x["trade_side"] == "BUY").sum()),
        "num_sell_trades": int((x["trade_side"] == "SELL").sum()),
        "num_symbols": x["securityid"].nunique(),
        "total_turnover": total_turnover,
        "total_gross_exposure": total_gross_exposure,
        "total_gross_pnl": total_gross_pnl,
        "total_spread_cost": total_spread_cost,
        "total_fee": total_fee,
        "total_slippage": total_slippage,
        "total_cost": total_cost,
        "total_net_pnl": total_net_pnl,
        "gross_pnl_bps_on_turnover": total_gross_pnl / total_turnover * 10000.0 if total_turnover > 0 else np.nan,
        "spread_cost_bps_on_turnover": total_spread_cost / total_turnover * 10000.0 if total_turnover > 0 else np.nan,
        "fee_bps_on_turnover": total_fee / total_turnover * 10000.0 if total_turnover > 0 else np.nan,
        "cost_bps_on_turnover": total_cost / total_turnover * 10000.0 if total_turnover > 0 else np.nan,
        "net_pnl_bps_on_turnover": total_net_pnl / total_turnover * 10000.0 if total_turnover > 0 else np.nan,
        "net_pnl_bps_on_gross_exposure": total_net_pnl / total_gross_exposure * 10000.0 if total_gross_exposure > 0 else np.nan,
        "return_on_capital": total_net_pnl / capital if capital > 0 else np.nan,
        "turnover_to_capital": total_turnover / capital if capital > 0 else np.nan,
        "avg_trade_notional": x.loc[x["trade_side"] != "NONE", "trade_notional"].mean(),
        "max_drawdown": max_drawdown,
    }

    return pd.DataFrame([summary])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    market_path = cfg["data"]["market_data_path"]
    optimizer_path = cfg["data"]["optimizer_output_path"]

    print("loading optimizer:", optimizer_path)
    opt_raw = pd.read_csv(optimizer_path, low_memory=False)

    print("preparing optimizer and applying v2 filters")
    opt = prepare_optimizer(opt_raw, cfg)

    print("optimizer rows:", len(opt))
    print("optimizer date range:", opt["execution_date"].min(), opt["execution_date"].max())
    print("optimizer symbols:", opt["securityid"].nunique())
    print("passed filter rows:", int(opt["passed_filter"].sum()))
    print("passed filter rate:", float(opt["passed_filter"].mean()))

    print("loading market first snapshots:", market_path)
    mkt = load_market_first_snapshot(market_path, opt, cfg)

    print("market matched first-snapshot rows:", len(mkt))
    print("market symbols:", mkt["securityid"].nunique() if not mkt.empty else 0)

    print("merging optimizer with market")
    merged = merge_optimizer_market(opt, mkt)

    missing_mid_rate = merged["mid_price"].isna().mean()
    missing_label_rate = merged["label"].isna().mean()

    print("merged shape:", merged.shape)
    print("missing mid rate:", missing_mid_rate)
    print("missing label rate:", missing_label_rate)

    debug_path = "TakerModel/outputs/v2/metrics/taker_model_v2_merge_debug_sample.csv"
    ensure_dir(debug_path)
    merged.head(500).to_csv(debug_path, index=False)
    print("saved merge debug sample:", debug_path)

    max_missing_rate = float(cfg.get("runtime", {}).get("max_missing_rate", 0.2))

    if missing_mid_rate > max_missing_rate or missing_label_rate > max_missing_rate:
        raise ValueError(
            "merge failed or label missing: "
            f"missing_mid_rate={missing_mid_rate}, "
            f"missing_label_rate={missing_label_rate}. "
            "Check market_data_path, universe, datetime alignment, and label_col."
        )

    print("computing target-position taker pnl v2")
    result = compute_positions_and_pnl(merged, cfg)

    minute_metrics = make_minute_metrics(result)
    daily_metrics = make_daily_metrics(result)
    bucket_metrics = make_bucket_metrics(result)
    filter_summary = make_filter_summary(opt)
    summary = make_summary(result, minute_metrics, opt, cfg)

    paths = cfg["output"]

    for p in paths.values():
        ensure_dir(p)

    result.to_csv(paths["position_output_path"], index=False)
    minute_metrics.to_csv(paths["minute_metrics_path"], index=False)
    daily_metrics.to_csv(paths["daily_metrics_path"], index=False)
    bucket_metrics.to_csv(paths["bucket_metrics_path"], index=False)
    filter_summary.to_csv(paths["filter_summary_path"], index=False)
    summary.to_csv(paths["summary_path"], index=False)

    print()
    print("saved position output:", paths["position_output_path"])
    print("saved minute metrics:", paths["minute_metrics_path"])
    print("saved daily metrics:", paths["daily_metrics_path"])
    print("saved bucket metrics:", paths["bucket_metrics_path"])
    print("saved filter summary:", paths["filter_summary_path"])
    print("saved summary:", paths["summary_path"])
    print()
    print("===== summary =====")
    print(summary.T)

    if not bucket_metrics.empty:
        print()
        print("===== bucket metrics =====")
        print(bucket_metrics.to_string(index=False))

    if not filter_summary.empty:
        print()
        print("===== filter summary =====")
        print(filter_summary.to_string(index=False))


if __name__ == "__main__":
    main()
