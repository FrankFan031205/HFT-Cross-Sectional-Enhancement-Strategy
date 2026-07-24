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


def normalize_symbol(s):
    x = pd.to_numeric(s, errors="coerce")
    if x.notna().mean() > 0.99:
        return x.astype("Int64")
    return s.astype(str).str.strip()


def to_bool(s):
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").fillna(0).astype(float) != 0
    return s.astype(str).str.strip().str.lower().isin(["1", "true", "yes", "y", "valid", "selected"])


def prepare_optimizer(path, cfg):
    cols = cfg["columns"]
    ta = cfg["time_alignment"]

    datetime_col = cols["datetime_col"].lower()
    symbol_col = cols["symbol_col"].lower()
    target_weight_col = cols["target_weight_col"].lower()

    opt = pd.read_csv(path, low_memory=False)
    opt = lower_columns(opt)

    for c in [datetime_col, symbol_col, target_weight_col]:
        if c not in opt.columns:
            raise ValueError(f"optimizer missing column: {c}")

    opt[datetime_col] = pd.to_datetime(opt[datetime_col], errors="coerce")
    opt[symbol_col] = normalize_symbol(opt[symbol_col])
    opt[target_weight_col] = pd.to_numeric(opt[target_weight_col], errors="coerce").fillna(0.0)

    role = str(ta.get("optimizer_datetime_role", "decision_time")).lower()
    lag = int(ta.get("execution_lag_minutes", 1))

    if role == "decision_time":
        decision_datetime = opt[datetime_col]
        execution_datetime = decision_datetime + pd.to_timedelta(lag, unit="m")
    elif role == "execution_time":
        execution_datetime = opt[datetime_col]
        decision_datetime = execution_datetime - pd.to_timedelta(lag, unit="m")
    else:
        raise ValueError("optimizer_datetime_role must be decision_time or execution_time")

    out = pd.DataFrame({
        "decision_datetime": decision_datetime,
        "execution_datetime": execution_datetime,
        "execution_minute": execution_datetime.dt.floor("min"),
        "execution_date": execution_datetime.dt.strftime("%Y%m%d"),
        "securityid": opt[symbol_col],
        "desired_target_weight": opt[target_weight_col].astype(float),
    })

    extra_cols = [
        "side",
        "optimizer_status",
        "net_alpha_bps",
        "alpha_bps",
        "spread_bps",
        "cost_bps",
        "selected",
        "valid_market",
        "blocked_reason",
        "abs_delta_notional",
        "raw_target_weight",
        "target_weight",
        "effective_target_weight",
        "target_notional",
        "current_qty",
        "sellable_qty",
        "target_qty",
        "delta_qty_raw",
        "delta_qty_executable",
        "effective_target_qty",
        "position_source",
        "alpha_rank",
    ]

    for c in extra_cols:
        if c in opt.columns:
            out[c] = opt[c].values

    if "side" in out.columns:
        out["optimizer_side"] = out["side"].astype(str).str.upper()
    else:
        out["optimizer_side"] = "UNKNOWN"

    if "optimizer_status" in out.columns:
        out["optimizer_status"] = out["optimizer_status"].astype(str)
    else:
        out["optimizer_status"] = "unknown"

    if "net_alpha_bps" in out.columns:
        out["net_alpha_bps"] = pd.to_numeric(out["net_alpha_bps"], errors="coerce")
    else:
        out["net_alpha_bps"] = np.nan

    if "valid_market" in out.columns:
        out["valid_market_bool"] = to_bool(out["valid_market"])
    else:
        out["valid_market_bool"] = True

    out = out.dropna(subset=["execution_datetime", "execution_minute", "securityid"])
    out = out.sort_values(["securityid", "execution_datetime"]).reset_index(drop=True)

    return out


def load_market_first_snapshot(path, opt, cfg):
    cols = cfg["columns"]
    runtime = cfg.get("runtime", {})

    datetime_col = cols["datetime_col"].lower()
    symbol_col = cols["symbol_col"].lower()
    bid_col = cols["bid_col"].lower()
    ask_col = cols["ask_col"].lower()
    mid_col = cols["mid_col"].lower()
    label_col = cols["label_col"].lower()

    chunksize = int(runtime.get("market_chunksize", 2000000))

    header = pd.read_csv(path, nrows=0)
    header_lower = [str(c).strip().lower() for c in header.columns]
    lower_to_orig = {str(c).strip().lower(): c for c in header.columns}

    need = [datetime_col, symbol_col, bid_col, ask_col, mid_col, label_col]
    for c in need:
        if c not in header_lower:
            raise ValueError(f"market missing column: {c}")

    usecols = [lower_to_orig[c] for c in need]

    needed_symbols = set(opt["securityid"].dropna().astype(int).unique())
    needed_minutes = set(pd.to_datetime(opt["execution_minute"].dropna().unique()))

    parts = []

    for i, chunk in enumerate(pd.read_csv(path, usecols=usecols, chunksize=chunksize, low_memory=False), start=1):
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

        chunk = chunk.rename(columns={
            datetime_col: "market_datetime",
            symbol_col: "securityid",
            bid_col: "bid_price",
            ask_col: "ask_price",
            mid_col: "mid_price",
            label_col: "label",
        })

        for c in ["bid_price", "ask_price", "mid_price", "label"]:
            chunk[c] = pd.to_numeric(chunk[c], errors="coerce")

        chunk = chunk[[
            "securityid",
            "execution_minute",
            "market_datetime",
            "bid_price",
            "ask_price",
            "mid_price",
            "label",
        ]].copy()

        chunk = chunk.sort_values(["securityid", "execution_minute", "market_datetime"])
        chunk = chunk.drop_duplicates(["securityid", "execution_minute"], keep="first")

        parts.append(chunk)

        if i % 10 == 0:
            print(f"  loaded market chunks: {i}, matched parts: {len(parts)}")

    if not parts:
        return pd.DataFrame(columns=[
            "securityid", "execution_minute", "market_datetime",
            "bid_price", "ask_price", "mid_price", "label"
        ])

    mkt = pd.concat(parts, ignore_index=True)
    mkt = mkt.sort_values(["securityid", "execution_minute", "market_datetime"])
    mkt = mkt.drop_duplicates(["securityid", "execution_minute"], keep="first")
    mkt["securityid"] = normalize_symbol(mkt["securityid"])

    return mkt.reset_index(drop=True)


def run_turnover_control(df, cfg):
    exe = cfg["execution"]
    filt = cfg["filters"]

    capital = float(exe.get("capital", 200000000))
    taker_fee_bps = float(exe.get("taker_fee_bps", 0.5))
    slippage_bps = float(exe.get("slippage_bps", 0.0))
    rebalance_ratio = float(exe.get("rebalance_ratio", 0.5))

    require_optimal = bool(filt.get("require_optimal", True))
    require_valid_market = bool(filt.get("require_valid_market", True))

    min_abs_delta_notional = float(filt.get("min_abs_delta_notional", 0.0))
    max_spread_bps = filt.get("max_spread_bps", None)
    min_abs_net_alpha_bps = filt.get("min_abs_net_alpha_bps", None)

    df = df.copy()
    df = df.sort_values(["securityid", "execution_datetime"]).reset_index(drop=True)

    for c in ["bid_price", "ask_price", "mid_price", "label", "desired_target_weight"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["spread_bps_realized"] = (df["ask_price"] - df["bid_price"]) / df["mid_price"] * 10000.0

    rows = []
    current_weight = {}

    for row in df.itertuples(index=False):
        sid = int(row.securityid)
        prev_w = current_weight.get(sid, 0.0)

        desired_w = float(row.desired_target_weight) if pd.notna(row.desired_target_weight) else prev_w
        raw_delta_w = desired_w - prev_w
        raw_delta_notional = abs(raw_delta_w) * capital

        reasons = []

        if require_optimal:
            status = str(getattr(row, "optimizer_status", "unknown")).lower()
            if status != "optimal":
                reasons.append("not_optimal")

        if require_valid_market:
            valid_market_bool = bool(getattr(row, "valid_market_bool", True))
            if not valid_market_bool:
                reasons.append("invalid_market")

        if raw_delta_notional < min_abs_delta_notional:
            reasons.append("small_delta")

        spread_bps = getattr(row, "spread_bps_realized", np.nan)
        if max_spread_bps is not None:
            if pd.isna(spread_bps) or spread_bps > float(max_spread_bps):
                reasons.append("wide_spread")

        net_alpha = getattr(row, "net_alpha_bps", np.nan)
        if min_abs_net_alpha_bps is not None:
            if pd.isna(net_alpha) or abs(float(net_alpha)) < float(min_abs_net_alpha_bps):
                reasons.append("weak_edge")

        should_trade = len(reasons) == 0

        if should_trade:
            executed_delta_w = rebalance_ratio * raw_delta_w
            new_w = prev_w + executed_delta_w
            decision = "trade"
        else:
            executed_delta_w = 0.0
            new_w = prev_w
            decision = "hold"

        trade_side = "NONE"
        if executed_delta_w > 0:
            trade_side = "BUY"
        elif executed_delta_w < 0:
            trade_side = "SELL"

        trade_notional = abs(executed_delta_w) * capital
        gross_exposure = abs(new_w) * capital
        net_exposure = new_w * capital

        effective_bid = row.bid_price if pd.notna(row.bid_price) and row.bid_price > 0 else row.mid_price
        effective_ask = row.ask_price if pd.notna(row.ask_price) and row.ask_price > 0 else row.mid_price

        if trade_side == "BUY":
            exec_price = effective_ask
            spread_cost_ret = max(effective_ask / row.mid_price - 1.0, 0.0) if row.mid_price > 0 else 0.0
        elif trade_side == "SELL":
            exec_price = effective_bid
            spread_cost_ret = max((row.mid_price - effective_bid) / row.mid_price, 0.0) if row.mid_price > 0 else 0.0
        else:
            exec_price = np.nan
            spread_cost_ret = 0.0

        gross_pnl = capital * new_w * row.label if pd.notna(row.label) else np.nan
        spread_cost = trade_notional * spread_cost_ret
        fee = trade_notional * taker_fee_bps / 10000.0
        slippage = trade_notional * slippage_bps / 10000.0
        cost = spread_cost + fee + slippage
        net_pnl = gross_pnl - cost if pd.notna(gross_pnl) else np.nan

        current_weight[sid] = new_w

        base = row._asdict()

        base.update({
            "prev_executed_weight": prev_w,
            "desired_target_weight": desired_w,
            "raw_delta_weight": raw_delta_w,
            "raw_delta_notional": raw_delta_notional,
            "executed_delta_weight": executed_delta_w,
            "executed_weight": new_w,
            "decision": decision,
            "skip_reason": "none" if should_trade else "|".join(reasons),
            "trade_side": trade_side,
            "trade_notional": trade_notional,
            "gross_exposure": gross_exposure,
            "net_exposure": net_exposure,
            "exec_price": exec_price,
            "spread_cost_ret": spread_cost_ret,
            "gross_pnl": gross_pnl,
            "spread_cost": spread_cost,
            "fee": fee,
            "slippage": slippage,
            "cost": cost,
            "net_pnl": net_pnl,
        })

        rows.append(base)

    out = pd.DataFrame(rows)

    out["gross_pnl_bps_on_turnover"] = out["gross_pnl"] / out["trade_notional"].replace(0, np.nan) * 10000.0
    out["net_pnl_bps_on_turnover"] = out["net_pnl"] / out["trade_notional"].replace(0, np.nan) * 10000.0
    out["net_pnl_bps_on_gross_exposure"] = out["net_pnl"] / out["gross_exposure"].replace(0, np.nan) * 10000.0

    keep = (
        (out["executed_weight"].abs() > 0)
        | (out["trade_notional"] > 0)
        | (out["net_pnl"].abs() > 0)
    )

    return out[keep].copy()


def make_minute_metrics(x):
    if x.empty:
        return pd.DataFrame()

    g = x.groupby(["execution_date", "execution_minute"], dropna=False)

    out = g.agg(
        num_rows=("securityid", "count"),
        num_trade_events=("trade_side", lambda s: (s != "NONE").sum()),
        num_buy_trades=("trade_side", lambda s: (s == "BUY").sum()),
        num_sell_trades=("trade_side", lambda s: (s == "SELL").sum()),
        turnover=("trade_notional", "sum"),
        gross_exposure=("gross_exposure", "sum"),
        net_exposure=("net_exposure", "sum"),
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
        num_rows=("securityid", "count"),
        num_trade_events=("trade_side", lambda s: (s != "NONE").sum()),
        num_buy_trades=("trade_side", lambda s: (s == "BUY").sum()),
        num_sell_trades=("trade_side", lambda s: (s == "SELL").sum()),
        num_symbols=("securityid", "nunique"),
        turnover=("trade_notional", "sum"),
        gross_exposure=("gross_exposure", "sum"),
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

    out = out.sort_values("execution_date").reset_index(drop=True)
    out["cum_net_pnl"] = out["net_pnl"].cumsum()
    out["cum_turnover"] = out["turnover"].cumsum()

    return out


def make_reason_metrics(x):
    if x.empty:
        return pd.DataFrame()

    g = x.groupby("skip_reason", dropna=False).agg(
        rows=("securityid", "count"),
        trade_events=("trade_side", lambda s: (s != "NONE").sum()),
        turnover=("trade_notional", "sum"),
        gross_pnl=("gross_pnl", "sum"),
        cost=("cost", "sum"),
        net_pnl=("net_pnl", "sum"),
    ).reset_index()

    g["net_pnl_bps_on_turnover"] = g["net_pnl"] / g["turnover"].replace(0, np.nan) * 10000.0
    return g.sort_values("rows", ascending=False)


def make_summary(x, minute_metrics, cfg):
    capital = float(cfg["execution"].get("capital", 200000000))

    if x.empty:
        return pd.DataFrame([{
            "capital": capital,
            "num_position_rows": 0,
            "num_trade_events": 0,
            "total_net_pnl": 0.0,
        }])

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

    return pd.DataFrame([{
        "start_date": x["execution_date"].min(),
        "end_date": x["execution_date"].max(),
        "capital": capital,
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
    }])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    optimizer_path = cfg["data"]["optimizer_output_path"]
    market_path = cfg["data"]["market_data_path"]

    print("loading optimizer:", optimizer_path)
    opt = prepare_optimizer(optimizer_path, cfg)

    print("optimizer rows:", len(opt))
    print("optimizer symbols:", opt["securityid"].nunique())
    print("execution minute range:", opt["execution_minute"].min(), opt["execution_minute"].max())

    print("loading market first snapshots:", market_path)
    mkt = load_market_first_snapshot(market_path, opt, cfg)

    print("market rows:", len(mkt))
    print("market symbols:", mkt["securityid"].nunique() if not mkt.empty else 0)

    print("merging")
    merged = opt.merge(mkt, on=["securityid", "execution_minute"], how="left")

    missing_mid = merged["mid_price"].isna().mean()
    missing_label = merged["label"].isna().mean()

    print("merged shape:", merged.shape)
    print("missing mid rate:", missing_mid)
    print("missing label rate:", missing_label)

    max_missing = float(cfg.get("runtime", {}).get("max_missing_rate", 0.2))
    if missing_mid > max_missing or missing_label > max_missing:
        raise ValueError(
            f"merge failed: missing_mid={missing_mid}, missing_label={missing_label}"
        )

    print("running turnover-control taker model")
    result = run_turnover_control(merged, cfg)

    minute_metrics = make_minute_metrics(result)
    daily_metrics = make_daily_metrics(result)
    reason_metrics = make_reason_metrics(result)
    summary = make_summary(result, minute_metrics, cfg)

    paths = cfg["output"]
    for p in paths.values():
        ensure_dir(p)

    result.to_csv(paths["position_output_path"], index=False)
    minute_metrics.to_csv(paths["minute_metrics_path"], index=False)
    daily_metrics.to_csv(paths["daily_metrics_path"], index=False)
    reason_metrics.to_csv(paths["trade_reason_path"], index=False)
    summary.to_csv(paths["summary_path"], index=False)

    print()
    print("saved position output:", paths["position_output_path"])
    print("saved minute metrics:", paths["minute_metrics_path"])
    print("saved daily metrics:", paths["daily_metrics_path"])
    print("saved trade reason:", paths["trade_reason_path"])
    print("saved summary:", paths["summary_path"])

    print()
    print("===== summary =====")
    print(summary.T)

    print()
    print("===== trade reason =====")
    print(reason_metrics.to_string(index=False))


if __name__ == "__main__":
    main()
