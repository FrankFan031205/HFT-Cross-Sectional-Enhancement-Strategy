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


def find_col(df, candidates, required=True, name="column"):
    for c in candidates:
        c = str(c).lower()
        if c in df.columns:
            return c
    if required:
        raise ValueError(f"cannot find {name}, candidates={candidates}, existing={list(df.columns)[:80]}")
    return None


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


def auto_weight_col(df):
    return find_col(
        df,
        [
            "target_weight",
            "weight",
            "opt_weight",
            "optimizer_weight",
            "final_weight",
            "position_weight",
            "signed_weight",
            "trade_weight",
            "w",
        ],
        required=False,
        name="weight_col",
    )


def auto_price_cols(df):
    bid_col = find_col(
        df,
        ["bid1", "bid_price", "bidprice1", "bid_price1", "bid1_price", "best_bid", "bid"],
        required=False,
        name="bid_col",
    )
    ask_col = find_col(
        df,
        ["ask1", "ask_price", "askprice1", "ask_price1", "ask1_price", "best_ask", "ask"],
        required=False,
        name="ask_col",
    )
    mid_col = find_col(
        df,
        ["mid_price", "midprice", "mid", "wap_mid"],
        required=False,
        name="mid_col",
    )
    return bid_col, ask_col, mid_col


def prepare_optimizer(opt_raw, cfg):
    opt = lower_columns(opt_raw)
    cols = cfg["columns"]
    ta = cfg.get("time_alignment", {})

    datetime_col = str(cols.get("datetime_col", "datetime")).lower()
    symbol_col = str(cols.get("symbol_col", "securityid")).lower()
    side_col = str(cols.get("side_col", "side")).lower()
    status_col = str(cols.get("optimizer_status_col", "optimizer_status")).lower()

    if datetime_col not in opt.columns:
        raise ValueError(f"optimizer missing datetime_col={datetime_col}")
    if symbol_col not in opt.columns:
        raise ValueError(f"optimizer missing symbol_col={symbol_col}")

    weight_col_cfg = str(cols.get("weight_col", "auto")).lower()
    weight_col = auto_weight_col(opt) if weight_col_cfg == "auto" else weight_col_cfg

    if weight_col is None or weight_col not in opt.columns:
        opt["weight"] = 0.0
        weight_col = "weight"

    opt[datetime_col] = pd.to_datetime(opt[datetime_col], errors="coerce")
    opt[symbol_col] = normalize_symbol(opt[symbol_col])
    opt[weight_col] = pd.to_numeric(opt[weight_col], errors="coerce").fillna(0.0)

    if side_col in opt.columns:
        side = opt[side_col].map(standardize_side)
    else:
        side = np.where(opt[weight_col] > 0, "BUY", np.where(opt[weight_col] < 0, "SELL", "NONE"))

    if status_col in opt.columns:
        status = opt[status_col].astype(str)
    else:
        status = "unknown"

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

    out = pd.DataFrame(
        {
            "decision_datetime": decision_datetime,
            "execution_datetime": execution_datetime,
            "execution_minute": execution_datetime.dt.floor("min"),
            "securityid": opt[symbol_col],
            "raw_side": side,
            "raw_weight": opt[weight_col].astype(float),
            "optimizer_status": status,
        }
    )

    out["decision_date"] = out["decision_datetime"].dt.strftime("%Y%m%d")
    out["execution_date"] = out["execution_datetime"].dt.strftime("%Y%m%d")

    only_optimal = bool(cfg.get("execution", {}).get("only_optimal", True))

    if only_optimal:
        ok = out["optimizer_status"].str.lower().eq("optimal")
    else:
        ok = pd.Series(True, index=out.index)

    signed = np.where(
        out["raw_side"].eq("BUY"),
        out["raw_weight"].abs(),
        np.where(out["raw_side"].eq("SELL"), -out["raw_weight"].abs(), 0.0),
    )

    signed = np.where(ok, signed, 0.0)

    out["target_weight"] = signed.astype(float)
    out["target_side"] = np.where(
        out["target_weight"] > 0,
        "BUY",
        np.where(out["target_weight"] < 0, "SELL", "NONE"),
    )

    out = out.dropna(subset=["execution_datetime", "execution_minute", "securityid"])
    out = out.sort_values(["securityid", "execution_datetime"]).reset_index(drop=True)

    return out


def prepare_market(mkt_raw, cfg):
    mkt = lower_columns(mkt_raw)
    cols = cfg["columns"]
    ta = cfg.get("time_alignment", {})

    datetime_col = str(cols.get("datetime_col", "datetime")).lower()
    date_col = str(cols.get("date_col", "date")).lower()
    symbol_col = str(cols.get("symbol_col", "securityid")).lower()
    label_col = str(cols.get("label_col", "label_60")).lower()

    if datetime_col not in mkt.columns:
        raise ValueError(f"market missing datetime_col={datetime_col}")
    if symbol_col not in mkt.columns:
        raise ValueError(f"market missing symbol_col={symbol_col}")
    if label_col not in mkt.columns:
        raise ValueError(f"market missing label_col={label_col}")

    auto_bid, auto_ask, auto_mid = auto_price_cols(mkt)

    bid_cfg = str(cols.get("bid_col", "auto")).lower()
    ask_cfg = str(cols.get("ask_col", "auto")).lower()
    mid_cfg = str(cols.get("mid_col", "auto")).lower()

    bid_col = auto_bid if bid_cfg == "auto" else bid_cfg
    ask_col = auto_ask if ask_cfg == "auto" else ask_cfg
    mid_col = auto_mid if mid_cfg == "auto" else mid_cfg

    if mid_col is None or mid_col not in mkt.columns:
        raise ValueError("cannot find mid price column in market data")

    keep = [datetime_col, symbol_col, mid_col, label_col]

    if date_col in mkt.columns:
        keep.append(date_col)
    if bid_col is not None and bid_col in mkt.columns:
        keep.append(bid_col)
    if ask_col is not None and ask_col in mkt.columns:
        keep.append(ask_col)

    keep = list(dict.fromkeys(keep))

    mkt = mkt[keep].copy()
    mkt[datetime_col] = pd.to_datetime(mkt[datetime_col], errors="coerce")
    mkt[symbol_col] = normalize_symbol(mkt[symbol_col])

    rename = {
        datetime_col: "market_datetime",
        symbol_col: "securityid",
        mid_col: "mid_price",
        label_col: "label",
    }

    if date_col in mkt.columns:
        rename[date_col] = "market_date_raw"
    if bid_col is not None and bid_col in mkt.columns:
        rename[bid_col] = "bid_price"
    if ask_col is not None and ask_col in mkt.columns:
        rename[ask_col] = "ask_price"

    mkt = mkt.rename(columns=rename)

    for c in ["mid_price", "bid_price", "ask_price", "label"]:
        if c in mkt.columns:
            mkt[c] = pd.to_numeric(mkt[c], errors="coerce")

    if "bid_price" not in mkt.columns:
        mkt["bid_price"] = np.nan
    if "ask_price" not in mkt.columns:
        mkt["ask_price"] = np.nan

    mkt["execution_minute"] = mkt["market_datetime"].dt.floor("min")
    mkt["execution_date"] = mkt["execution_minute"].dt.strftime("%Y%m%d")

    price_time = str(ta.get("execution_price_time", "first_snapshot_of_minute")).lower()

    if price_time == "first_snapshot_of_minute":
        mkt = mkt.sort_values(["securityid", "execution_minute", "market_datetime"])
        mkt = mkt.drop_duplicates(["securityid", "execution_minute"], keep="first")
    elif price_time == "exact_time":
        mkt = mkt.rename(columns={"market_datetime": "execution_datetime"})
    else:
        raise ValueError("execution_price_time must be first_snapshot_of_minute or exact_time")

    return mkt.reset_index(drop=True)


def merge_optimizer_market(opt, mkt, cfg):
    price_time = str(cfg.get("time_alignment", {}).get("execution_price_time", "first_snapshot_of_minute")).lower()

    if price_time == "first_snapshot_of_minute":
        keys = ["securityid", "execution_minute"]
    else:
        keys = ["securityid", "execution_datetime"]

    return opt.merge(mkt, on=keys, how="left", suffixes=("", "_mkt"))


def compute_positions_and_pnl(df, cfg):
    exe = cfg.get("execution", {})

    capital = float(exe.get("capital", 200000000))
    taker_fee_bps = float(exe.get("taker_fee_bps", 1.5))
    slippage_bps = float(exe.get("slippage_bps", 0.0))
    allow_sell = bool(exe.get("allow_sell", True))
    min_abs_trade_weight = float(exe.get("min_abs_trade_weight", 0.0))

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

    df["target_notional"] = capital * df["abs_target_weight"]
    df["trade_notional"] = capital * df["abs_trade_weight"]

    df["gross_exposure"] = df["target_notional"]
    df["net_exposure"] = capital * df["target_weight"]

    df["gross_pnl"] = capital * df["target_weight"] * df["label"]

    df["spread_cost"] = df["trade_notional"] * df["spread_cost_ret"]
    df["fee"] = df["trade_notional"] * taker_fee_bps / 10000.0
    df["slippage"] = df["trade_notional"] * slippage_bps / 10000.0
    df["cost"] = df["spread_cost"] + df["fee"] + df["slippage"]

    df["net_pnl"] = df["gross_pnl"] - df["cost"]

    df["position_return_bps"] = df["label"] * np.sign(df["target_weight"]) * 10000.0
    df["pnl_bps_on_trade_turnover"] = df["net_pnl"] / df["trade_notional"].replace(0, np.nan) * 10000.0
    df["pnl_bps_on_gross_exposure"] = df["net_pnl"] / df["gross_exposure"].replace(0, np.nan) * 10000.0

    keep = (
        (df["abs_target_weight"] > 0)
        | (df["abs_trade_weight"] > min_abs_trade_weight)
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
        gross_weight=("abs_target_weight", "sum"),
        net_weight=("target_weight", "sum"),
        trade_weight=("abs_trade_weight", "sum"),
        gross_exposure=("gross_exposure", "sum"),
        net_exposure=("net_exposure", "sum"),
        turnover=("trade_notional", "sum"),
        gross_pnl=("gross_pnl", "sum"),
        spread_cost=("spread_cost", "sum"),
        fee=("fee", "sum"),
        cost=("cost", "sum"),
        net_pnl=("net_pnl", "sum"),
        avg_position_return_bps=("position_return_bps", "mean"),
    ).reset_index()

    out["pnl_bps_on_turnover"] = out["net_pnl"] / out["turnover"].replace(0, np.nan) * 10000.0
    out["pnl_bps_on_gross_exposure"] = out["net_pnl"] / out["gross_exposure"].replace(0, np.nan) * 10000.0

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
        gross_weight_sum=("abs_target_weight", "sum"),
        trade_weight_sum=("abs_trade_weight", "sum"),
        turnover=("trade_notional", "sum"),
        gross_pnl=("gross_pnl", "sum"),
        spread_cost=("spread_cost", "sum"),
        fee=("fee", "sum"),
        cost=("cost", "sum"),
        net_pnl=("net_pnl", "sum"),
        win_rate_rows=("net_pnl", lambda s: (s > 0).mean()),
    ).reset_index()

    out["pnl_bps_on_turnover"] = out["net_pnl"] / out["turnover"].replace(0, np.nan) * 10000.0

    out = out.sort_values("execution_date").reset_index(drop=True)

    out["cum_net_pnl"] = out["net_pnl"].cumsum()
    out["cum_turnover"] = out["turnover"].cumsum()

    return out


def make_summary(x, minute_metrics, cfg):
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
    total_net_pnl = x["net_pnl"].sum()

    max_drawdown = np.nan
    if minute_metrics is not None and not minute_metrics.empty:
        max_drawdown = minute_metrics["drawdown"].min()

    summary = {
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
        "total_gross_pnl": x["gross_pnl"].sum(),
        "total_spread_cost": x["spread_cost"].sum(),
        "total_fee": x["fee"].sum(),
        "total_cost": x["cost"].sum(),
        "total_net_pnl": total_net_pnl,
        "pnl_bps_on_turnover": total_net_pnl / total_turnover * 10000.0 if total_turnover > 0 else np.nan,
        "pnl_bps_on_gross_exposure": total_net_pnl / total_gross_exposure * 10000.0 if total_gross_exposure > 0 else np.nan,
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

    print("loading market data:", market_path)
    mkt_raw = pd.read_csv(market_path, low_memory=False)

    print("preparing optimizer")
    opt = prepare_optimizer(opt_raw, cfg)

    print("preparing market")
    mkt = prepare_market(mkt_raw, cfg)

    print("optimizer shape:", opt.shape)
    print("market execution rows:", mkt.shape)
    print("optimizer execution minute range:", opt["execution_minute"].min(), opt["execution_minute"].max())
    print("market execution minute range:", mkt["execution_minute"].min(), mkt["execution_minute"].max())

    print("merging optimizer with market")
    merged = merge_optimizer_market(opt, mkt, cfg)

    print("merged shape:", merged.shape)
    print("missing mid rate:", merged["mid_price"].isna().mean())
    print("missing label rate:", merged["label"].isna().mean())

    print("computing target-position taker pnl")
    result = compute_positions_and_pnl(merged, cfg)

    minute_metrics = make_minute_metrics(result)
    daily_metrics = make_daily_metrics(result)
    summary = make_summary(result, minute_metrics, cfg)

    paths = cfg["output"]

    for p in paths.values():
        ensure_dir(p)

    result.to_csv(paths["position_output_path"], index=False)
    minute_metrics.to_csv(paths["minute_metrics_path"], index=False)
    daily_metrics.to_csv(paths["daily_metrics_path"], index=False)
    summary.to_csv(paths["summary_path"], index=False)

    print()
    print("saved position output:", paths["position_output_path"])
    print("saved minute metrics:", paths["minute_metrics_path"])
    print("saved daily metrics:", paths["daily_metrics_path"])
    print("saved summary:", paths["summary_path"])
    print()
    print("===== summary =====")
    print(summary.T)


if __name__ == "__main__":
    main()