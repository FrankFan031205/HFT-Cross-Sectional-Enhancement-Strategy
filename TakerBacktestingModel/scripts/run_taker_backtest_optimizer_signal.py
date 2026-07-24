import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import yaml


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def read_header(path: str):
    return list(pd.read_csv(path, nrows=0).columns)


def lower_map(cols):
    return {str(c).lower(): c for c in cols}


def choose_col(cols, configured, candidates, required=True, name="column"):
    if configured is not None and configured != "auto":
        if configured in cols:
            return configured
        lm = lower_map(cols)
        if str(configured).lower() in lm:
            return lm[str(configured).lower()]
        if required:
            raise ValueError(f"configured {name}={configured} not found in csv columns")
        return None

    lm = lower_map(cols)
    for c in candidates:
        if c in cols:
            return c
        if c.lower() in lm:
            return lm[c.lower()]

    if required:
        raise ValueError(
            f"cannot auto-detect {name}; "
            f"candidates={candidates}; columns={cols[:50]}"
        )
    return None


def normalize_symbol(s: pd.Series, zfill=None) -> pd.Series:
    out = s.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    if zfill is not None:
        out = out.str.zfill(int(zfill))
    return out


def to_float_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def apply_price_scale(df: pd.DataFrame, price_cols, scale):
    if scale is None:
        scale = 1

    if str(scale).lower() == "auto":
        vals = []
        for c in price_cols:
            if c in df.columns:
                sample = pd.to_numeric(df[c], errors="coerce").dropna()
                if len(sample):
                    vals.append(sample.median())
        med = np.nanmedian(vals) if vals else np.nan
        scale = 100 if np.isfinite(med) and med > 10000 else 1

    scale = float(scale)
    if scale != 1:
        for c in price_cols:
            if c in df.columns:
                df[c] = df[c] / scale
    return df


def load_market_minute(path: str, cfg: dict) -> pd.DataFrame:
    cols_cfg = cfg.get("columns", {})
    exe_cfg = cfg.get("execution", {})
    data_cfg = cfg.get("data", {})

    cols = read_header(path)

    dt_col = choose_col(
        cols,
        cols_cfg.get("datetime_col", "datetime"),
        ["datetime", "time", "timestamp", "DateTime"],
        True,
        "datetime_col",
    )
    sym_col = choose_col(
        cols,
        cols_cfg.get("symbol_col", "securityid"),
        ["securityid", "SecurityID", "symbol", "ticker", "instrument"],
        True,
        "symbol_col",
    )
    bid_col = choose_col(
        cols,
        cols_cfg.get("bid_col", "auto"),
        ["bid1", "bid_price1", "bidprice1", "best_bid", "bid"],
        False,
        "bid_col",
    )
    ask_col = choose_col(
        cols,
        cols_cfg.get("ask_col", "auto"),
        ["ask1", "ask_price1", "askprice1", "best_ask", "ask"],
        False,
        "ask_col",
    )
    mid_col = choose_col(
        cols,
        cols_cfg.get("mid_col", "auto"),
        ["mid_price", "midprice", "mid", "wap", "close"],
        False,
        "mid_col",
    )

    if mid_col is None and (bid_col is None or ask_col is None):
        raise ValueError("market csv must contain either mid price, or both bid and ask")

    usecols = [c for c in [dt_col, sym_col, bid_col, ask_col, mid_col] if c is not None]
    chunksize = int(data_cfg.get("chunksize", 2_000_000))
    zfill = cols_cfg.get("symbol_zfill", None)
    price_scale = exe_cfg.get("price_scale", 1)

    print(f"loading market data: {path}")
    print(
        f"market columns: datetime={dt_col}, symbol={sym_col}, "
        f"bid={bid_col}, ask={ask_col}, mid={mid_col}"
    )

    parts = []
    reader = pd.read_csv(path, usecols=usecols, chunksize=chunksize)

    for i, chunk in enumerate(reader, 1):
        rename = {
            dt_col: "datetime",
            sym_col: "securityid",
        }
        if bid_col is not None:
            rename[bid_col] = "bid"
        if ask_col is not None:
            rename[ask_col] = "ask"
        if mid_col is not None:
            rename[mid_col] = "mid"

        chunk = chunk.rename(columns=rename)

        chunk["datetime"] = pd.to_datetime(chunk["datetime"], errors="coerce")
        chunk["securityid"] = normalize_symbol(chunk["securityid"], zfill)
        chunk = chunk.dropna(subset=["datetime", "securityid"])
        chunk["minute"] = chunk["datetime"].dt.floor("min")

        for c in ["bid", "ask", "mid"]:
            if c in chunk.columns:
                chunk[c] = to_float_series(chunk[c])

        chunk = apply_price_scale(chunk, ["bid", "ask", "mid"], price_scale)

        if "mid" not in chunk.columns:
            chunk["mid"] = (chunk["bid"] + chunk["ask"]) / 2.0
        if "bid" not in chunk.columns:
            chunk["bid"] = np.nan
        if "ask" not in chunk.columns:
            chunk["ask"] = np.nan

        chunk = chunk.replace([np.inf, -np.inf], np.nan)
        chunk = chunk.dropna(subset=["minute", "securityid", "mid"])

        chunk = chunk.sort_values(["minute", "securityid", "datetime"])
        part = chunk.groupby(["minute", "securityid"], as_index=False).last()
        part = part[["minute", "securityid", "bid", "ask", "mid"]]

        parts.append(part)

        print(
            f"  market chunk {i}: "
            f"minute rows={len(part)}, total parts={len(parts)}"
        )

    if not parts:
        raise ValueError("no market data loaded")

    market = pd.concat(parts, ignore_index=True)
    market = market.sort_values(["minute", "securityid"])
    market = market.groupby(["minute", "securityid"], as_index=False).last()
    market = market.sort_values(["minute", "securityid"]).reset_index(drop=True)

    print(
        f"market minute rows: {len(market)}, "
        f"minutes={market['minute'].nunique()}, "
        f"symbols={market['securityid'].nunique()}"
    )

    return market


def side_to_sign(side: Any) -> int:
    if pd.isna(side):
        return 0

    s = str(side).strip().upper()

    if s in {"BUY", "B", "LONG", "1", "+1"}:
        return 1
    if s in {"SELL", "S", "SHORT", "-1"}:
        return -1

    return 0


def load_signal(path: str, cfg: dict) -> pd.DataFrame:
    cols_cfg = cfg.get("columns", {})
    data_cfg = cfg.get("data", {})
    exe_cfg = cfg.get("execution", {})

    cols = read_header(path)

    dt_col = choose_col(
        cols,
        cols_cfg.get("datetime_col", "datetime"),
        ["datetime", "minute", "time", "timestamp", "DateTime"],
        True,
        "datetime_col",
    )
    sym_col = choose_col(
        cols,
        cols_cfg.get("symbol_col", "securityid"),
        ["securityid", "SecurityID", "symbol", "ticker", "instrument"],
        True,
        "symbol_col",
    )
    side_col = choose_col(
        cols,
        cols_cfg.get("side_col", "auto"),
        ["side", "trade_side", "target_side", "signal_side"],
        False,
        "side_col",
    )
    status_col = choose_col(
        cols,
        cols_cfg.get("status_col", "auto"),
        ["optimizer_status", "status", "solve_status"],
        False,
        "status_col",
    )
    selected_col = choose_col(
        cols,
        cols_cfg.get("selected_col", "auto"),
        ["selected", "is_selected", "trade", "active"],
        False,
        "selected_col",
    )
    target_weight_col = choose_col(
        cols,
        cols_cfg.get("target_weight_col", "auto"),
        [
            "target_weight",
            "target_w",
            "signed_weight",
            "signed_w",
            "position_weight",
            "net_weight",
            "target_position_weight",
        ],
        False,
        "target_weight_col",
    )
    weight_col = choose_col(
        cols,
        cols_cfg.get("weight_col", "auto"),
        [
            "weight",
            "w",
            "opt_weight",
            "abs_weight",
            "selected_weight",
            "gross_weight",
            "notional_weight",
        ],
        False,
        "weight_col",
    )

    usecols = [
        c
        for c in [
            dt_col,
            sym_col,
            side_col,
            status_col,
            selected_col,
            target_weight_col,
            weight_col,
        ]
        if c is not None
    ]

    sig = pd.read_csv(path, usecols=usecols)

    rename = {
        dt_col: "datetime",
        sym_col: "securityid",
    }
    if side_col is not None:
        rename[side_col] = "side"
    if status_col is not None:
        rename[status_col] = "optimizer_status"
    if selected_col is not None:
        rename[selected_col] = "selected"
    if target_weight_col is not None:
        rename[target_weight_col] = "target_weight_raw"
    if weight_col is not None and weight_col != target_weight_col:
        rename[weight_col] = "weight_raw"

    sig = sig.rename(columns=rename)

    zfill = cols_cfg.get("symbol_zfill", None)

    sig["datetime"] = pd.to_datetime(sig["datetime"], errors="coerce")
    sig["securityid"] = normalize_symbol(sig["securityid"], zfill)
    sig = sig.dropna(subset=["datetime", "securityid"])
    sig["minute"] = sig["datetime"].dt.floor("min")

    if "side" in sig.columns:
        sig["side_sign"] = sig["side"].map(side_to_sign).astype(int)
    else:
        sig["side_sign"] = 0

    if "selected" in sig.columns:
        selected_num = pd.to_numeric(sig["selected"], errors="coerce")
        if selected_num.notna().any():
            sig["selected_bool"] = selected_num.fillna(0).astype(float) != 0
        else:
            sig["selected_bool"] = (
                sig["selected"]
                .astype(str)
                .str.lower()
                .isin(["true", "t", "yes", "y"])
            )
    else:
        sig["selected_bool"] = sig["side_sign"] != 0

    if "target_weight_raw" in sig.columns:
        sig["target_weight"] = to_float_series(sig["target_weight_raw"]).fillna(0.0)

        if "side" in sig.columns:
            none_mask = sig["side_sign"] == 0
            sig.loc[none_mask, "target_weight"] = 0.0

    elif "weight_raw" in sig.columns:
        w = to_float_series(sig["weight_raw"]).fillna(0.0)

        if "side" in sig.columns:
            sig["target_weight"] = w.abs() * sig["side_sign"]
        else:
            sig["target_weight"] = w

    else:
        active = (sig["side_sign"] != 0) & sig["selected_bool"]
        n_active = sig.loc[active].groupby("minute")["securityid"].transform("count")
        gross = float(exe_cfg.get("target_gross_weight_if_missing", 0.10))

        sig["target_weight"] = 0.0
        sig.loc[active, "target_weight"] = (
            sig.loc[active, "side_sign"] * gross / n_active.replace(0, np.nan)
        )
        sig["target_weight"] = sig["target_weight"].fillna(0.0)

    if bool(exe_cfg.get("selected_only", True)):
        sig.loc[
            ~sig["selected_bool"] & (sig["side_sign"] == 0),
            "target_weight",
        ] = 0.0

    ok_status = data_cfg.get("optimizer_status_ok", ["optimal"])
    if ok_status is not None and "optimizer_status" in sig.columns:
        ok_set = {str(x).lower() for x in ok_status}
        bad = ~sig["optimizer_status"].astype(str).str.lower().isin(ok_set)
        sig.loc[bad, "target_weight"] = 0.0

    delay = int(exe_cfg.get("signal_delay_minutes", 1))
    sig["minute"] = sig["minute"] + pd.to_timedelta(delay, unit="min")

    sig = sig.replace([np.inf, -np.inf], np.nan)
    sig["target_weight"] = sig["target_weight"].fillna(0.0)

    sig = sig.sort_values(["minute", "securityid", "datetime"])
    sig = sig.groupby(["minute", "securityid"], as_index=False).last()
    sig = sig[["minute", "securityid", "target_weight"]]

    date_start = data_cfg.get("date_start")
    date_end = data_cfg.get("date_end")

    if date_start is not None:
        sig = sig[sig["minute"] >= pd.to_datetime(str(date_start))]
    if date_end is not None:
        sig = sig[
            sig["minute"] < pd.to_datetime(str(date_end)) + pd.Timedelta(days=1)
        ]

    print(f"loading signal: {path}")
    print(
        f"signal columns: datetime={dt_col}, symbol={sym_col}, "
        f"side={side_col}, target_weight={target_weight_col}, "
        f"weight={weight_col}, status={status_col}, selected={selected_col}"
    )
    print(
        f"signal rows: {len(sig)}, "
        f"minutes={sig['minute'].nunique()}, "
        f"symbols={sig['securityid'].nunique()}"
    )
    print(
        "target weight abs sum per minute:\n"
        f"{sig.groupby('minute')['target_weight'].apply(lambda x: x.abs().sum()).describe()}"
    )

    return sig


def round_qty(qty: float, lot: int) -> float:
    if lot is None or int(lot) <= 1:
        return float(qty)

    lot = int(lot)
    return float(np.sign(qty) * math.floor(abs(qty) / lot) * lot)


def get_exec_price(quote: dict, direction: int, fallback_half_spread_bps: float) -> float:
    mid = quote.get("mid", np.nan)
    bid = quote.get("bid", np.nan)
    ask = quote.get("ask", np.nan)

    if direction > 0:
        if pd.notna(ask) and ask > 0:
            return float(ask)
        return float(mid) * (1.0 + fallback_half_spread_bps / 10000.0)

    if pd.notna(bid) and bid > 0:
        return float(bid)
    return float(mid) * (1.0 - fallback_half_spread_bps / 10000.0)


def make_summary(
    equity_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    cfg: dict,
) -> dict:
    capital = float(cfg.get("execution", {}).get("initial_capital", 200_000_000))

    if len(equity_df) == 0:
        return {}

    final_equity = float(equity_df["equity"].iloc[-1])
    total_pnl = final_equity - capital

    total_turnover = float(trades_df["notional"].sum()) if len(trades_df) else 0.0
    total_cost = float(trades_df["total_cost"].sum()) if len(trades_df) else 0.0

    eq = equity_df["equity"].astype(float)
    running_max = eq.cummax()
    dd = eq - running_max
    dd_pct = eq / running_max - 1.0

    daily_last = equity_df.copy()
    daily_last["date"] = pd.to_datetime(daily_last["datetime"]).dt.date
    daily_last = daily_last.groupby("date", as_index=False).last()
    daily_last["daily_pnl"] = daily_last["equity"].diff()

    if len(daily_last):
        daily_last.loc[daily_last.index[0], "daily_pnl"] = (
            daily_last.loc[daily_last.index[0], "equity"] - capital
        )

    daily_ret_on_capital = daily_last["daily_pnl"] / capital

    sharpe = np.nan
    if len(daily_ret_on_capital) > 1 and daily_ret_on_capital.std(ddof=1) > 0:
        sharpe = float(
            daily_ret_on_capital.mean()
            / daily_ret_on_capital.std(ddof=1)
            * np.sqrt(242)
        )

    out = {
        "initial_capital": capital,
        "final_equity": final_equity,
        "total_pnl": float(total_pnl),
        "return_on_capital": float(total_pnl / capital),
        "num_trades": int(len(trades_df)),
        "num_buy_trades": int((trades_df["trade_qty"] > 0).sum())
        if len(trades_df)
        else 0,
        "num_sell_trades": int((trades_df["trade_qty"] < 0).sum())
        if len(trades_df)
        else 0,
        "total_turnover": total_turnover,
        "pnl_bps_on_turnover": float(total_pnl / total_turnover * 10000.0)
        if total_turnover > 0
        else np.nan,
        "total_cost": total_cost,
        "cost_bps_on_turnover": float(total_cost / total_turnover * 10000.0)
        if total_turnover > 0
        else np.nan,
        "max_drawdown": float(dd.min()),
        "max_drawdown_pct": float(dd_pct.min()),
        "max_gross_exposure": float(equity_df["gross_exposure"].max())
        if "gross_exposure" in equity_df
        else np.nan,
        "max_abs_net_exposure": float(equity_df["net_exposure"].abs().max())
        if "net_exposure" in equity_df
        else np.nan,
        "daily_sharpe_on_capital": sharpe,
        "start_time": str(equity_df["datetime"].iloc[0]),
        "end_time": str(equity_df["datetime"].iloc[-1]),
        "num_minutes": int(len(equity_df)),
    }

    return out


def run_backtest(
    market: pd.DataFrame,
    signal: pd.DataFrame,
    cfg: dict,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    exe_cfg = cfg.get("execution", {})

    capital = float(exe_cfg.get("initial_capital", 200_000_000))
    commission_bps = float(exe_cfg.get("commission_bps", 1.0))
    stamp_tax_bps = float(exe_cfg.get("stamp_tax_bps", 5.0))
    slippage_bps = float(exe_cfg.get("slippage_bps", 0.0))
    fallback_half_spread_bps = float(exe_cfg.get("fallback_half_spread_bps", 2.0))

    round_lot = int(exe_cfg.get("round_lot", 100))
    min_trade_notional = float(exe_cfg.get("min_trade_notional", 0.0))

    allow_short = bool(exe_cfg.get("allow_short", True))
    liquidate_missing = bool(exe_cfg.get("liquidate_missing_signals", True))

    market = market.sort_values(["minute", "securityid"]).reset_index(drop=True)
    signal = signal.sort_values(["minute", "securityid"]).reset_index(drop=True)

    market_groups = {k: v for k, v in market.groupby("minute", sort=True)}
    signal_groups = {k: v for k, v in signal.groupby("minute", sort=True)}

    times = sorted(market_groups.keys())

    cash = capital
    positions: Dict[str, float] = {}
    last_quote: Dict[str, dict] = {}

    trades = []
    equity_rows = []

    cumulative_turnover = 0.0
    cumulative_cost = 0.0

    print("running taker backtest...")

    for idx, t in enumerate(times, 1):
        mdf = market_groups[t]

        for r in mdf.itertuples(index=False):
            last_quote[r.securityid] = {
                "bid": r.bid,
                "ask": r.ask,
                "mid": r.mid,
            }

        sdf = signal_groups.get(t)
        target_weights = {}

        if sdf is not None:
            target_weights = dict(zip(sdf["securityid"], sdf["target_weight"]))

        if sdf is not None:
            exec_symbols = set(target_weights.keys())

            if liquidate_missing:
                exec_symbols |= set(positions.keys())
        else:
            exec_symbols = set()

        for sym in sorted(exec_symbols):
            quote = last_quote.get(sym)

            if quote is None:
                continue

            if pd.isna(quote.get("mid")) or quote.get("mid", 0) <= 0:
                continue

            if sym in target_weights:
                target_weight = float(target_weights.get(sym, 0.0))
                target_qty = target_weight * capital / float(quote["mid"])

                if not allow_short and target_qty < 0:
                    target_qty = 0.0

                target_qty = round_qty(target_qty, round_lot)
            else:
                target_weight = 0.0
                target_qty = 0.0

            prev_qty = float(positions.get(sym, 0.0))
            trade_qty = target_qty - prev_qty

            if abs(trade_qty) < 1e-9:
                continue

            direction = 1 if trade_qty > 0 else -1
            exec_price = get_exec_price(quote, direction, fallback_half_spread_bps)

            notional = abs(trade_qty) * exec_price

            if notional < min_trade_notional and target_qty != 0:
                continue

            commission = notional * commission_bps / 10000.0
            stamp_tax = notional * stamp_tax_bps / 10000.0 if direction < 0 else 0.0
            slippage = notional * slippage_bps / 10000.0
            total_cost = commission + stamp_tax + slippage

            cash -= trade_qty * exec_price
            cash -= total_cost

            if abs(target_qty) < 1e-9:
                positions.pop(sym, None)
            else:
                positions[sym] = target_qty

            cumulative_turnover += notional
            cumulative_cost += total_cost

            trades.append(
                {
                    "datetime": t,
                    "securityid": sym,
                    "target_weight": target_weight,
                    "prev_qty": prev_qty,
                    "target_qty": target_qty,
                    "trade_qty": trade_qty,
                    "trade_side": "BUY" if direction > 0 else "SELL",
                    "exec_price": exec_price,
                    "mid_price": float(quote["mid"]),
                    "bid": float(quote["bid"])
                    if pd.notna(quote.get("bid"))
                    else np.nan,
                    "ask": float(quote["ask"])
                    if pd.notna(quote.get("ask"))
                    else np.nan,
                    "notional": notional,
                    "commission": commission,
                    "stamp_tax": stamp_tax,
                    "slippage": slippage,
                    "total_cost": total_cost,
                    "cash_after": cash,
                }
            )

        position_value = 0.0
        gross_value = 0.0
        net_value = 0.0

        for sym, qty in positions.items():
            quote = last_quote.get(sym)

            if quote is None or pd.isna(quote.get("mid")):
                continue

            value = qty * float(quote["mid"])
            position_value += value
            gross_value += abs(value)
            net_value += value

        equity = cash + position_value

        equity_rows.append(
            {
                "datetime": t,
                "cash": cash,
                "position_value": position_value,
                "equity": equity,
                "pnl": equity - capital,
                "return_on_capital": (equity - capital) / capital,
                "gross_exposure": gross_value / capital,
                "net_exposure": net_value / capital,
                "num_positions": int(
                    sum(1 for q in positions.values() if abs(q) > 1e-9)
                ),
                "cumulative_turnover": cumulative_turnover,
                "cumulative_cost": cumulative_cost,
            }
        )

        if idx % 500 == 0:
            print(
                f"  minute {idx}/{len(times)}: {t}, "
                f"trades={len(trades)}, pnl={equity - capital:.2f}"
            )

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_rows)

    summary = make_summary(equity_df, trades_df, cfg)

    return trades_df, equity_df, summary


def json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)

    if isinstance(o, (np.floating,)):
        if np.isnan(o):
            return None
        return float(o)

    if isinstance(o, (pd.Timestamp,)):
        return str(o)

    return str(o)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="TakerBacktestingModel/config/taker_backtest_optimizer_signal_v1.yaml",
    )
    parser.add_argument("--signal-path", default=None)
    parser.add_argument("--market-data-path", default=None)
    parser.add_argument("--tag", default=None)

    args = parser.parse_args()

    cfg = load_yaml(args.config)

    if args.signal_path is not None:
        cfg.setdefault("data", {})["optimizer_signal_path"] = args.signal_path

    if args.market_data_path is not None:
        cfg.setdefault("data", {})["market_data_path"] = args.market_data_path

    if args.tag is not None:
        cfg.setdefault("outputs", {})["tag"] = args.tag

    data_cfg = cfg.get("data", {})
    out_cfg = cfg.get("outputs", {})

    signal_path = data_cfg["optimizer_signal_path"]
    market_path = data_cfg["market_data_path"]
    output_dir = data_cfg.get("output_dir", "TakerBacktestingModel/outputs")

    tag = out_cfg.get("tag", "optimizer_signal_v1")

    trades_dir = os.path.join(output_dir, "trades")
    metrics_dir = os.path.join(output_dir, "metrics")

    ensure_dir(trades_dir)
    ensure_dir(metrics_dir)

    signal = load_signal(signal_path, cfg)
    market = load_market_minute(market_path, cfg)

    date_start = data_cfg.get("date_start")
    date_end = data_cfg.get("date_end")

    if date_start is not None:
        market = market[market["minute"] >= pd.to_datetime(str(date_start))]

    if date_end is not None:
        market = market[
            market["minute"] < pd.to_datetime(str(date_end)) + pd.Timedelta(days=1)
        ]

    trades_df, equity_df, summary = run_backtest(market, signal, cfg)

    trades_path = os.path.join(trades_dir, f"taker_trades_{tag}.csv")
    equity_path = os.path.join(metrics_dir, f"taker_equity_curve_{tag}.csv")
    summary_json_path = os.path.join(metrics_dir, f"taker_summary_{tag}.json")
    summary_csv_path = os.path.join(metrics_dir, f"taker_summary_{tag}.csv")

    if bool(out_cfg.get("save_trades", True)):
        trades_df.to_csv(trades_path, index=False)

    if bool(out_cfg.get("save_equity_curve", True)):
        equity_df.to_csv(equity_path, index=False)

    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=json_default)

    pd.DataFrame([summary]).to_csv(summary_csv_path, index=False)

    print("\n===== taker backtest summary =====")
    for k, v in summary.items():
        print(f"{k}: {v}")

    print(f"\nsaved trades: {trades_path}")
    print(f"saved equity: {equity_path}")
    print(f"saved summary: {summary_json_path}")


if __name__ == "__main__":
    main()