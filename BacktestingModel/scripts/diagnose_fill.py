import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.io import (
    load_yaml,
    load_quote_decisions,
    load_snapshot_state_for_quotes,
    load_trade_replay_from_csv,
    standardize_trades,
)
from src.fill_model import normalize_trade_side, estimate_queue_ahead


def as_bool(x):
    if isinstance(x, bool):
        return x
    return str(x).strip().lower() in {"1", "true", "t", "yes", "y"}


def main():
    cfg = load_yaml("config/backtest.yaml")

    quotes = load_quote_decisions(cfg)
    quotes = quotes.head(5000).copy()
    quotes = load_snapshot_state_for_quotes(cfg, quotes)

    trade_cache = cfg["output"]["trade_cache_path"]
    trades = load_trade_replay_from_csv(trade_cache)
    trades = standardize_trades(trades, cfg)
    trades = normalize_trade_side(trades, cfg)

    latency = pd.Timedelta(milliseconds=int(cfg["backtest"]["latency_ms"]))
    ttl = pd.Timedelta(milliseconds=int(cfg["backtest"]["quote_ttl_ms"]))

    print("\n===== quote price stats =====")
    cols = [c for c in ["bid1", "ask1", "bid_quote_price", "ask_quote_price"] if c in quotes.columns]
    print(quotes[cols].describe().T)

    print("\n===== trade price stats =====")
    print(trades["price"].describe())
    print("\ntrade side normalized counts:")
    print(trades["side_norm"].value_counts(dropna=False))

    trade_groups = {}
    for sid, g in trades.groupby("securityid"):
        trade_groups[sid] = g.sort_values("datetime").reset_index(drop=True)

    records = []

    for _, row in quotes.iterrows():
        sid = row["securityid"]
        tg = trade_groups.get(sid)
        if tg is None or len(tg) == 0:
            continue

        start = row["datetime"] + latency
        end = start + ttl
        w = tg[(tg["datetime"] >= start) & (tg["datetime"] < end)]

        if as_bool(row.get("quote_bid", False)):
            p = float(row["bid_quote_price"])
            q0, level, source = estimate_queue_ahead(row, "BUY", p, cfg)

            opp = w[w["side_norm"] == "SELL"]
            touched = opp[opp["price"] <= p]
            same_level = opp[np.isclose(opp["price"], p, atol=float(cfg["backtest"]["price_eps"]))]
            crossed = opp[opp["price"] < p - float(cfg["backtest"]["price_eps"])]

            same_qty = same_level["qty"].sum() if len(same_level) else 0.0

            records.append({
                "side": "BUY",
                "quote_price": p,
                "book_level": level,
                "queue_source": source,
                "queue_ahead": q0,
                "n_opp_trades": len(opp),
                "n_touched": len(touched),
                "n_same_level": len(same_level),
                "n_crossed": len(crossed),
                "same_level_qty": same_qty,
                "queue_exhausted_by_same_level": same_qty > q0 if np.isfinite(q0) else False,
            })

        if as_bool(row.get("quote_ask", False)):
            p = float(row["ask_quote_price"])
            q0, level, source = estimate_queue_ahead(row, "SELL", p, cfg)

            opp = w[w["side_norm"] == "BUY"]
            touched = opp[opp["price"] >= p]
            same_level = opp[np.isclose(opp["price"], p, atol=float(cfg["backtest"]["price_eps"]))]
            crossed = opp[opp["price"] > p + float(cfg["backtest"]["price_eps"])]

            same_qty = same_level["qty"].sum() if len(same_level) else 0.0

            records.append({
                "side": "SELL",
                "quote_price": p,
                "book_level": level,
                "queue_source": source,
                "queue_ahead": q0,
                "n_opp_trades": len(opp),
                "n_touched": len(touched),
                "n_same_level": len(same_level),
                "n_crossed": len(crossed),
                "same_level_qty": same_qty,
                "queue_exhausted_by_same_level": same_qty > q0 if np.isfinite(q0) else False,
            })

    d = pd.DataFrame(records)
    if len(d) == 0:
        print("no quote orders found")
        return

    print("\n===== order count =====")
    print(d["side"].value_counts())

    print("\n===== queue source =====")
    print(d["queue_source"].value_counts(dropna=False))

    print("\n===== book level =====")
    print(d.groupby(["side", "book_level"]).size())

    print("\n===== touched diagnostics by side =====")
    out = d.groupby("side").agg(
        orders=("side", "size"),
        touched_orders=("n_touched", lambda x: int((x > 0).sum())),
        same_level_orders=("n_same_level", lambda x: int((x > 0).sum())),
        crossed_orders=("n_crossed", lambda x: int((x > 0).sum())),
        queue_exhausted_orders=("queue_exhausted_by_same_level", "sum"),
        avg_queue_ahead=("queue_ahead", lambda x: np.nanmean([v for v in x if np.isfinite(v)])),
        avg_same_level_qty=("same_level_qty", "mean"),
    )
    out["touched_rate"] = out["touched_orders"] / out["orders"]
    out["same_level_rate"] = out["same_level_orders"] / out["orders"]
    out["crossed_rate"] = out["crossed_orders"] / out["orders"]
    print(out)

    print("\n===== examples touched but not queue exhausted =====")
    ex = d[(d["n_touched"] > 0) & (~d["queue_exhausted_by_same_level"])].head(20)
    print(ex.to_string(index=False))

    d.to_csv("outputs/cache/fill_diagnostics_5000.csv", index=False)
    print("\nsaved: outputs/cache/fill_diagnostics_5000.csv")


if __name__ == "__main__":
    main()
