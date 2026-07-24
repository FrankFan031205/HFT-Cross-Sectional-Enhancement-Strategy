import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.io import (
    load_yaml,
    load_quote_decisions,
    load_snapshot_state_for_quotes,
    load_trade_replay_from_csv,
    standardize_trades,
    save_csv,
)
from src.db import get_clickhouse_client, fetch_trades_from_db
from src.fill_model import QueueAwareTradeFillModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/backtest.yaml")
    parser.add_argument("--max-quotes", type=int, default=None)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    print("[1] loading quote decisions")
    quotes = load_quote_decisions(cfg)

    if args.max_quotes is not None:
        quotes = quotes.head(args.max_quotes).copy()

    print("quotes shape:", quotes.shape)
    print("quote date range:", quotes["datetime"].min(), "->", quotes["datetime"].max())
    print("num symbols:", quotes["securityid"].nunique())

    print("[2] loading snapshot state by chunks, no 18G copy")
    quotes = load_snapshot_state_for_quotes(cfg, quotes)

    required_book_cols = []
    for i in range(1, 11):
        required_book_cols += [f"bid{i}", f"ask{i}", f"bid{i}_volume", f"ask{i}_volume"]

    missing_book_cols = [c for c in required_book_cols if c not in quotes.columns]
    if missing_book_cols:
        raise RuntimeError(
            "Missing book columns for queue estimation: "
            + str(missing_book_cols[:20])
            + (" ..." if len(missing_book_cols) > 20 else "")
        )

    trade_replay_path = cfg["input"].get("trade_replay_path", "")
    trade_cache_path = cfg["output"].get("trade_cache_path", "")

    if trade_replay_path:
        print("[3] loading raw trade replay csv")
        trades = load_trade_replay_from_csv(trade_replay_path)
        trades = standardize_trades(trades, cfg)
    else:
        cache_exists = bool(trade_cache_path) and Path(trade_cache_path).exists() and not args.no_cache

        if cache_exists:
            print("[3] loading cached raw trades:", trade_cache_path)
            trades = load_trade_replay_from_csv(trade_cache_path)
            trades = standardize_trades(trades, cfg)
        else:
            print("[3] querying raw trades from ClickHouse")
            first_date = int(quotes["datetime"].dt.strftime("%Y%m%d").iloc[0])
            client = get_clickhouse_client(cfg, date=first_date)
            trades = fetch_trades_from_db(client, quotes, cfg)
            trades = standardize_trades(trades, cfg)

            if trade_cache_path:
                save_csv(trades, trade_cache_path)

    print("trades shape:", trades.shape)
    if len(trades) > 0:
        print("trade date range:", trades["datetime"].min(), "->", trades["datetime"].max())
        print("trade side counts:")
        print(trades["side"].value_counts(dropna=False).head(20))

    print("[4] running queue-aware fill simulation")
    model = QueueAwareTradeFillModel(cfg)
    fills = model.simulate(quotes, trades, max_quotes=None)

    print("[5] saving fills")
    save_csv(fills, cfg["output"]["fill_path"])

    if len(fills) > 0:
        print("fills by side:")
        print(fills.groupby("side")["fill_qty"].agg(["count", "sum", "mean"]))
        print("queue_source:")
        print(fills["queue_source"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
