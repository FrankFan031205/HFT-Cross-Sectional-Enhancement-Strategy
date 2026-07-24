import argparse
from collections import defaultdict
import pandas as pd
import numpy as np


def sid(x):
    return str(x).replace(".0", "").zfill(6)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trades", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--skipped-output", required=True)

    parser.add_argument("--initial-position-per-symbol", type=float, default=5000)
    parser.add_argument("--max-position-per-symbol", type=float, default=15000)
    parser.add_argument("--sell-floor-position", type=float, default=2000)
    parser.add_argument("--buy-block-position", type=float, default=10000)

    parser.add_argument("--tplus1", type=int, default=1)
    parser.add_argument("--verbose", type=int, default=1)
    args = parser.parse_args()

    raw = pd.read_csv(args.trades, low_memory=False)
    df = raw.copy()

    df["fill_time"] = pd.to_datetime(df["fill_time"])
    df["securityid"] = df["securityid"].map(sid)
    df["side"] = df["side"].astype(str).str.upper()
    df["fill_qty"] = pd.to_numeric(df["fill_qty"], errors="coerce")

    df = df.dropna(subset=["fill_time", "securityid", "side", "fill_qty"]).copy()
    df = df.sort_values(["fill_time", "securityid"]).reset_index().rename(columns={"index": "original_index"})

    symbols = sorted(df["securityid"].unique())

    pos = defaultdict(float)
    sellable = defaultdict(float)

    for s in symbols:
        pos[s] = args.initial_position_per_symbol
        sellable[s] = args.initial_position_per_symbol

    accepted_idx = []
    skipped = []

    current_date = None

    for r in df.itertuples(index=False):
        date = r.fill_time.strftime("%Y%m%d")
        sym = r.securityid
        qty = float(r.fill_qty)

        if current_date is None:
            current_date = date

        if date != current_date:
            if args.tplus1:
                for s in symbols:
                    sellable[s] = max(pos[s], 0.0)
            current_date = date

        p0 = pos[sym]
        sellable0 = sellable[sym]

        if r.side == "BUY":
            if p0 + qty > args.max_position_per_symbol:
                skipped.append({
                    "original_index": r.original_index,
                    "fill_time": r.fill_time,
                    "securityid": sym,
                    "side": r.side,
                    "fill_qty": qty,
                    "reason": "hard_max_position",
                    "position_before": p0,
                    "sellable_before": sellable0,
                })
                continue

            if p0 >= args.buy_block_position:
                skipped.append({
                    "original_index": r.original_index,
                    "fill_time": r.fill_time,
                    "securityid": sym,
                    "side": r.side,
                    "fill_qty": qty,
                    "reason": "inventory_high_block_buy",
                    "position_before": p0,
                    "sellable_before": sellable0,
                })
                continue

            accepted_idx.append(r.original_index)
            pos[sym] += qty

            if not args.tplus1:
                sellable[sym] += qty

        elif r.side == "SELL":
            if sellable0 < qty:
                skipped.append({
                    "original_index": r.original_index,
                    "fill_time": r.fill_time,
                    "securityid": sym,
                    "side": r.side,
                    "fill_qty": qty,
                    "reason": "insufficient_sellable_inventory",
                    "position_before": p0,
                    "sellable_before": sellable0,
                })
                continue

            if p0 - qty < args.sell_floor_position:
                skipped.append({
                    "original_index": r.original_index,
                    "fill_time": r.fill_time,
                    "securityid": sym,
                    "side": r.side,
                    "fill_qty": qty,
                    "reason": "inventory_low_block_sell",
                    "position_before": p0,
                    "sellable_before": sellable0,
                })
                continue

            accepted_idx.append(r.original_index)
            pos[sym] -= qty
            sellable[sym] -= qty

        else:
            skipped.append({
                "original_index": r.original_index,
                "fill_time": r.fill_time,
                "securityid": sym,
                "side": r.side,
                "fill_qty": qty,
                "reason": "unknown_side",
                "position_before": p0,
                "sellable_before": sellable0,
            })

    accepted = raw.iloc[accepted_idx].copy()
    skipped_df = pd.DataFrame(skipped)

    accepted.to_csv(args.output, index=False)
    skipped_df.to_csv(args.skipped_output, index=False)

    ending = pd.DataFrame({
        "securityid": list(pos.keys()),
        "ending_position": [pos[s] for s in pos.keys()],
        "ending_sellable": [sellable[s] for s in pos.keys()],
    })

    ending_path = args.output.replace(".csv", "_ending_positions.csv")
    ending.to_csv(ending_path, index=False)

    print("\n===== inventory-aware overlay summary =====")
    print("input trades:", len(df))
    print("accepted trades:", len(accepted))
    print("skipped trades:", len(skipped_df))
    print("accept rate:", len(accepted) / len(df) if len(df) else np.nan)
    print("initial_position_per_symbol:", args.initial_position_per_symbol)
    print("max_position_per_symbol:", args.max_position_per_symbol)
    print("sell_floor_position:", args.sell_floor_position)
    print("buy_block_position:", args.buy_block_position)
    print("tplus1:", args.tplus1)

    if len(accepted):
        print("\naccepted side counts:")
        print(accepted["side"].value_counts())

    if len(skipped_df):
        print("\nskipped reason counts:")
        print(skipped_df["reason"].value_counts())

    print("\nsaved:", args.output)
    print("saved:", args.skipped_output)
    print("saved:", ending_path)


if __name__ == "__main__":
    main()
