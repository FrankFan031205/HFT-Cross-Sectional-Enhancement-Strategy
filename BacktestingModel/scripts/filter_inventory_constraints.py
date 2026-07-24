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
    parser.add_argument("--initial-position-per-symbol", type=float, default=0.0)
    parser.add_argument("--max-position-per-symbol", type=float, default=0.0)
    parser.add_argument("--tplus1", type=int, default=1)
    args = parser.parse_args()

    print("[1] loading trades:", args.trades)
    df = pd.read_csv(args.trades, low_memory=False)

    df["fill_time"] = pd.to_datetime(df["fill_time"])
    df["securityid"] = df["securityid"].map(sid)
    df["side"] = df["side"].astype(str).str.upper()
    df["fill_qty"] = pd.to_numeric(df["fill_qty"], errors="coerce")

    df = df.dropna(subset=["fill_time", "securityid", "side", "fill_qty"]).copy()
    df = df.sort_values(["fill_time", "securityid"]).reset_index(drop=False).rename(columns={"index": "original_index"})

    symbols = sorted(df["securityid"].unique())

    pos = defaultdict(float)
    sellable = defaultdict(float)

    for s in symbols:
        pos[s] = args.initial_position_per_symbol
        sellable[s] = args.initial_position_per_symbol

    max_pos = args.max_position_per_symbol
    use_max_pos = max_pos > 0

    accepted_idx = []
    skipped = []

    current_date = None

    for r in df.itertuples(index=False):
        date = r.fill_time.strftime("%Y%m%d")
        s = r.securityid
        qty = float(r.fill_qty)

        if current_date is None:
            current_date = date

        if date != current_date:
            if args.tplus1:
                for sym in symbols:
                    sellable[sym] = max(pos[sym], 0.0)
            current_date = date

        if r.side == "BUY":
            if use_max_pos and pos[s] + qty > max_pos:
                skipped.append({
                    "original_index": r.original_index,
                    "fill_time": r.fill_time,
                    "securityid": s,
                    "side": r.side,
                    "fill_qty": qty,
                    "reason": "max_position",
                    "position_before": pos[s],
                    "sellable_before": sellable[s],
                })
                continue

            accepted_idx.append(r.original_index)
            pos[s] += qty

            if not args.tplus1:
                sellable[s] += qty

        elif r.side == "SELL":
            if sellable[s] < qty:
                skipped.append({
                    "original_index": r.original_index,
                    "fill_time": r.fill_time,
                    "securityid": s,
                    "side": r.side,
                    "fill_qty": qty,
                    "reason": "insufficient_sellable_inventory",
                    "position_before": pos[s],
                    "sellable_before": sellable[s],
                })
                continue

            accepted_idx.append(r.original_index)
            pos[s] -= qty
            sellable[s] -= qty

        else:
            skipped.append({
                "original_index": r.original_index,
                "fill_time": r.fill_time,
                "securityid": s,
                "side": r.side,
                "fill_qty": qty,
                "reason": "unknown_side",
                "position_before": pos[s],
                "sellable_before": sellable[s],
            })

    accepted = pd.read_csv(args.trades, low_memory=False).iloc[accepted_idx].copy()
    skipped_df = pd.DataFrame(skipped)

    accepted.to_csv(args.output, index=False)
    skipped_df.to_csv(args.skipped_output, index=False)

    print("\n===== inventory constraint summary =====")
    print("input trades:", len(df))
    print("accepted trades:", len(accepted))
    print("skipped trades:", len(skipped_df))
    print("accept rate:", len(accepted) / len(df) if len(df) else np.nan)
    print("initial_position_per_symbol:", args.initial_position_per_symbol)
    print("max_position_per_symbol:", args.max_position_per_symbol)
    print("tplus1:", args.tplus1)

    if len(accepted):
        print("\naccepted side counts:")
        print(accepted["side"].value_counts())

    if len(skipped_df):
        print("\nskipped reason counts:")
        print(skipped_df["reason"].value_counts())

    ending = pd.DataFrame({
        "securityid": list(pos.keys()),
        "ending_position": [pos[s] for s in pos.keys()],
        "ending_sellable": [sellable[s] for s in pos.keys()],
    })

    ending_path = args.output.replace(".csv", "_ending_positions.csv")
    ending.to_csv(ending_path, index=False)
    print("saved:", args.output)
    print("saved:", args.skipped_output)
    print("saved:", ending_path)


if __name__ == "__main__":
    main()
