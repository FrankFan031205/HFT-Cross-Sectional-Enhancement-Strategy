import argparse
import numpy as np
import pandas as pd
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--threshold-bps", type=float, required=True)
    ap.add_argument("--start-date", type=int, default=None)
    ap.add_argument("--end-date", type=int, default=None)
    args = ap.parse_args()

    inp = Path(args.input)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    print("reading:", inp)
    df = pd.read_csv(inp, low_memory=False)

    df["execution_date"] = df["execution_date"].astype(int)
    df["execution_datetime"] = pd.to_datetime(df["execution_datetime"])
    df["securityid"] = df["securityid"].astype(int)

    if args.start_date is not None:
        df = df[df["execution_date"] >= args.start_date].copy()
    if args.end_date is not None:
        df = df[df["execution_date"] <= args.end_date].copy()

    df = df.sort_values(["securityid", "execution_datetime"]).reset_index(drop=True)

    for c in ["effective_target_weight", "buy_edge_bps", "sell_edge_bps"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "effective_target_weight" not in df.columns:
        raise ValueError("missing effective_target_weight")

    df["effective_target_weight_raw"] = df["effective_target_weight"]
    df["cost_aware_threshold_bps"] = args.threshold_bps
    df["cost_aware_blocked"] = False
    df["cost_aware_edge_bps"] = np.nan

    states = df["state"].astype(str).fillna("").values if "state" in df.columns else np.array([""] * len(df))
    sides = df["side"].astype(str).fillna("").values if "side" in df.columns else np.array([""] * len(df))

    raw_w = df["effective_target_weight_raw"].fillna(0.0).values.astype(float)
    buy_edge = df["buy_edge_bps"].values.astype(float) if "buy_edge_bps" in df.columns else np.full(len(df), np.nan)
    sell_edge = df["sell_edge_bps"].values.astype(float) if "sell_edge_bps" in df.columns else np.full(len(df), np.nan)

    filtered = raw_w.copy()
    blocked = np.zeros(len(df), dtype=bool)
    used_edge = np.full(len(df), np.nan)

    # stateful per stock:
    # ENTRY below threshold -> target 0
    # ADD below threshold -> keep previous filtered target, not increase
    # HOLD after blocked entry -> keep 0 until a valid ENTRY/ADD appears
    for sid, idx in df.groupby("securityid").indices.items():
        idx = np.array(idx, dtype=np.int64)
        prev = 0.0

        for j in idx:
            raw = raw_w[j]
            st = states[j]
            side = sides[j]

            if raw > 0:
                edge = buy_edge[j]
            elif raw < 0:
                edge = sell_edge[j]
            elif side == "BUY":
                edge = buy_edge[j]
            elif side == "SELL":
                edge = sell_edge[j]
            else:
                edge = np.nan

            used_edge[j] = edge

            is_entry = st == "ENTRY"
            is_add = st == "ADD"
            is_hold = st == "HOLD"
            is_exit = st == "EXIT"
            is_flat = st == "FLAT"

            edge_ok = np.isfinite(edge) and (edge >= args.threshold_bps)

            if is_entry:
                if edge_ok:
                    new = raw
                else:
                    new = 0.0
                    blocked[j] = abs(raw) > 1e-12

            elif is_add:
                if edge_ok:
                    new = raw
                else:
                    # do not add; keep previous filtered target
                    new = prev
                    blocked[j] = abs(raw - prev) > 1e-12

            elif is_hold:
                # if we already have a filtered position, allow hold;
                # otherwise don't let HOLD resurrect a blocked entry
                if abs(prev) > 1e-12:
                    new = raw
                else:
                    new = 0.0
                    blocked[j] = abs(raw) > 1e-12

            elif is_exit or is_flat:
                new = 0.0

            else:
                # fallback: if trying to create new target, require edge
                if abs(prev) <= 1e-12 and abs(raw) > 1e-12:
                    if edge_ok:
                        new = raw
                    else:
                        new = 0.0
                        blocked[j] = True
                else:
                    new = raw

            filtered[j] = new
            prev = new

    df["effective_target_weight"] = filtered
    df["cost_aware_blocked"] = blocked
    df["cost_aware_edge_bps"] = used_edge

    # keep columns consistent for downstream scripts
    if "target_weight" in df.columns:
        df["target_weight"] = df["effective_target_weight"]
    if "desired_weight" in df.columns:
        df["desired_weight"] = df["effective_target_weight"]

    df = df.sort_values(["execution_datetime", "securityid"]).reset_index(drop=True)

    print("rows:", len(df))
    print("blocked rows:", int(df["cost_aware_blocked"].sum()))
    print("nonzero before:", int((df["effective_target_weight_raw"].abs() > 1e-12).sum()))
    print("nonzero after:", int((df["effective_target_weight"].abs() > 1e-12).sum()))

    gross_before = df.groupby("execution_datetime")["effective_target_weight_raw"].apply(lambda x: x.abs().sum())
    gross_after = df.groupby("execution_datetime")["effective_target_weight"].apply(lambda x: x.abs().sum())

    print("\ngross before:")
    print(gross_before.describe())
    print("\ngross after:")
    print(gross_after.describe())

    df.to_csv(out, index=False)
    print("saved:", out)


if __name__ == "__main__":
    main()
