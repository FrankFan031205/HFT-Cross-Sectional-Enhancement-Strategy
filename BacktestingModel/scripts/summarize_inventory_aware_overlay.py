import glob
import os
import re
import pandas as pd

rows = []

paths = sorted(glob.glob(
    "outputs/portfolio/portfolio_replay_queue_0p05_*_inventory5000_cap*_tplus1_mlp2_h60_202410_100_summary.csv"
))

for path in paths:
    fname = os.path.basename(path)

    if "_hardonly_tplus1_" not in fname and "_loose_tplus1_" not in fname and "_medium_tplus1_" not in fname and "_tight_tplus1_" not in fname and "_medium_cap20000_tplus1_" not in fname:
        continue

    df = pd.read_csv(path)
    row = df.iloc[0].to_dict()

    m = re.search(r"queue_0p05_(.*)_inventory(\d+)_cap(\d+)_(.*)_tplus1", fname)

    if m:
        row["policy"] = m.group(1)
        row["initial_inventory"] = int(m.group(2))
        row["max_position_cap"] = int(m.group(3))
        row["band"] = m.group(4)
    else:
        row["policy"] = "unknown"
        row["initial_inventory"] = -1
        row["max_position_cap"] = -1
        row["band"] = "unknown"

    skip_path = (
        "outputs/trades/skipped_queue_mult_0p05_"
        f"{row['policy']}_inventory{row['initial_inventory']}_cap{row['max_position_cap']}_{row['band']}_tplus1_mlp2_h60_202410_100.csv"
    )

    if os.path.exists(skip_path):
        skip = pd.read_csv(skip_path, low_memory=False)
        row["num_skipped"] = len(skip)
        if len(skip):
            counts = skip["reason"].value_counts().to_dict()
            for k, v in counts.items():
                row[f"skip_{k}"] = v
    else:
        row["num_skipped"] = None

    rows.append(row)

out = pd.DataFrame(rows)

cols = [
    "policy",
    "band",
    "initial_inventory",
    "max_position_cap",
    "num_trades",
    "num_skipped",
    "skip_inventory_high_block_buy",
    "skip_inventory_low_block_sell",
    "skip_hard_max_position",
    "skip_insufficient_sellable_inventory",
    "total_pnl",
    "total_turnover",
    "pnl_bps_on_turnover",
    "final_cash",
    "final_inventory_value",
    "final_equity",
    "max_gross_exposure",
    "return_on_max_gross_exposure",
    "max_abs_net_exposure",
    "max_drawdown",
    "num_long_symbols_end",
    "num_short_symbols_end",
    "num_short_events",
    "num_short_violations_if_no_short",
]

cols = [c for c in cols if c in out.columns]
out = out[cols].sort_values(
    ["policy", "max_position_cap", "band"]
)

pd.set_option("display.max_columns", 120)
pd.set_option("display.width", 300)

print("\n===== inventory-aware overlay summary =====")
print(out.to_string(index=False))

out.to_csv("outputs/portfolio/inventory_aware_overlay_b5_summary.csv", index=False)

print("\n===== top by total_pnl =====")
print(out.sort_values("total_pnl", ascending=False).head(20).to_string(index=False))

print("\n===== top by return_on_max_gross_exposure =====")
print(out.sort_values("return_on_max_gross_exposure", ascending=False).head(20).to_string(index=False))

print("\n===== top by lowest max_drawdown among positive pnl =====")
pos = out[out["total_pnl"] > 0].copy()
print(pos.sort_values("max_drawdown", ascending=False).head(20).to_string(index=False))

print("\nsaved: outputs/portfolio/inventory_aware_overlay_b5_summary.csv")
