import glob
import os
import pandas as pd

TAG = "mlp2_h60_202410_100"
QUOTE_PATH = "../MarketMakingModel/outputs/quote_decisions/quote_decisions_mlp2_h60_202410_100.csv"

def as_bool(s):
    return s.astype(str).str.lower().isin(["1", "true", "t", "yes", "y"])

q = pd.read_csv(QUOTE_PATH, nrows=5000, low_memory=False)
total_orders = int(as_bool(q["quote_bid"]).sum() + as_bool(q["quote_ask"]).sum())

rows = []

paths = sorted(glob.glob(f"outputs/fills/fills_queue_mult_*_sample_{TAG}.csv"))

print("found files:")
for p in paths:
    print(" ", p)

for path in paths:
    name = os.path.basename(path)
    mult = name.split("fills_queue_mult_")[1].split("_sample")[0].replace("p", ".")

    df = pd.read_csv(path, low_memory=False)

    rows.append({
        "queue_multiplier": mult,
        "total_orders": total_orders,
        "fills": len(df),
        "fill_rate": len(df) / total_orders if total_orders else 0.0,
        "buy_fills": int((df["side"] == "BUY").sum()) if len(df) else 0,
        "sell_fills": int((df["side"] == "SELL").sum()) if len(df) else 0,
        "avg_queue_ahead_initial": df["queue_ahead_initial"].mean() if len(df) else None,
        "median_queue_ahead_initial": df["queue_ahead_initial"].median() if len(df) else None,
        "avg_fill_qty": df["fill_qty"].mean() if len(df) else None,
    })

out = pd.DataFrame(rows)

if len(out) == 0:
    print("No queue sensitivity fill files found.")
else:
    out["queue_multiplier_float"] = out["queue_multiplier"].astype(float)
    out = out.sort_values("queue_multiplier_float").drop(columns=["queue_multiplier_float"])

    print("\n===== queue sensitivity summary =====")
    print(out.to_string(index=False))

    out.to_csv("outputs/metrics/queue_sensitivity_sample_summary.csv", index=False)
    print("\nsaved: outputs/metrics/queue_sensitivity_sample_summary.csv")
