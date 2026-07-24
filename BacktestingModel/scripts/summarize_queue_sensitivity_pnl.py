import glob
import os
import pandas as pd

TAG = "mlp2_h60_202410_100"

rows = []

for path in sorted(glob.glob(f"outputs/metrics/summary_queue_mult_*_sample_{TAG}.csv")):
    name = os.path.basename(path)
    mult = name.split("summary_queue_mult_")[1].split("_sample")[0].replace("p", ".")

    df = pd.read_csv(path)
    if len(df) == 0:
        continue

    row = df.iloc[0].to_dict()
    row["queue_multiplier"] = mult
    rows.append(row)

out = pd.DataFrame(rows)

if len(out) == 0:
    print("No summary files found.")
else:
    out["queue_multiplier_float"] = out["queue_multiplier"].astype(float)
    out = out.sort_values("queue_multiplier_float")

    cols = [
        "queue_multiplier",
        "num_trades",
        "num_buy",
        "num_sell",
        "total_notional",
        "total_fee",
        "total_gross_pnl",
        "total_net_pnl",
        "avg_net_pnl",
        "win_rate",
        "avg_net_pnl_bps",
        "buy_total_net_pnl",
        "sell_total_net_pnl",
    ]

    cols = [c for c in cols if c in out.columns]
    out = out[cols]

    print(out.to_string(index=False))

    out.to_csv("outputs/metrics/queue_sensitivity_pnl_sample_summary.csv", index=False)
    print("saved: outputs/metrics/queue_sensitivity_pnl_sample_summary.csv")
