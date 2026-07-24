import glob
import os
import pandas as pd

rows = []

for path in sorted(glob.glob("outputs/portfolio/portfolio_replay_queue_0p05_inventory*_tplus1_mlp2_h60_202410_100_summary.csv")):
    df = pd.read_csv(path)
    row = df.iloc[0].to_dict()
    row["file"] = os.path.basename(path)
    rows.append(row)

out = pd.DataFrame(rows)

cols = [
    "model",
    "num_trades",
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

pd.set_option("display.max_columns", 80)
pd.set_option("display.width", 220)

print(out[cols].to_string(index=False))

out[cols].to_csv("outputs/portfolio/inventory_scenarios_b5_summary.csv", index=False)
print("saved: outputs/portfolio/inventory_scenarios_b5_summary.csv")
