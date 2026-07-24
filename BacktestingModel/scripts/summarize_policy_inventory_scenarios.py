import glob
import os
import re
import pandas as pd

rows = []

for path in sorted(glob.glob("outputs/portfolio/portfolio_replay_queue_0p05_*_inventory*_tplus1_mlp2_h60_202410_100_summary.csv")):
    df = pd.read_csv(path)
    row = df.iloc[0].to_dict()
    fname = os.path.basename(path)

    m = re.search(r"queue_0p05_(.*)_inventory(\d+)_tplus1", fname)
    if m:
        row["policy"] = m.group(1)
        row["initial_inventory"] = int(m.group(2))
    else:
        row["policy"] = "unknown"
        row["initial_inventory"] = -1

    rows.append(row)

out = pd.DataFrame(rows)

cols = [
    "policy",
    "initial_inventory",
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
out = out[cols].sort_values(
    ["initial_inventory", "total_pnl"],
    ascending=[True, False]
)

pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 260)

print(out.to_string(index=False))

out.to_csv("outputs/portfolio/policy_inventory_scenarios_b5_summary.csv", index=False)
print("saved: outputs/portfolio/policy_inventory_scenarios_b5_summary.csv")

print("\n===== top by total_pnl =====")
print(out.sort_values("total_pnl", ascending=False).head(20).to_string(index=False))

print("\n===== top by return_on_max_gross_exposure =====")
print(out.sort_values("return_on_max_gross_exposure", ascending=False).head(20).to_string(index=False))
