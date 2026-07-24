import glob
import os
import re
import pandas as pd

rows = []

paths = sorted(glob.glob(
    "outputs/portfolio/portfolio_replay_queue_0p05_*_inventory5000_cap*_tplus1_mlp2_h60_202410_100_summary.csv"
))

for path in paths:
    df = pd.read_csv(path)
    row = df.iloc[0].to_dict()
    fname = os.path.basename(path)

    m = re.search(r"queue_0p05_(.*)_inventory5000_cap(\d+)_tplus1", fname)
    if m:
        row["policy"] = m.group(1)
        row["initial_inventory"] = 5000
        row["max_position_cap"] = int(m.group(2))
    else:
        row["policy"] = "unknown"
        row["initial_inventory"] = 5000
        row["max_position_cap"] = -1

    rows.append(row)

out = pd.DataFrame(rows)

cols = [
    "policy",
    "initial_inventory",
    "max_position_cap",
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
    ["policy", "max_position_cap"]
)

pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 260)

print("\n===== inventory cap sensitivity =====")
print(out.to_string(index=False))

out.to_csv("outputs/portfolio/inventory_cap_sensitivity_b5_summary.csv", index=False)

print("\n===== top by total_pnl =====")
print(out.sort_values("total_pnl", ascending=False).head(20).to_string(index=False))

print("\n===== top by return_on_max_gross_exposure =====")
print(out.sort_values("return_on_max_gross_exposure", ascending=False).head(20).to_string(index=False))

print("\nsaved: outputs/portfolio/inventory_cap_sensitivity_b5_summary.csv")
