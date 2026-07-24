import glob
import os
import pandas as pd

rows = []

for path in sorted(glob.glob("outputs/portfolio/portfolio_replay_*_summary.csv")):
    try:
        df = pd.read_csv(path)
        if len(df) == 0:
            continue
        row = df.iloc[0].to_dict()
        row["summary_file"] = os.path.basename(path)
        rows.append(row)
    except Exception as e:
        print("failed:", path, e)

if not rows:
    print("No portfolio summary files found.")
    raise SystemExit

out = pd.DataFrame(rows)

cols = [
    "model",
    "num_trades",
    "num_securities",
    "total_pnl",
    "total_turnover",
    "pnl_bps_on_turnover",
    "max_gross_exposure",
    "return_on_max_gross_exposure",
    "max_abs_net_exposure",
    "max_drawdown",
    "num_long_symbols_end",
    "num_short_symbols_end",
    "num_short_events",
    "num_short_violations_if_no_short",
    "summary_file",
]

cols = [c for c in cols if c in out.columns]
out = out[cols].copy()

if "total_pnl" in out.columns:
    out = out.sort_values("total_pnl", ascending=False)

pd.set_option("display.max_columns", 120)
pd.set_option("display.width", 260)

print(out.to_string(index=False))

os.makedirs("outputs/portfolio", exist_ok=True)
out.to_csv("outputs/portfolio/all_experiments_summary.csv", index=False)

print("\nsaved: outputs/portfolio/all_experiments_summary.csv")
