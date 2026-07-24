import os
import yaml
import subprocess
import pandas as pd

CONFIG = "config/final_inventory_aware_backtest.yaml"

with open(CONFIG, "r") as f:
    cfg = yaml.safe_load(f)

base_path = cfg["input"]["base_trades_path"]
signal_col = cfg["input"]["signal_col"]

policy_path = cfg["output"]["final_policy_trades_path"]
inventory_path = cfg["output"]["final_inventory_trades_path"]
skipped_path = cfg["output"]["final_skipped_path"]
portfolio_prefix = cfg["output"]["final_portfolio_prefix"]

os.makedirs("outputs/trades", exist_ok=True)
os.makedirs("outputs/portfolio", exist_ok=True)

print("===== final inventory-aware backtest =====")
print("config:", CONFIG)
print("base trades:", base_path)
print("policy:", cfg["policy"]["name"])
print("inventory:", cfg["inventory"])

print("\n[1] create abs_top40 policy trades")
df = pd.read_csv(base_path, low_memory=False)

s = df[signal_col]
abs_s = s.abs()
threshold = abs_s.quantile(cfg["policy"]["quantile"])

sub = df[abs_s >= threshold].copy()
sub.to_csv(policy_path, index=False)

print("signal threshold:", threshold)
print("base trades:", len(df))
print("policy trades:", len(sub))
print("saved:", policy_path)

print("\n[2] apply inventory-aware overlay")
inv = cfg["inventory"]

cmd = [
    "python", "scripts/filter_inventory_aware_overlay.py",
    "--trades", policy_path,
    "--output", inventory_path,
    "--skipped-output", skipped_path,
    "--initial-position-per-symbol", str(inv["initial_position_per_symbol"]),
    "--max-position-per-symbol", str(inv["max_position_per_symbol"]),
    "--sell-floor-position", str(inv["sell_floor_position"]),
    "--buy-block-position", str(inv["buy_block_position"]),
    "--tplus1", "1" if inv["tplus1"] else "0",
]
subprocess.run(cmd, check=True)

print("\n[3] run portfolio replay")
cmd = [
    "python", "scripts/run_portfolio_replay.py",
    "--trades", inventory_path,
    "--model", "final_queue005_abs_top40_inventory5000_cap15000_tight_tplus1",
    "--out-prefix", portfolio_prefix,
    "--capital", "0",
    "--initial-position-per-symbol", str(inv["initial_position_per_symbol"]),
    "--allow-short", "0",
    "--record-every", "10000",
]
subprocess.run(cmd, check=True)

print("\n[4] final summary")
summary_path = portfolio_prefix + "_summary.csv"
daily_path = portfolio_prefix + "_daily.csv"

summary = pd.read_csv(summary_path)
daily = pd.read_csv(daily_path)

pd.set_option("display.max_columns", 120)
pd.set_option("display.width", 260)

cols = [
    "model",
    "num_trades",
    "num_securities",
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
cols = [c for c in cols if c in summary.columns]

print("\n===== final portfolio summary =====")
print(summary[cols].to_string(index=False))

print("\n===== final daily pnl =====")
daily_cols = [
    "model",
    "date",
    "num_trades",
    "turnover",
    "fee",
    "daily_pnl",
    "daily_return_on_turnover_bps",
    "max_gross_exposure",
    "max_abs_net_exposure",
    "max_drawdown",
    "num_long_symbols",
    "num_short_symbols",
]
daily_cols = [c for c in daily_cols if c in daily.columns]
print(daily[daily_cols].to_string(index=False))

print("\nsaved:")
print(policy_path)
print(inventory_path)
print(skipped_path)
print(summary_path)
print(daily_path)
