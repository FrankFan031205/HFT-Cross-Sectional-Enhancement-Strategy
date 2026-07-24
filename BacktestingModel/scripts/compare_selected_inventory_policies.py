import pandas as pd

cases = {
    "all_inv5000": "outputs/portfolio/portfolio_replay_queue_0p05_all_inventory5000_tplus1_mlp2_h60_202410_100_daily.csv",
    "all_inv10000": "outputs/portfolio/portfolio_replay_queue_0p05_all_inventory10000_tplus1_mlp2_h60_202410_100_daily.csv",
    "abs50_inv5000": "outputs/portfolio/portfolio_replay_queue_0p05_abs_top50_inventory5000_tplus1_mlp2_h60_202410_100_daily.csv",
    "abs40_inv5000": "outputs/portfolio/portfolio_replay_queue_0p05_abs_top40_inventory5000_tplus1_mlp2_h60_202410_100_daily.csv",
    "abs30_buy_inv5000": "outputs/portfolio/portfolio_replay_queue_0p05_abs_top30_buy_only_inventory5000_tplus1_mlp2_h60_202410_100_daily.csv",
}

rows = []

for name, path in cases.items():
    df = pd.read_csv(path)
    df["case"] = name
    rows.append(df)

out = pd.concat(rows, ignore_index=True)

cols = [
    "case", "date", "num_trades", "turnover", "daily_pnl",
    "daily_return_on_turnover_bps",
    "max_gross_exposure", "max_abs_net_exposure",
    "max_drawdown", "num_long_symbols", "num_short_symbols"
]

out = out[cols]

pd.set_option("display.max_columns", 80)
pd.set_option("display.width", 240)

print("\n===== selected daily comparison =====")
print(out.to_string(index=False))

summary = out.groupby("case").agg(
    days=("date", "count"),
    positive_days=("daily_pnl", lambda x: (x > 0).sum()),
    total_pnl=("daily_pnl", "sum"),
    avg_daily_pnl=("daily_pnl", "mean"),
    min_daily_pnl=("daily_pnl", "min"),
    max_daily_pnl=("daily_pnl", "max"),
    total_turnover=("turnover", "sum"),
    avg_daily_return_bps=("daily_return_on_turnover_bps", "mean"),
    max_gross_exposure=("max_gross_exposure", "max"),
    worst_drawdown=("max_drawdown", "min"),
).reset_index()

summary["pnl_bps_on_turnover"] = summary["total_pnl"] / summary["total_turnover"] * 10000
summary["positive_day_ratio"] = summary["positive_days"] / summary["days"]

print("\n===== selected summary =====")
print(summary.to_string(index=False))

out.to_csv("outputs/portfolio/selected_inventory_policy_daily_comparison.csv", index=False)
summary.to_csv("outputs/portfolio/selected_inventory_policy_summary.csv", index=False)

print("\nsaved:")
print("outputs/portfolio/selected_inventory_policy_daily_comparison.csv")
print("outputs/portfolio/selected_inventory_policy_summary.csv")
