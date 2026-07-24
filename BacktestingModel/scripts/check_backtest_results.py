import os
import pandas as pd

pd.set_option("display.max_columns", 80)
pd.set_option("display.width", 240)

files = {
    "touched_fills": "outputs/fills/fills_touched_mlp2_h60_202410_100.csv",
    "touched_enriched": "outputs/fills/fills_touched_enriched_mlp2_h60_202410_100.csv",
    "touched_trades": "outputs/trades/trades_pnl_touched_mlp2_h60_202410_100.csv",
    "touched_summary": "outputs/metrics/summary_touched_mlp2_h60_202410_100.csv",
    "touched_daily": "outputs/metrics/daily_pnl_touched_mlp2_h60_202410_100.csv",
    "touched_factor": "outputs/metrics/factor_pnl_touched_mlp2_h60_202410_100.csv",
    "queue005_fills": "outputs/fills/fills_queue_mult_0p05_mlp2_h60_202410_100.csv",
    "queue005_trades": "outputs/trades/trades_pnl_queue_mult_0p05_mlp2_h60_202410_100.csv",
    "queue005_summary": "outputs/metrics/summary_queue_mult_0p05_mlp2_h60_202410_100.csv",
    "queue005_factor": "outputs/metrics/factor_pnl_queue_mult_0p05_mlp2_h60_202410_100.csv",
}

print("===== file existence =====")
for k, f in files.items():
    print(f"{k:18s}", os.path.exists(f), f)

def inspect_trades(path, name):
    if not os.path.exists(path):
        print(f"\n{name}: missing")
        return

    df = pd.read_csv(path, low_memory=False)
    print(f"\n===== {name} =====")
    print("shape:", df.shape)

    if len(df) == 0:
        return

    if "decision_time" in df.columns:
        df["decision_time"] = pd.to_datetime(df["decision_time"])
        print("date range:", df["decision_time"].min(), "->", df["decision_time"].max())
        print("num days:", df["decision_time"].dt.strftime("%Y%m%d").nunique())

    if "securityid" in df.columns:
        print("num securities:", df["securityid"].nunique())

    if "side" in df.columns:
        print("side counts:")
        print(df["side"].value_counts())

    if {"total_notional", "total_net_pnl"}.issubset(df.columns):
        pass

    if {"notional", "net_pnl"}.issubset(df.columns):
        total_notional = df["notional"].sum()
        total_net = df["net_pnl"].sum()
        print("total_notional:", total_notional)
        print("total_net_pnl:", total_net)
        print("return_bps_on_notional:", total_net / total_notional * 10000 if total_notional else None)
        print("win_rate:", df["win"].mean() if "win" in df.columns else None)

inspect_trades(files["touched_fills"], "touched fills")
inspect_trades(files["touched_trades"], "touched trades pnl")
inspect_trades(files["queue005_fills"], "queue 0.05 fills")
inspect_trades(files["queue005_trades"], "queue 0.05 trades pnl")

for key in ["touched_summary", "queue005_summary"]:
    path = files[key]
    if os.path.exists(path):
        print(f"\n===== {key} =====")
        df = pd.read_csv(path)
        print(df.T.to_string())

for key in ["touched_daily"]:
    path = files[key]
    if os.path.exists(path):
        print(f"\n===== {key} =====")
        df = pd.read_csv(path)
        print(df.to_string(index=False))

for key in ["touched_factor", "queue005_factor"]:
    path = files[key]
    if os.path.exists(path):
        print(f"\n===== {key} =====")
        df = pd.read_csv(path)
        print(df.to_string(index=False))
