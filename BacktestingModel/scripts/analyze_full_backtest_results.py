import os
import pandas as pd
import numpy as np

pd.set_option("display.max_columns", 120)
pd.set_option("display.width", 260)

TAG = "mlp2_h60_202410_100"

PATHS = {
    "touched_fills": f"outputs/fills/fills_touched_{TAG}.csv",
    "touched_trades": f"outputs/trades/trades_pnl_touched_{TAG}.csv",
    "touched_factor": f"outputs/metrics/factor_pnl_touched_{TAG}.csv",
    "queue_fills": f"outputs/fills/fills_queue_mult_0p05_{TAG}.csv",
    "queue_trades": f"outputs/trades/trades_pnl_queue_mult_0p05_{TAG}.csv",
    "queue_factor": f"outputs/metrics/factor_pnl_queue_mult_0p05_{TAG}.csv",
}

OUT_DIR = "outputs/metrics/analysis"
os.makedirs(OUT_DIR, exist_ok=True)


def load(path):
    if not os.path.exists(path):
        print("missing:", path)
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def prep_trades(df):
    if len(df) == 0:
        return df
    df = df.copy()
    df["decision_time"] = pd.to_datetime(df["decision_time"])
    df["date"] = df["decision_time"].dt.strftime("%Y%m%d")
    df["securityid"] = df["securityid"].astype(str).str.replace(".0", "", regex=False).str.zfill(6)
    return df


def weighted_bps(net_pnl, notional):
    return net_pnl / notional * 10000 if notional != 0 else np.nan


def overall(trades, fills, name, touched_fills_n=None):
    if len(trades) == 0:
        return {
            "model": name,
            "num_fills": len(fills),
            "num_trades_with_pnl": 0,
        }

    total_notional = trades["notional"].sum()
    total_fee = trades["fee"].sum()
    gross = trades["gross_pnl"].sum()
    net = trades["net_pnl"].sum()

    out = {
        "model": name,
        "num_fills": len(fills),
        "num_trades_with_pnl": len(trades),
        "num_securities": trades["securityid"].nunique(),
        "num_days": trades["date"].nunique(),
        "buy_trades": int((trades["side"] == "BUY").sum()),
        "sell_trades": int((trades["side"] == "SELL").sum()),
        "total_notional": total_notional,
        "total_fee": total_fee,
        "total_gross_pnl": gross,
        "total_net_pnl": net,
        "gross_return_bps_weighted": weighted_bps(gross, total_notional),
        "net_return_bps_weighted": weighted_bps(net, total_notional),
        "avg_net_pnl": trades["net_pnl"].mean(),
        "median_net_pnl": trades["net_pnl"].median(),
        "avg_net_pnl_bps": trades["net_pnl_bps"].mean(),
        "win_rate": trades["win"].mean(),
        "fee_bps_weighted": weighted_bps(total_fee, total_notional),
    }

    if touched_fills_n:
        out["fills_vs_touched_ratio"] = len(fills) / touched_fills_n

    return out


def by_date(trades, name):
    if len(trades) == 0:
        return pd.DataFrame()

    g = trades.groupby("date").agg(
        num_trades=("net_pnl", "size"),
        num_securities=("securityid", "nunique"),
        total_notional=("notional", "sum"),
        total_fee=("fee", "sum"),
        gross_pnl=("gross_pnl", "sum"),
        net_pnl=("net_pnl", "sum"),
        avg_net_pnl=("net_pnl", "mean"),
        win_rate=("win", "mean"),
        avg_net_pnl_bps=("net_pnl_bps", "mean"),
    ).reset_index()

    g["net_return_bps_weighted"] = g["net_pnl"] / g["total_notional"] * 10000
    g["model"] = name
    return g


def by_side(trades, name):
    if len(trades) == 0:
        return pd.DataFrame()

    g = trades.groupby("side").agg(
        num_trades=("net_pnl", "size"),
        num_securities=("securityid", "nunique"),
        total_notional=("notional", "sum"),
        total_fee=("fee", "sum"),
        gross_pnl=("gross_pnl", "sum"),
        net_pnl=("net_pnl", "sum"),
        avg_net_pnl=("net_pnl", "mean"),
        win_rate=("win", "mean"),
        avg_net_pnl_bps=("net_pnl_bps", "mean"),
    ).reset_index()

    g["net_return_bps_weighted"] = g["net_pnl"] / g["total_notional"] * 10000
    g["model"] = name
    return g


def by_symbol(trades, name):
    if len(trades) == 0:
        return pd.DataFrame()

    g = trades.groupby("securityid").agg(
        num_trades=("net_pnl", "size"),
        total_notional=("notional", "sum"),
        total_fee=("fee", "sum"),
        gross_pnl=("gross_pnl", "sum"),
        net_pnl=("net_pnl", "sum"),
        avg_net_pnl=("net_pnl", "mean"),
        win_rate=("win", "mean"),
        avg_net_pnl_bps=("net_pnl_bps", "mean"),
    ).reset_index()

    g["net_return_bps_weighted"] = g["net_pnl"] / g["total_notional"] * 10000
    g["model"] = name
    return g.sort_values("net_pnl", ascending=False)


def concentration(symbol_df, name):
    if len(symbol_df) == 0:
        return {}

    total = symbol_df["net_pnl"].sum()
    pos = symbol_df[symbol_df["net_pnl"] > 0]["net_pnl"].sum()
    neg = symbol_df[symbol_df["net_pnl"] < 0]["net_pnl"].sum()

    top10 = symbol_df.sort_values("net_pnl", ascending=False).head(10)["net_pnl"].sum()
    bottom10 = symbol_df.sort_values("net_pnl", ascending=True).head(10)["net_pnl"].sum()

    return {
        "model": name,
        "num_symbols": len(symbol_df),
        "positive_symbols": int((symbol_df["net_pnl"] > 0).sum()),
        "negative_symbols": int((symbol_df["net_pnl"] < 0).sum()),
        "total_net_pnl": total,
        "positive_pnl_sum": pos,
        "negative_pnl_sum": neg,
        "top10_symbol_pnl": top10,
        "top10_share_of_total": top10 / total if total != 0 else np.nan,
        "bottom10_symbol_pnl": bottom10,
        "bottom10_share_of_total": bottom10 / total if total != 0 else np.nan,
    }


def by_risk_state(trades, name):
    if len(trades) == 0 or "risk_state" not in trades.columns:
        return pd.DataFrame()

    g = trades.groupby("risk_state").agg(
        num_trades=("net_pnl", "size"),
        total_notional=("notional", "sum"),
        total_fee=("fee", "sum"),
        gross_pnl=("gross_pnl", "sum"),
        net_pnl=("net_pnl", "sum"),
        avg_net_pnl=("net_pnl", "mean"),
        win_rate=("win", "mean"),
        avg_net_pnl_bps=("net_pnl_bps", "mean"),
    ).reset_index()

    g["net_return_bps_weighted"] = g["net_pnl"] / g["total_notional"] * 10000
    g["model"] = name
    return g.sort_values("net_pnl", ascending=False)


def main():
    touched_fills = load(PATHS["touched_fills"])
    touched_trades = prep_trades(load(PATHS["touched_trades"]))
    queue_fills = load(PATHS["queue_fills"])
    queue_trades = prep_trades(load(PATHS["queue_trades"]))

    print("===== loaded =====")
    for k, p in PATHS.items():
        print(k, os.path.exists(p), p)

    touched_fills_n = len(touched_fills)

    overall_df = pd.DataFrame([
        overall(touched_trades, touched_fills, "touched", touched_fills_n),
        overall(queue_trades, queue_fills, "queue_0p05", touched_fills_n),
    ])

    daily_df = pd.concat([
        by_date(touched_trades, "touched"),
        by_date(queue_trades, "queue_0p05"),
    ], ignore_index=True)

    side_df = pd.concat([
        by_side(touched_trades, "touched"),
        by_side(queue_trades, "queue_0p05"),
    ], ignore_index=True)

    touched_symbol = by_symbol(touched_trades, "touched")
    queue_symbol = by_symbol(queue_trades, "queue_0p05")

    conc_df = pd.DataFrame([
        concentration(touched_symbol, "touched"),
        concentration(queue_symbol, "queue_0p05"),
    ])

    risk_df = pd.concat([
        by_risk_state(touched_trades, "touched"),
        by_risk_state(queue_trades, "queue_0p05"),
    ], ignore_index=True)

    touched_symbol.to_csv(f"{OUT_DIR}/symbol_pnl_touched_{TAG}.csv", index=False)
    queue_symbol.to_csv(f"{OUT_DIR}/symbol_pnl_queue_0p05_{TAG}.csv", index=False)
    overall_df.to_csv(f"{OUT_DIR}/overall_comparison_{TAG}.csv", index=False)
    daily_df.to_csv(f"{OUT_DIR}/daily_comparison_{TAG}.csv", index=False)
    side_df.to_csv(f"{OUT_DIR}/side_comparison_{TAG}.csv", index=False)
    conc_df.to_csv(f"{OUT_DIR}/symbol_concentration_{TAG}.csv", index=False)
    risk_df.to_csv(f"{OUT_DIR}/risk_state_pnl_{TAG}.csv", index=False)

    print("\n===== overall comparison =====")
    print(overall_df.to_string(index=False))

    print("\n===== daily comparison =====")
    print(daily_df.to_string(index=False))

    print("\n===== side comparison =====")
    print(side_df.to_string(index=False))

    print("\n===== symbol concentration =====")
    print(conc_df.to_string(index=False))

    if len(risk_df):
        print("\n===== risk_state pnl =====")
        print(risk_df.to_string(index=False))

    print("\n===== top 20 touched symbols =====")
    print(touched_symbol.head(20).to_string(index=False))

    print("\n===== bottom 20 touched symbols =====")
    print(touched_symbol.tail(20).to_string(index=False))

    print("\n===== top 20 queue 0.05 symbols =====")
    print(queue_symbol.head(20).to_string(index=False))

    print("\n===== bottom 20 queue 0.05 symbols =====")
    print(queue_symbol.tail(20).to_string(index=False))

    print("\nsaved analysis tables to:", OUT_DIR)


if __name__ == "__main__":
    main()
