import os
import numpy as np
import pandas as pd

pd.set_option("display.max_columns", 120)
pd.set_option("display.width", 260)

TAG = "mlp2_h60_202410_100"

PATHS = {
    "touched": f"outputs/trades/trades_pnl_touched_{TAG}.csv",
    "queue_0p05": f"outputs/trades/trades_pnl_queue_mult_0p05_{TAG}.csv",
}

OUT = f"outputs/metrics/analysis/filter_candidates_{TAG}.csv"
os.makedirs(os.path.dirname(OUT), exist_ok=True)


def prep(df):
    df = df.copy()
    df["decision_time"] = pd.to_datetime(df["decision_time"])
    df["date"] = df["decision_time"].dt.strftime("%Y%m%d")
    df["securityid"] = df["securityid"].astype(str).str.replace(".0", "", regex=False).str.zfill(6)
    return df


def weighted_bps(x):
    n = x["notional"].sum()
    return x["net_pnl"].sum() / n * 10000 if n else np.nan


def summarize(df, model, rule):
    if len(df) == 0:
        return {
            "model": model,
            "rule": rule,
            "num_trades": 0,
        }

    daily = df.groupby("date")["net_pnl"].sum()
    total_notional = df["notional"].sum()
    total_net = df["net_pnl"].sum()

    out = {
        "model": model,
        "rule": rule,
        "num_trades": len(df),
        "num_days": df["date"].nunique(),
        "num_securities": df["securityid"].nunique(),
        "buy_trades": int((df["side"] == "BUY").sum()),
        "sell_trades": int((df["side"] == "SELL").sum()),
        "total_notional": total_notional,
        "total_fee": df["fee"].sum(),
        "total_gross_pnl": df["gross_pnl"].sum(),
        "total_net_pnl": total_net,
        "net_return_bps_weighted": total_net / total_notional * 10000 if total_notional else np.nan,
        "avg_net_pnl": df["net_pnl"].mean(),
        "median_net_pnl": df["net_pnl"].median(),
        "avg_net_pnl_bps": df["net_pnl_bps"].mean(),
        "win_rate": df["win"].mean(),
        "positive_day_ratio": (daily > 0).mean(),
        "min_daily_pnl": daily.min(),
        "max_daily_pnl": daily.max(),
        "buy_net_pnl": df.loc[df["side"] == "BUY", "net_pnl"].sum(),
        "sell_net_pnl": df.loc[df["side"] == "SELL", "net_pnl"].sum(),
    }
    return out


def make_candidates(df):
    c = {}

    c["all"] = df
    c["buy_only"] = df[df["side"] == "BUY"]
    c["sell_only"] = df[df["side"] == "SELL"]

    if "risk_state" in df.columns:
        c["risk_strong_alpha"] = df[df["risk_state"] == "strong_alpha"]
        c["risk_high_volatility"] = df[df["risk_state"] == "high_volatility"]
        c["risk_alpha_clipped"] = df[df["risk_state"] == "alpha_clipped"]
        c["risk_strong_or_highvol_or_clipped"] = df[df["risk_state"].isin(["strong_alpha", "high_volatility", "alpha_clipped"])]
        c["risk_exclude_normal_weak"] = df[~df["risk_state"].isin(["normal", "weak_alpha"])]

    signal_col = "hidden_factor_mlp2_h60"
    if signal_col in df.columns:
        s = df[signal_col]

        abs_s = s.abs()
        q50 = abs_s.quantile(0.50)
        q70 = abs_s.quantile(0.70)
        q80 = abs_s.quantile(0.80)
        q90 = abs_s.quantile(0.90)

        c["abs_signal_top50"] = df[abs_s >= q50]
        c["abs_signal_top30"] = df[abs_s >= q70]
        c["abs_signal_top20"] = df[abs_s >= q80]
        c["abs_signal_top10"] = df[abs_s >= q90]

        q_buy_60 = s.quantile(0.60)
        q_buy_70 = s.quantile(0.70)
        q_buy_80 = s.quantile(0.80)
        q_sell_40 = s.quantile(0.40)
        q_sell_30 = s.quantile(0.30)
        q_sell_20 = s.quantile(0.20)

        c["directional_q60_40"] = df[((df["side"] == "BUY") & (s >= q_buy_60)) | ((df["side"] == "SELL") & (s <= q_sell_40))]
        c["directional_q70_30"] = df[((df["side"] == "BUY") & (s >= q_buy_70)) | ((df["side"] == "SELL") & (s <= q_sell_30))]
        c["directional_q80_20"] = df[((df["side"] == "BUY") & (s >= q_buy_80)) | ((df["side"] == "SELL") & (s <= q_sell_20))]

        c["buy_positive_sell_negative"] = df[((df["side"] == "BUY") & (s > 0)) | ((df["side"] == "SELL") & (s < 0))]
        c["buy_positive_only"] = df[(df["side"] == "BUY") & (s > 0)]
        c["sell_negative_only"] = df[(df["side"] == "SELL") & (s < 0)]

    # concentration diagnostics: remove top/bottom contributors based on this model's own PnL
    sym = df.groupby("securityid")["net_pnl"].sum().sort_values(ascending=False)
    top10 = set(sym.head(10).index)
    bottom10 = set(sym.tail(10).index)

    c["exclude_top10_symbols"] = df[~df["securityid"].isin(top10)]
    c["exclude_bottom10_symbols"] = df[~df["securityid"].isin(bottom10)]
    c["exclude_top_bottom10_symbols"] = df[~df["securityid"].isin(top10 | bottom10)]

    return c


def main():
    rows = []

    for model, path in PATHS.items():
        print("loading", model, path)
        df = prep(pd.read_csv(path, low_memory=False))

        candidates = make_candidates(df)

        for rule, sub in candidates.items():
            rows.append(summarize(sub, model, rule))

    out = pd.DataFrame(rows)

    preferred_cols = [
        "model", "rule", "num_trades", "num_days", "num_securities",
        "buy_trades", "sell_trades",
        "total_notional", "total_fee", "total_gross_pnl", "total_net_pnl",
        "net_return_bps_weighted", "avg_net_pnl", "median_net_pnl",
        "avg_net_pnl_bps", "win_rate", "positive_day_ratio",
        "min_daily_pnl", "max_daily_pnl", "buy_net_pnl", "sell_net_pnl"
    ]
    out = out[[c for c in preferred_cols if c in out.columns]]

    out = out.sort_values(["model", "net_return_bps_weighted"], ascending=[True, False])

    print("\n===== filter candidates =====")
    print(out.to_string(index=False))

    out.to_csv(OUT, index=False)
    print("\nsaved:", OUT)


if __name__ == "__main__":
    main()
