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

OUT = f"outputs/metrics/analysis/combined_filter_candidates_{TAG}.csv"
os.makedirs(os.path.dirname(OUT), exist_ok=True)


def prep(df):
    df = df.copy()
    df["decision_time"] = pd.to_datetime(df["decision_time"])
    df["date"] = df["decision_time"].dt.strftime("%Y%m%d")
    df["securityid"] = df["securityid"].astype(str).str.replace(".0", "", regex=False).str.zfill(6)
    return df


def summarize(df, model, rule):
    if len(df) == 0:
        return {
            "model": model,
            "rule": rule,
            "num_trades": 0,
        }

    daily = df.groupby("date")["net_pnl"].sum()
    sym = df.groupby("securityid")["net_pnl"].sum().sort_values(ascending=False)
    total_net = df["net_pnl"].sum()
    total_notional = df["notional"].sum()
    top10 = sym.head(10).sum()
    bottom10 = sym.tail(10).sum()

    return {
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
        "positive_symbols": int((sym > 0).sum()),
        "negative_symbols": int((sym < 0).sum()),
        "top10_symbol_pnl": top10,
        "top10_share_of_total": top10 / total_net if total_net else np.nan,
        "bottom10_symbol_pnl": bottom10,
        "bottom10_share_of_total": bottom10 / total_net if total_net else np.nan,
    }


def add_rule(rules, name, df):
    rules[name] = df


def make_rules(df):
    signal_col = "hidden_factor_mlp2_h60"
    s = df[signal_col]
    abs_s = s.abs()

    risk_good = df["risk_state"].isin(["strong_alpha", "high_volatility", "alpha_clipped"]) if "risk_state" in df.columns else pd.Series(True, index=df.index)
    risk_not_weak = ~df["risk_state"].isin(["normal", "weak_alpha"]) if "risk_state" in df.columns else pd.Series(True, index=df.index)

    q_abs_50 = abs_s.quantile(0.50)
    q_abs_60 = abs_s.quantile(0.60)
    q_abs_70 = abs_s.quantile(0.70)
    q_abs_80 = abs_s.quantile(0.80)
    q_abs_90 = abs_s.quantile(0.90)

    q_pos_60 = s.quantile(0.60)
    q_pos_70 = s.quantile(0.70)
    q_pos_80 = s.quantile(0.80)
    q_neg_40 = s.quantile(0.40)
    q_neg_30 = s.quantile(0.30)
    q_neg_20 = s.quantile(0.20)

    rules = {}

    add_rule(rules, "all", df)

    for label, q in [
        ("abs_top50", q_abs_50),
        ("abs_top40", q_abs_60),
        ("abs_top30", q_abs_70),
        ("abs_top20", q_abs_80),
        ("abs_top10", q_abs_90),
    ]:
        m = abs_s >= q
        add_rule(rules, label, df[m])
        add_rule(rules, label + "_risk_good", df[m & risk_good])
        add_rule(rules, label + "_exclude_normal_weak", df[m & risk_not_weak])
        add_rule(rules, label + "_buy_only", df[m & (df["side"] == "BUY")])
        add_rule(rules, label + "_sell_only", df[m & (df["side"] == "SELL")])

    directional_60_40 = ((df["side"] == "BUY") & (s >= q_pos_60)) | ((df["side"] == "SELL") & (s <= q_neg_40))
    directional_70_30 = ((df["side"] == "BUY") & (s >= q_pos_70)) | ((df["side"] == "SELL") & (s <= q_neg_30))
    directional_80_20 = ((df["side"] == "BUY") & (s >= q_pos_80)) | ((df["side"] == "SELL") & (s <= q_neg_20))

    add_rule(rules, "directional_60_40", df[directional_60_40])
    add_rule(rules, "directional_70_30", df[directional_70_30])
    add_rule(rules, "directional_80_20", df[directional_80_20])

    add_rule(rules, "directional_60_40_risk_good", df[directional_60_40 & risk_good])
    add_rule(rules, "directional_70_30_risk_good", df[directional_70_30 & risk_good])
    add_rule(rules, "directional_80_20_risk_good", df[directional_80_20 & risk_good])

    add_rule(rules, "buy_only", df[df["side"] == "BUY"])
    add_rule(rules, "sell_only", df[df["side"] == "SELL"])

    add_rule(rules, "risk_good", df[risk_good])
    add_rule(rules, "risk_exclude_normal_weak", df[risk_not_weak])

    return rules


def main():
    rows = []

    for model, path in PATHS.items():
        print("loading", model, path)
        df = prep(pd.read_csv(path, low_memory=False))

        rules = make_rules(df)

        for rule, sub in rules.items():
            rows.append(summarize(sub, model, rule))

    out = pd.DataFrame(rows)

    cols = [
        "model", "rule", "num_trades", "num_days", "num_securities",
        "buy_trades", "sell_trades",
        "total_notional", "total_fee", "total_gross_pnl", "total_net_pnl",
        "net_return_bps_weighted", "avg_net_pnl", "median_net_pnl",
        "avg_net_pnl_bps", "win_rate", "positive_day_ratio",
        "min_daily_pnl", "max_daily_pnl",
        "buy_net_pnl", "sell_net_pnl",
        "positive_symbols", "negative_symbols",
        "top10_symbol_pnl", "top10_share_of_total",
        "bottom10_symbol_pnl", "bottom10_share_of_total",
    ]

    out = out[[c for c in cols if c in out.columns]]

    out = out.sort_values(["model", "net_return_bps_weighted"], ascending=[True, False])

    print("\n===== combined filter candidates =====")
    print(out.to_string(index=False))

    out.to_csv(OUT, index=False)
    print("\nsaved:", OUT)

    print("\n===== queue_0p05 top 20 by weighted bps =====")
    q = out[out["model"] == "queue_0p05"].sort_values("net_return_bps_weighted", ascending=False).head(20)
    print(q.to_string(index=False))

    print("\n===== queue_0p05 top 20 by total net pnl =====")
    q2 = out[out["model"] == "queue_0p05"].sort_values("total_net_pnl", ascending=False).head(20)
    print(q2.to_string(index=False))


if __name__ == "__main__":
    main()
