#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


def mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def to_num(x):
    return pd.to_numeric(x, errors="coerce")


def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def fmt_value(x, digits=4):
    if pd.isna(x):
        return ""
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, (float, np.floating)):
        ax = abs(float(x))
        if ax >= 1e8 or (ax > 0 and ax < 1e-4):
            return f"{float(x):.{digits}e}"
        if ax >= 1000:
            return f"{float(x):,.2f}"
        return f"{float(x):.{digits}f}"
    return str(x)


def df_to_markdown(df, max_rows=None):
    if max_rows is not None:
        df = df.head(max_rows).copy()

    if df.empty:
        return "_empty_"

    cols = list(df.columns)
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")

    for _, row in df.iterrows():
        vals = [fmt_value(row[c]) for c in cols]
        vals = [v.replace("\n", " ") for v in vals]
        lines.append("| " + " | ".join(vals) + " |")

    return "\n".join(lines)


def save_markdown(sections, out_path):
    text = "\n\n".join(sections)
    Path(out_path).write_text(text, encoding="utf-8")
    return text


def daily_path(portfolio_dir, model_key):
    path = os.path.join(portfolio_dir, f"portfolio_replay_{model_key}_daily.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return path


def summary_path(portfolio_dir, model_key):
    path = os.path.join(portfolio_dir, f"portfolio_replay_{model_key}_summary.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return path


def prep_daily(path):
    df = pd.read_csv(path, low_memory=False)

    if "date" not in df.columns:
        df["date"] = np.arange(len(df)).astype(str)
    df["date"] = df["date"].astype(str)
    df = df.sort_values("date")

    pnl_col = find_col(df, ["daily_pnl", "pnl", "net_pnl", "total_pnl"])
    if pnl_col is None:
        raise ValueError(f"no daily pnl column in {path}")
    df["daily_pnl"] = to_num(df[pnl_col])

    turnover_col = find_col(df, ["turnover", "total_turnover", "total_notional", "notional"])
    if turnover_col is not None:
        df["turnover"] = to_num(df[turnover_col])
        df["return_bps"] = df["daily_pnl"] / df["turnover"].replace(0, np.nan) * 10000.0

    if "max_gross_exposure" in df.columns:
        df["max_gross_exposure"] = to_num(df["max_gross_exposure"])

    if "max_drawdown" in df.columns:
        df["max_drawdown"] = to_num(df["max_drawdown"])

    if "num_trades" in df.columns:
        df["num_trades"] = to_num(df["num_trades"])

    df["cum_pnl"] = df["daily_pnl"].cumsum()
    df["drawdown"] = df["cum_pnl"] - df["cum_pnl"].cummax()

    return df


def load_summary(path):
    df = pd.read_csv(path, low_memory=False)
    return df.iloc[0].to_dict() if len(df) else {}


def build_overall_table(base_summary, opt_summary):
    metrics = [
        "num_trades",
        "num_days",
        "num_securities",
        "total_pnl",
        "total_turnover",
        "pnl_bps_on_turnover",
        "positive_day_ratio",
        "daily_pnl_sharpe",
        "max_gross_exposure",
        "return_on_max_gross_exposure",
        "max_abs_net_exposure",
        "max_drawdown",
        "num_short_events",
        "num_short_violations_if_no_short",
    ]

    rows = []
    for m in metrics:
        if m in base_summary or m in opt_summary:
            b = base_summary.get(m, np.nan)
            o = opt_summary.get(m, np.nan)
            diff = o - b if pd.notna(b) and pd.notna(o) and isinstance(b, (int, float, np.integer, np.floating)) and isinstance(o, (int, float, np.integer, np.floating)) else np.nan
            pct = diff / b if pd.notna(diff) and pd.notna(b) and b != 0 else np.nan
            rows.append({
                "metric": m,
                "baseline": b,
                "optimizer": o,
                "diff_opt_minus_base": diff,
                "pct_diff": pct,
            })

    return pd.DataFrame(rows)


def build_daily_attribution(base_daily, opt_daily):
    b = prep_daily(base_daily)
    o = prep_daily(opt_daily)

    common = sorted(set(b["date"]) & set(o["date"]))
    b = b[b["date"].isin(common)].copy()
    o = o[o["date"].isin(common)].copy()

    keep = ["date", "num_trades", "turnover", "daily_pnl", "return_bps", "max_gross_exposure", "max_drawdown", "cum_pnl", "drawdown"]
    b_keep = [c for c in keep if c in b.columns]
    o_keep = [c for c in keep if c in o.columns]

    m = b[b_keep].merge(o[o_keep], on="date", suffixes=("_base", "_opt"))

    if "daily_pnl_base" in m.columns and "daily_pnl_opt" in m.columns:
        m["pnl_diff"] = m["daily_pnl_opt"] - m["daily_pnl_base"]

    if "return_bps_base" in m.columns and "return_bps_opt" in m.columns:
        m["bps_diff"] = m["return_bps_opt"] - m["return_bps_base"]

    if "turnover_base" in m.columns and "turnover_opt" in m.columns:
        m["turnover_diff"] = m["turnover_opt"] - m["turnover_base"]

    if "max_gross_exposure_base" in m.columns and "max_gross_exposure_opt" in m.columns:
        m["exposure_diff"] = m["max_gross_exposure_opt"] - m["max_gross_exposure_base"]

    final_cols = [
        "date",
        "daily_pnl_base",
        "daily_pnl_opt",
        "pnl_diff",
        "return_bps_base",
        "return_bps_opt",
        "bps_diff",
        "turnover_base",
        "turnover_opt",
        "turnover_diff",
        "max_gross_exposure_base",
        "max_gross_exposure_opt",
        "exposure_diff",
        "max_drawdown_base",
        "max_drawdown_opt",
    ]
    final_cols = [c for c in final_cols if c in m.columns]
    return m[final_cols].copy()


def daily_summary_table(daily_attr):
    rows = []
    for col in ["pnl_diff", "bps_diff", "turnover_diff", "exposure_diff"]:
        if col in daily_attr.columns:
            s = to_num(daily_attr[col]).dropna()
            rows.append({
                "metric": col,
                "mean": s.mean(),
                "median": s.median(),
                "min": s.min(),
                "max": s.max(),
                "positive_count": int((s > 0).sum()),
                "negative_count": int((s < 0).sum()),
            })
    return pd.DataFrame(rows)


def prep_trade(path, signal_col):
    df = pd.read_csv(path, low_memory=False)

    side_col = find_col(df, ["side", "trade_side", "fill_side"])
    df["_side"] = df[side_col].astype(str) if side_col else "ALL"

    risk_col = find_col(df, ["risk_state", "regime", "state"])
    df["_risk_state"] = df[risk_col].astype(str) if risk_col else "UNKNOWN"

    pnl_col = find_col(df, ["net_pnl", "total_net_pnl", "pnl"])
    if pnl_col is None:
        raise ValueError(f"no pnl column in {path}")
    df["_pnl"] = to_num(df[pnl_col])

    bps_col = find_col(df, ["net_pnl_bps", "avg_net_pnl_bps", "pnl_bps", "return_bps"])
    if bps_col:
        df["_bps"] = to_num(df[bps_col])
    else:
        notional_col = find_col(df, ["notional", "total_notional", "turnover"])
        if notional_col:
            df["_bps"] = df["_pnl"] / to_num(df[notional_col]).replace(0, np.nan) * 10000.0
        else:
            df["_bps"] = np.nan

    sig_col = signal_col if signal_col in df.columns else find_col(df, ["signal", "pred", "raw_pred", "pred_used"])
    if sig_col:
        df["_signal"] = to_num(df[sig_col])
        df["_abs_signal"] = df["_signal"].abs()
    else:
        df["_signal"] = np.nan
        df["_abs_signal"] = np.nan

    tcol = find_col(df, ["datetime", "fill_time", "decision_time", "timestamp", "time"])
    if tcol:
        t = pd.to_datetime(df[tcol], errors="coerce")
        df["_time_bucket"] = t.dt.strftime("%H:00")
    else:
        df["_time_bucket"] = "UNKNOWN"

    try:
        if df["_abs_signal"].notna().sum() >= 5:
            df["_signal_bin"] = pd.qcut(df["_abs_signal"], q=5, labels=False, duplicates="drop").astype(str)
        else:
            df["_signal_bin"] = "UNKNOWN"
    except Exception:
        df["_signal_bin"] = "UNKNOWN"

    return df


def aggregate_trade(df, group_cols):
    g = df.groupby(group_cols, dropna=False)
    out = g.agg(
        num_trades=("_pnl", "size"),
        total_pnl=("_pnl", "sum"),
        avg_pnl=("_pnl", "mean"),
        avg_bps=("_bps", "mean"),
        win_rate=("_pnl", lambda x: (x > 0).mean()),
        avg_abs_signal=("_abs_signal", "mean"),
    ).reset_index()
    return out


def compare_trade_group(base, opt, group_cols):
    b = aggregate_trade(base, group_cols)
    o = aggregate_trade(opt, group_cols)
    m = b.merge(o, on=group_cols, how="outer", suffixes=("_base", "_opt")).fillna(0)

    for c in ["num_trades", "total_pnl", "avg_pnl", "avg_bps", "win_rate", "avg_abs_signal"]:
        bcol = c + "_base"
        ocol = c + "_opt"
        if bcol in m.columns and ocol in m.columns:
            m[c + "_diff"] = m[ocol] - m[bcol]

    order_cols = group_cols + [
        "num_trades_base",
        "num_trades_opt",
        "num_trades_diff",
        "total_pnl_base",
        "total_pnl_opt",
        "total_pnl_diff",
        "avg_bps_base",
        "avg_bps_opt",
        "avg_bps_diff",
        "win_rate_base",
        "win_rate_opt",
        "win_rate_diff",
        "avg_abs_signal_base",
        "avg_abs_signal_opt",
        "avg_abs_signal_diff",
    ]
    order_cols = [c for c in order_cols if c in m.columns]
    return m[order_cols].copy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-key", required=True)
    parser.add_argument("--opt-model-key", required=True)
    parser.add_argument("--portfolio-dir", default="outputs/portfolio")
    parser.add_argument("--out-dir", default="outputs/metrics/optimizer_attribution_tables")
    parser.add_argument("--base-trades", default="")
    parser.add_argument("--opt-trades", default="")
    parser.add_argument("--signal-col", default="hidden_factor_mlp2_h60")
    parser.add_argument("--max-rows", type=int, default=30)
    args = parser.parse_args()

    mkdir(args.out_dir)

    base_summary = load_summary(summary_path(args.portfolio_dir, args.base_model_key))
    opt_summary = load_summary(summary_path(args.portfolio_dir, args.opt_model_key))

    overall = build_overall_table(base_summary, opt_summary)
    daily = build_daily_attribution(
        daily_path(args.portfolio_dir, args.base_model_key),
        daily_path(args.portfolio_dir, args.opt_model_key),
    )
    daily_sum = daily_summary_table(daily)

    overall.to_csv(os.path.join(args.out_dir, "overall_comparison.csv"), index=False)
    daily.to_csv(os.path.join(args.out_dir, "daily_attribution.csv"), index=False)
    daily_sum.to_csv(os.path.join(args.out_dir, "daily_attribution_summary.csv"), index=False)

    sections = []
    sections.append("# Optimizer Attribution Tables")
    sections.append("## Overall Comparison\n\n" + df_to_markdown(overall))
    sections.append("## Daily Attribution\n\n" + df_to_markdown(daily, max_rows=args.max_rows))
    sections.append("## Daily Attribution Summary\n\n" + df_to_markdown(daily_sum))

    if args.base_trades and args.opt_trades:
        base_trade = prep_trade(args.base_trades, args.signal_col)
        opt_trade = prep_trade(args.opt_trades, args.signal_col)

        for name, group_cols in [
            ("by_side", ["_side"]),
            ("by_risk_state", ["_risk_state"]),
            ("by_signal_bin", ["_signal_bin"]),
            ("by_time_bucket", ["_time_bucket"]),
            ("by_side_risk_state", ["_side", "_risk_state"]),
        ]:
            tab = compare_trade_group(base_trade, opt_trade, group_cols)
            tab.to_csv(os.path.join(args.out_dir, f"trade_{name}.csv"), index=False)
            sections.append(f"## Trade Attribution {name}\n\n" + df_to_markdown(tab, max_rows=args.max_rows))
    else:
        sections.append(
            "## Trade Attribution\n\n"
            "Skipped because `--base-trades` and `--opt-trades` were not provided."
        )

    report_path = os.path.join(args.out_dir, "optimizer_attribution_report.md")
    report = save_markdown(sections, report_path)

    print(report)
    print("")
    print("saved markdown:", report_path)
    print("saved csv dir:", args.out_dir)


if __name__ == "__main__":
    main()
