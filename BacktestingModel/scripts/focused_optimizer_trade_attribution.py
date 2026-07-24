#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


def mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def to_num(x):
    return pd.to_numeric(x, errors="coerce")


def normalize_side(x):
    s = str(x).strip().upper()
    if s in ["BUY", "BID"]:
        return "BUY"
    if s in ["SELL", "ASK"]:
        return "SELL"
    if s in ["1", "1.0"]:
        return "BUY"
    if s in ["0", "0.0"]:
        return "SELL"
    return s


def extract_date(series):
    s = series.astype(str)
    first8 = s.str.extract(r"(\d{8})", expand=False)
    out = first8.copy()
    mask = out.isna()
    if mask.any():
        dt = pd.to_datetime(s[mask], errors="coerce")
        out.loc[mask] = dt.dt.strftime("%Y%m%d")
    return out


def load_trades(path, signal_col):
    df = pd.read_csv(path, low_memory=False)

    date_col = find_col(
        df,
        [
            "date",
            "trading_date",
            "trade_date",
            "datetime",
            "fill_time",
            "decision_time",
            "timestamp",
            "time",
        ],
    )
    if date_col is None:
        raise ValueError("cannot infer date/time column from " + path)
    df["_date"] = extract_date(df[date_col])

    side_col = find_col(df, ["side", "quote_side", "trade_side", "fill_side"])
    if side_col is None:
        df["_side"] = "ALL"
    else:
        df["_side"] = df[side_col].map(normalize_side)

    risk_col = find_col(df, ["risk_state", "regime", "state"])
    if risk_col is None:
        df["_risk_state"] = "UNKNOWN"
    else:
        df["_risk_state"] = df[risk_col].astype(str)

    pnl_col = find_col(df, ["net_pnl", "total_net_pnl", "pnl"])
    if pnl_col is None:
        raise ValueError("cannot infer pnl column from " + path)
    df["_pnl"] = to_num(df[pnl_col])

    bps_col = find_col(df, ["net_pnl_bps", "avg_net_pnl_bps", "pnl_bps", "return_bps"])
    if bps_col is not None:
        df["_bps"] = to_num(df[bps_col])
    else:
        notional_col = find_col(df, ["notional", "total_notional", "turnover"])
        if notional_col is not None:
            df["_bps"] = df["_pnl"] / to_num(df[notional_col]).replace(0, np.nan) * 10000.0
        else:
            df["_bps"] = np.nan

    sig_col = signal_col if signal_col in df.columns else find_col(df, ["signal", "pred", "raw_pred", "pred_used"])
    if sig_col is not None:
        df["_signal"] = to_num(df[sig_col])
        df["_abs_signal"] = df["_signal"].abs()
    else:
        df["_signal"] = np.nan
        df["_abs_signal"] = np.nan

    security_col = find_col(df, ["securityid", "SecurityID", "symbol"])
    if security_col is not None:
        df["_securityid"] = df[security_col].astype(str).str.zfill(6)
    else:
        df["_securityid"] = "UNKNOWN"

    return df


def agg(df, group_cols):
    if len(df) == 0:
        return pd.DataFrame(columns=group_cols)

    out = (
        df.groupby(group_cols, dropna=False)
        .agg(
            num_trades=("_pnl", "size"),
            total_pnl=("_pnl", "sum"),
            avg_pnl=("_pnl", "mean"),
            avg_bps=("_bps", "mean"),
            win_rate=("_pnl", lambda x: float((x > 0).mean())),
            avg_abs_signal=("_abs_signal", "mean"),
            num_securities=("_securityid", "nunique"),
        )
        .reset_index()
    )
    return out


def compare(base, opt, group_cols):
    b = agg(base, group_cols)
    o = agg(opt, group_cols)

    m = b.merge(o, on=group_cols, how="outer", suffixes=("_base", "_opt")).fillna(0)

    metrics = [
        "num_trades",
        "total_pnl",
        "avg_pnl",
        "avg_bps",
        "win_rate",
        "avg_abs_signal",
        "num_securities",
    ]

    for c in metrics:
        bcol = c + "_base"
        ocol = c + "_opt"
        if bcol in m.columns and ocol in m.columns:
            m[c + "_diff"] = m[ocol] - m[bcol]

    ordered = group_cols + [
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
        "num_securities_base",
        "num_securities_opt",
        "num_securities_diff",
    ]

    ordered = [c for c in ordered if c in m.columns]
    return m[ordered].copy()


def fmt(x):
    if pd.isna(x):
        return ""
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, (float, np.floating)):
        ax = abs(float(x))
        if ax >= 1000:
            return f"{float(x):,.2f}"
        return f"{float(x):.4f}"
    return str(x)


def to_markdown(df, max_rows):
    if max_rows is not None:
        df = df.head(max_rows).copy()

    if len(df) == 0:
        return "_empty_"

    cols = list(df.columns)
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")

    for _, row in df.iterrows():
        vals = [fmt(row[c]).replace("\n", " ") for c in cols]
        lines.append("| " + " | ".join(vals) + " |")

    return "\n".join(lines)


def add_section(sections, title, df, max_rows):
    sections.append("## " + title + "\n\n" + to_markdown(df, max_rows))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-trades", required=True)
    parser.add_argument("--opt-trades", required=True)
    parser.add_argument("--dates", default="20241030,20241023,20241022")
    parser.add_argument("--signal-col", default="hidden_factor_mlp2_h60")
    parser.add_argument("--out-dir", default="outputs/metrics/optimizer_trade_attribution_focus")
    parser.add_argument("--max-rows", type=int, default=40)
    args = parser.parse_args()

    mkdir(args.out_dir)

    dates = [x.strip() for x in args.dates.split(",") if x.strip()]

    base = load_trades(args.base_trades, args.signal_col)
    opt = load_trades(args.opt_trades, args.signal_col)

    base = base[base["_date"].isin(dates)].copy()
    opt = opt[opt["_date"].isin(dates)].copy()

    sections = []
    sections.append("# Focused Optimizer Trade Attribution")
    sections.append("Dates: `" + ",".join(dates) + "`")

    tables = {}

    tables["by_date"] = compare(base, opt, ["_date"]).sort_values("total_pnl_diff")
    tables["by_date_side"] = compare(base, opt, ["_date", "_side"]).sort_values(["_date", "total_pnl_diff"])
    tables["by_date_risk_state"] = compare(base, opt, ["_date", "_risk_state"]).sort_values(["_date", "total_pnl_diff"])
    tables["by_date_side_risk_state"] = compare(base, opt, ["_date", "_side", "_risk_state"]).sort_values(["_date", "total_pnl_diff"])
    tables["by_side"] = compare(base, opt, ["_side"]).sort_values("total_pnl_diff")
    tables["by_risk_state"] = compare(base, opt, ["_risk_state"]).sort_values("total_pnl_diff")
    tables["by_side_risk_state"] = compare(base, opt, ["_side", "_risk_state"]).sort_values("total_pnl_diff")

    for name, tab in tables.items():
        tab.to_csv(os.path.join(args.out_dir, name + ".csv"), index=False)

    worst = tables["by_date_side_risk_state"].sort_values("total_pnl_diff").head(args.max_rows)
    best = tables["by_date_side_risk_state"].sort_values("total_pnl_diff", ascending=False).head(args.max_rows)

    worst.to_csv(os.path.join(args.out_dir, "worst_date_side_risk_groups.csv"), index=False)
    best.to_csv(os.path.join(args.out_dir, "best_date_side_risk_groups.csv"), index=False)

    add_section(sections, "By Date", tables["by_date"], args.max_rows)
    add_section(sections, "By Date + Side", tables["by_date_side"], args.max_rows)
    add_section(sections, "By Date + Risk State", tables["by_date_risk_state"], args.max_rows)
    add_section(sections, "Worst Date + Side + Risk State Groups", worst, args.max_rows)
    add_section(sections, "Best Date + Side + Risk State Groups", best, args.max_rows)
    add_section(sections, "Overall by Side", tables["by_side"], args.max_rows)
    add_section(sections, "Overall by Risk State", tables["by_risk_state"], args.max_rows)
    add_section(sections, "Overall by Side + Risk State", tables["by_side_risk_state"], args.max_rows)

    report = "\n\n".join(sections)
    report_path = os.path.join(args.out_dir, "focused_trade_attribution_report.md")
    Path(report_path).write_text(report, encoding="utf-8")

    print(report)
    print("")
    print("saved report:", report_path)
    print("saved csv dir:", args.out_dir)


if __name__ == "__main__":
    main()
