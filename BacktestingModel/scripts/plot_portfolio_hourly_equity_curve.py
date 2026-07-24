#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def parse_hft_datetime_series(s):
    s = s.astype(str).str.strip()

    dt = pd.to_datetime(s, errors="coerce")

    bad = dt.isna()
    if bad.any():
        x = s[bad]

        m = x.str.extract(r"(?P<date>\d{8})[_\sT]?(?P<time>\d{6,9})")
        ok = m["date"].notna() & m["time"].notna()

        if ok.any():
            date_part = m.loc[ok, "date"]
            time_part = m.loc[ok, "time"].astype(str)

            hhmmss = time_part.str.slice(0, 6)
            frac = time_part.str.slice(6)

            base = pd.to_datetime(
                date_part + hhmmss,
                format="%Y%m%d%H%M%S",
                errors="coerce",
            )

            ns = frac.apply(lambda z: int(str(z).ljust(9, "0")[:9]) if str(z) else 0)
            parsed = base + pd.to_timedelta(ns.values, unit="ns")

            dt.loc[x.index[ok]] = parsed.values

    bad = dt.isna()
    if bad.any():
        x = s[bad]
        m = x.str.extract(r"(?P<date>\d{8})")
        ok = m["date"].notna()

        if ok.any():
            parsed = pd.to_datetime(m.loc[ok, "date"], format="%Y%m%d", errors="coerce")
            dt.loc[x.index[ok]] = parsed.values

    return dt


def parse_datetime(df):
    date_col = find_col(df, ["date", "trading_date", "trade_date"])
    time_col = find_col(df, ["clock_time", "bar_time", "hhmmss"])

    if date_col is not None and time_col is not None:
        s = df[date_col].astype(str) + "_" + df[time_col].astype(str)
        dt = parse_hft_datetime_series(s)
        if dt.notna().sum() > 0:
            return dt

    dt_col = find_col(
        df,
        [
            "datetime",
            "timestamp",
            "portfolio_time",
            "event_time",
            "decision_time",
            "fill_time",
            "markout_time",
            "time",
        ],
    )

    if dt_col is not None:
        dt = parse_hft_datetime_series(df[dt_col])
        if dt.notna().sum() > 0:
            return dt

    if date_col is not None:
        dt = parse_hft_datetime_series(df[date_col])
        if dt.notna().sum() > 0:
            return dt

    raise ValueError("Cannot infer datetime column. Please check input columns.")


def pick_value_col(df, user_col=None):
    if user_col is not None:
        if user_col not in df.columns:
            raise ValueError(f"value column not found: {user_col}")
        return user_col

    candidates = [
        "equity",
        "final_equity",
        "portfolio_equity",
        "total_equity",
        "account_value",
        "portfolio_value",
        "cum_pnl",
        "cumulative_pnl",
        "daily_cum_pnl",
        "total_pnl",
        "pnl",
    ]

    c = find_col(df, candidates)
    if c is not None:
        return c

    cash_col = find_col(df, ["cash", "final_cash"])
    inv_col = find_col(df, ["inventory_value", "final_inventory_value"])

    if cash_col is not None and inv_col is not None:
        df["_computed_equity"] = (
            pd.to_numeric(df[cash_col], errors="coerce")
            + pd.to_numeric(df[inv_col], errors="coerce")
        )
        return "_computed_equity"

    numeric_cols = []
    for col in df.columns:
        s = pd.to_numeric(df[col], errors="coerce")
        if s.notna().sum() > 0:
            numeric_cols.append(col)

    if not numeric_cols:
        raise ValueError("Cannot infer value column.")

    return numeric_cols[-1]


def infer_paths(model_key, curve_file, summary_file):
    if curve_file is None:
        if model_key is None:
            raise ValueError("Either --model-key or --curve-file is required.")
        curve_file = f"outputs/portfolio/portfolio_replay_{model_key}_curve.csv"

    if summary_file is None and model_key is not None:
        summary_file = f"outputs/portfolio/portfolio_replay_{model_key}_summary.csv"

    return Path(curve_file), Path(summary_file) if summary_file else None


def read_summary(summary_file):
    if summary_file is None or not summary_file.exists():
        return {}

    df = pd.read_csv(summary_file)
    if df.empty:
        return {}

    row = df.iloc[0].to_dict()
    out = {}
    for k, v in row.items():
        try:
            out[k] = float(v)
        except Exception:
            out[k] = v
    return out


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key", default=None)
    parser.add_argument("--curve-file", default=None)
    parser.add_argument("--summary-file", default=None)
    parser.add_argument("--output-dir", default="outputs/metrics/hourly_portfolio_curves")
    parser.add_argument("--value-col", default=None)
    parser.add_argument("--freq", default="1H")
    parser.add_argument("--capital", type=float, default=None)
    parser.add_argument("--max-xticks", type=int, default=24)
    parser.add_argument("--title", default=None)
    parser.add_argument("--no-rebase", action="store_true")
    args = parser.parse_args()

    curve_file, summary_file = infer_paths(args.model_key, args.curve_file, args.summary_file)

    if not curve_file.exists():
        raise FileNotFoundError(f"curve file not found: {curve_file}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(curve_file, low_memory=False)
    if df.empty:
        raise ValueError(f"empty curve file: {curve_file}")

    df["_dt"] = parse_datetime(df)
    df = df[df["_dt"].notna()].copy()
    df = df.sort_values("_dt")

    value_col = pick_value_col(df, args.value_col)
    df["_equity_raw"] = pd.to_numeric(df[value_col], errors="coerce")
    df = df[df["_equity_raw"].notna()].copy()

    if df.empty:
        raise ValueError("no valid equity values after parsing")

    df["_hour"] = df["_dt"].dt.floor(args.freq)

    hourly = (
        df.sort_values("_dt")
        .groupby("_hour", as_index=False)
        .tail(1)
        .copy()
        .sort_values("_hour")
    )

    hourly = hourly[["_hour", "_equity_raw"]].rename(
        columns={"_hour": "datetime", "_equity_raw": "equity"}
    )

    if not args.no_rebase:
        hourly["equity_rebased"] = hourly["equity"] - hourly["equity"].iloc[0]
        plot_col = "equity_rebased"
        y_label = "Equity / cumulative PnL, rebased"
    else:
        plot_col = "equity"
        y_label = value_col

    hourly["hourly_change"] = hourly[plot_col].diff()
    hourly.loc[hourly.index[0], "hourly_change"] = hourly.loc[hourly.index[0], plot_col]

    hourly["trading_index"] = range(len(hourly))
    hourly["label"] = pd.to_datetime(hourly["datetime"]).dt.strftime("%Y-%m-%d %H:%M")

    summary = read_summary(summary_file)

    final_pnl = safe_float(summary.get("total_pnl", hourly[plot_col].iloc[-1]))
    total_turnover = safe_float(summary.get("total_turnover", np.nan))
    max_gross_exposure = safe_float(summary.get("max_gross_exposure", np.nan))
    pnl_bps = safe_float(summary.get("pnl_bps_on_turnover", np.nan))
    max_drawdown = safe_float(summary.get("max_drawdown", np.nan))

    if args.capital is not None and args.capital != 0:
        return_on_capital = final_pnl / args.capital
    else:
        return_on_capital = np.nan

    if np.isfinite(max_gross_exposure) and max_gross_exposure != 0:
        return_on_max_gross_exposure = final_pnl / max_gross_exposure
    else:
        return_on_max_gross_exposure = np.nan

    stem = curve_file.stem
    out_csv = out_dir / f"{stem}_hourly_portfolio_curve.csv"
    out_png = out_dir / f"{stem}_hourly_portfolio_curve.png"
    out_bar_png = out_dir / f"{stem}_hourly_portfolio_change_bar.png"
    out_summary = out_dir / f"{stem}_hourly_summary.csv"

    hourly.to_csv(out_csv, index=False)

    tick_step = max(1, int(np.ceil(len(hourly) / max(1, args.max_xticks))))
    tick_idx = list(range(0, len(hourly), tick_step))
    if len(hourly) - 1 not in tick_idx:
        tick_idx.append(len(hourly) - 1)

    plt.figure(figsize=(16, 6))
    plt.plot(hourly["trading_index"], hourly[plot_col], marker="o", linewidth=1.5)
    plt.title(args.title or f"Hourly Portfolio Equity Curve: {stem}")
    plt.xlabel("Trading-hour index")
    plt.ylabel(y_label)
    plt.xticks(
        hourly.loc[tick_idx, "trading_index"],
        hourly.loc[tick_idx, "label"],
        rotation=45,
        ha="right",
    )
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

    plt.figure(figsize=(16, 6))
    plt.bar(hourly["trading_index"], hourly["hourly_change"])
    plt.title(args.title or f"Hourly Portfolio Change: {stem}")
    plt.xlabel("Trading-hour index")
    plt.ylabel("Hourly change")
    plt.xticks(
        hourly.loc[tick_idx, "trading_index"],
        hourly.loc[tick_idx, "label"],
        rotation=45,
        ha="right",
    )
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_bar_png, dpi=160)
    plt.close()

    report = pd.DataFrame(
        [
            {
                "curve_file": str(curve_file),
                "summary_file": str(summary_file) if summary_file else "",
                "value_col": value_col,
                "hourly_points": len(hourly),
                "first_time": hourly["datetime"].iloc[0],
                "last_time": hourly["datetime"].iloc[-1],
                "final_pnl_from_summary": final_pnl,
                "total_turnover": total_turnover,
                "pnl_bps_on_turnover": pnl_bps,
                "max_gross_exposure": max_gross_exposure,
                "max_drawdown": max_drawdown,
                "capital": args.capital if args.capital is not None else np.nan,
                "return_on_capital": return_on_capital,
                "return_on_capital_pct": return_on_capital * 100 if np.isfinite(return_on_capital) else np.nan,
                "return_on_max_gross_exposure": return_on_max_gross_exposure,
                "return_on_max_gross_exposure_pct": return_on_max_gross_exposure * 100 if np.isfinite(return_on_max_gross_exposure) else np.nan,
            }
        ]
    )

    report.to_csv(out_summary, index=False)

    print("input curve:", curve_file)
    print("summary file:", summary_file)
    print("value_col:", value_col)
    print("hourly points:", len(hourly))
    print("saved curve csv:", out_csv)
    print("saved equity png:", out_png)
    print("saved hourly change png:", out_bar_png)
    print("saved summary:", out_summary)
    print("")
    print("===== return summary =====")
    print(report.T.to_string(header=False))


if __name__ == "__main__":
    main()
