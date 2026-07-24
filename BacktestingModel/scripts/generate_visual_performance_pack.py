#!/usr/bin/env python3
import argparse
import glob
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def safe_div(a, b):
    if b is None or pd.isna(b) or b == 0:
        return np.nan
    return a / b


def to_num(s):
    return pd.to_numeric(s, errors="coerce")


def calc_sharpe(values, annualization=252.0):
    x = pd.Series(values).dropna().astype(float)
    if len(x) < 2:
        return np.nan
    std = x.std(ddof=1)
    if pd.isna(std) or std == 0:
        return np.nan
    return x.mean() / std * math.sqrt(annualization)


def calc_rolling_sharpe(values, window=5, annualization=252.0):
    x = pd.Series(values).astype(float)
    mean = x.rolling(window=window, min_periods=2).mean()
    std = x.rolling(window=window, min_periods=2).std(ddof=1)
    return mean / std.replace(0, np.nan) * math.sqrt(annualization)


def calc_drawdown(curve):
    x = pd.Series(curve).astype(float)
    running_max = x.cummax()
    return x - running_max


def infer_model_key(summary_path):
    base = os.path.basename(summary_path)
    prefix = "portfolio_replay_"
    suffix = "_summary.csv"
    if base.startswith(prefix) and base.endswith(suffix):
        return base[len(prefix):-len(suffix)]
    if base.endswith("_summary.csv"):
        return base[:-len("_summary.csv")]
    return os.path.splitext(base)[0]


def build_paths(portfolio_dir, model_key):
    prefix = os.path.join(portfolio_dir, "portfolio_replay_" + model_key)
    return {
        "summary": prefix + "_summary.csv",
        "daily": prefix + "_daily.csv",
        "curve": prefix + "_curve.csv",
        "positions": prefix + "_positions.csv",
    }


def find_summary_by_model_key(portfolio_dir, model_key):
    path = os.path.join(portfolio_dir, "portfolio_replay_" + model_key + "_summary.csv")
    if not os.path.exists(path):
        raise FileNotFoundError("summary file not found: " + path)
    return path


def find_time_col(df):
    for col in ["time", "datetime", "timestamp", "fill_time", "decision_time", "date"]:
        if col in df.columns:
            return col
    return None


def find_value_col(df):
    for col in [
        "equity",
        "total_equity",
        "portfolio_equity",
        "cum_pnl",
        "cumulative_pnl",
        "pnl",
        "total_pnl",
        "curve_value",
    ]:
        if col in df.columns:
            return col
    return None


def read_csv_if_exists(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


def save_table(df, path):
    ensure_dir(os.path.dirname(path))
    df.to_csv(path, index=False)


def plot_line(df, x_col, y_col, title, ylabel, out_path):
    if not HAS_MATPLOTLIB:
        return
    if df is None or len(df) == 0 or y_col not in df.columns:
        return

    plt.figure(figsize=(12, 6))
    if x_col is not None and x_col in df.columns:
        marker = "o" if len(df) <= 30 else None
        plt.plot(df[x_col], df[y_col], marker=marker)
        plt.xlabel(x_col)
        if len(df) <= 40:
            plt.xticks(rotation=45)
    else:
        plt.plot(df[y_col].values)
        plt.xlabel("index")

    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_bar(df, x_col, y_col, title, ylabel, out_path):
    if not HAS_MATPLOTLIB:
        return
    if df is None or len(df) == 0 or y_col not in df.columns:
        return

    plt.figure(figsize=(12, 6))
    if x_col is not None and x_col in df.columns:
        x = df[x_col].astype(str)
    else:
        x = np.arange(len(df))

    plt.bar(x, df[y_col])
    plt.xlabel(x_col if x_col else "index")
    plt.ylabel(ylabel)
    plt.title(title)
    if len(df) <= 40:
        plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_hist(series, title, xlabel, out_path, bins=20):
    if not HAS_MATPLOTLIB:
        return
    x = pd.Series(series).dropna().astype(float)
    if len(x) == 0:
        return

    plt.figure(figsize=(10, 6))
    plt.hist(x, bins=bins)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_scatter(df, x_col, y_col, title, out_path):
    if not HAS_MATPLOTLIB:
        return
    if df is None or len(df) == 0:
        return
    if x_col not in df.columns or y_col not in df.columns:
        return

    plt.figure(figsize=(10, 6))
    plt.scatter(df[x_col], df[y_col])
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def build_daily_metrics(daily_df, annualization, rolling_window):
    df = daily_df.copy()
    if len(df) == 0:
        return pd.DataFrame()

    if "date" in df.columns:
        df["date"] = df["date"].astype(str)
        df = df.sort_values("date")

    pnl_col = None
    for col in ["daily_pnl", "pnl", "net_pnl", "total_pnl"]:
        if col in df.columns:
            pnl_col = col
            break

    if pnl_col is None:
        raise ValueError("daily file does not contain daily_pnl / pnl / net_pnl / total_pnl column")

    df["daily_pnl"] = to_num(df[pnl_col])
    df["cum_pnl"] = df["daily_pnl"].cumsum()
    df["drawdown"] = calc_drawdown(df["cum_pnl"])
    df["rolling_sharpe"] = calc_rolling_sharpe(
        df["daily_pnl"],
        window=rolling_window,
        annualization=annualization,
    )

    if "turnover" in df.columns:
        df["turnover"] = to_num(df["turnover"])
        df["daily_return_bps"] = df["daily_pnl"] / df["turnover"].replace(0, np.nan) * 10000.0
    elif "total_turnover" in df.columns:
        df["turnover"] = to_num(df["total_turnover"])
        df["daily_return_bps"] = df["daily_pnl"] / df["turnover"].replace(0, np.nan) * 10000.0

    if "max_gross_exposure" in df.columns:
        df["max_gross_exposure"] = to_num(df["max_gross_exposure"])
        df["daily_return_on_max_gross_exposure"] = (
            df["daily_pnl"] / df["max_gross_exposure"].replace(0, np.nan)
        )

    if "fee" in df.columns:
        df["fee"] = to_num(df["fee"])

    if "num_trades" in df.columns:
        df["num_trades"] = to_num(df["num_trades"])

    return df


def build_intraday_metrics(curve_df):
    df = curve_df.copy()
    if len(df) == 0:
        return pd.DataFrame()

    time_col = find_time_col(df)
    if time_col is not None:
        if time_col != "date":
            df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        df = df.sort_values(time_col)

    value_col = find_value_col(df)
    if value_col is None:
        return pd.DataFrame()

    df["curve_value"] = to_num(df[value_col])
    df["drawdown"] = calc_drawdown(df["curve_value"])

    if "turnover" in df.columns:
        df["turnover"] = to_num(df["turnover"])

    if "max_gross_exposure" in df.columns:
        df["max_gross_exposure"] = to_num(df["max_gross_exposure"])

    return df


def build_summary_metrics(model_key, summary_df, daily_metrics, intraday_metrics, positions_df, annualization):
    out = {"model_key": model_key}

    if len(summary_df) > 0:
        row = summary_df.iloc[0].to_dict()
        for col in [
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
            "final_cash",
            "final_inventory_value",
            "final_equity",
            "num_long_symbols_end",
            "num_short_symbols_end",
            "num_short_events",
            "num_short_violations_if_no_short",
        ]:
            if col in row:
                out[col] = row[col]

    if len(daily_metrics) > 0:
        daily_pnl = to_num(daily_metrics["daily_pnl"]).dropna()
        out["num_days"] = len(daily_pnl)
        out["positive_days"] = int((daily_pnl > 0).sum())
        out["negative_days"] = int((daily_pnl < 0).sum())
        out["zero_days"] = int((daily_pnl == 0).sum())
        out["positive_day_ratio"] = safe_div(out["positive_days"], out["num_days"])
        out["avg_daily_pnl"] = daily_pnl.mean()
        out["median_daily_pnl"] = daily_pnl.median()
        out["std_daily_pnl"] = daily_pnl.std(ddof=1) if len(daily_pnl) >= 2 else np.nan
        out["min_daily_pnl"] = daily_pnl.min()
        out["max_daily_pnl"] = daily_pnl.max()
        out["daily_pnl_sharpe"] = calc_sharpe(daily_pnl, annualization)
        out["daily_cum_pnl_max_drawdown"] = daily_metrics["drawdown"].min()

        if "turnover" in daily_metrics.columns:
            turnover = to_num(daily_metrics["turnover"])
            out["total_turnover_from_daily"] = turnover.sum()
            out["avg_daily_turnover"] = turnover.mean()
            out["median_daily_turnover"] = turnover.median()

        if "daily_return_bps" in daily_metrics.columns:
            daily_return_bps = to_num(daily_metrics["daily_return_bps"]).dropna()
            out["avg_daily_return_bps"] = daily_return_bps.mean()
            out["median_daily_return_bps"] = daily_return_bps.median()
            out["daily_return_bps_sharpe"] = calc_sharpe(daily_return_bps, annualization)

        if "max_gross_exposure" in daily_metrics.columns:
            exposure = to_num(daily_metrics["max_gross_exposure"])
            out["avg_daily_max_gross_exposure"] = exposure.mean()
            out["max_daily_max_gross_exposure"] = exposure.max()

        if "fee" in daily_metrics.columns:
            fee = to_num(daily_metrics["fee"])
            out["total_fee_from_daily"] = fee.sum()

    if len(intraday_metrics) > 0:
        out["intraday_curve_max_drawdown"] = intraday_metrics["drawdown"].min()
        out["intraday_curve_final_value"] = intraday_metrics["curve_value"].iloc[-1]
        out["intraday_curve_min_value"] = intraday_metrics["curve_value"].min()
        out["intraday_curve_max_value"] = intraday_metrics["curve_value"].max()

    if len(positions_df) > 0:
        out["num_position_rows"] = len(positions_df)
        for col in ["ending_value", "abs_ending_value"]:
            if col in positions_df.columns:
                v = to_num(positions_df[col])
                out["total_" + col] = v.sum()
                out["max_" + col] = v.max()

    return pd.DataFrame([out])


def generate_charts(model_key, daily_metrics, intraday_metrics, positions_df, charts_dir):
    ensure_dir(charts_dir)

    if len(daily_metrics) > 0:
        x_col = "date" if "date" in daily_metrics.columns else None

        plot_line(
            daily_metrics,
            x_col,
            "cum_pnl",
            "Cumulative Daily PnL - " + model_key,
            "Cumulative PnL",
            os.path.join(charts_dir, "01_cumulative_daily_pnl.png"),
        )

        plot_bar(
            daily_metrics,
            x_col,
            "daily_pnl",
            "Daily PnL - " + model_key,
            "Daily PnL",
            os.path.join(charts_dir, "02_daily_pnl_bar.png"),
        )

        plot_line(
            daily_metrics,
            x_col,
            "drawdown",
            "Daily Drawdown - " + model_key,
            "Drawdown",
            os.path.join(charts_dir, "03_daily_drawdown.png"),
        )

        if "rolling_sharpe" in daily_metrics.columns:
            plot_line(
                daily_metrics,
                x_col,
                "rolling_sharpe",
                "Rolling Daily PnL Sharpe - " + model_key,
                "Rolling Sharpe",
                os.path.join(charts_dir, "04_rolling_sharpe.png"),
            )

        plot_hist(
            daily_metrics["daily_pnl"],
            "Daily PnL Distribution - " + model_key,
            "Daily PnL",
            os.path.join(charts_dir, "05_daily_pnl_histogram.png"),
            bins=min(20, max(5, len(daily_metrics))),
        )

        if "daily_return_bps" in daily_metrics.columns:
            plot_bar(
                daily_metrics,
                x_col,
                "daily_return_bps",
                "Daily Return on Turnover Bps - " + model_key,
                "bps",
                os.path.join(charts_dir, "06_daily_return_bps.png"),
            )

        if "max_gross_exposure" in daily_metrics.columns:
            plot_line(
                daily_metrics,
                x_col,
                "max_gross_exposure",
                "Daily Max Gross Exposure - " + model_key,
                "Max Gross Exposure",
                os.path.join(charts_dir, "07_daily_max_gross_exposure.png"),
            )

        if "turnover" in daily_metrics.columns:
            plot_bar(
                daily_metrics,
                x_col,
                "turnover",
                "Daily Turnover - " + model_key,
                "Turnover",
                os.path.join(charts_dir, "08_daily_turnover.png"),
            )

        if "turnover" in daily_metrics.columns:
            plot_scatter(
                daily_metrics,
                "turnover",
                "daily_pnl",
                "Daily PnL vs Turnover - " + model_key,
                os.path.join(charts_dir, "09_daily_pnl_vs_turnover.png"),
            )

    if len(intraday_metrics) > 0:
        x_col = find_time_col(intraday_metrics)

        plot_line(
            intraday_metrics,
            x_col,
            "curve_value",
            "Intraday Equity / PnL Curve - " + model_key,
            "Curve Value",
            os.path.join(charts_dir, "10_intraday_curve.png"),
        )

        plot_line(
            intraday_metrics,
            x_col,
            "drawdown",
            "Intraday Drawdown - " + model_key,
            "Drawdown",
            os.path.join(charts_dir, "11_intraday_drawdown.png"),
        )

    if len(positions_df) > 0 and "ending_value" in positions_df.columns:
        tmp = positions_df.copy()
        tmp["ending_value"] = to_num(tmp["ending_value"])
        tmp = tmp.sort_values("ending_value", ascending=False).head(30)

        x_col = None
        for col in ["securityid", "symbol", "SecurityID"]:
            if col in tmp.columns:
                x_col = col
                break

        plot_bar(
            tmp,
            x_col,
            "ending_value",
            "Top Ending Inventory Values - " + model_key,
            "Ending Value",
            os.path.join(charts_dir, "12_top_ending_inventory_value.png"),
        )


def write_markdown_report(model_key, summary_metrics, out_dir, charts_dir, paths):
    report_path = os.path.join(out_dir, "report.md")
    row = summary_metrics.iloc[0].to_dict() if len(summary_metrics) > 0 else {}

    def fmt_num(x, digits=4):
        if x is None or pd.isna(x):
            return "NA"
        try:
            return f"{float(x):,.{digits}f}"
        except Exception:
            return str(x)

    lines = []
    lines.append("# Backtest Performance Pack")
    lines.append("")
    lines.append("## Model")
    lines.append("")
    lines.append("```text")
    lines.append(str(model_key))
    lines.append("```")
    lines.append("")
    lines.append("## Key Metrics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")

    for metric in [
        "num_trades",
        "num_days",
        "num_securities",
        "total_pnl",
        "total_turnover",
        "pnl_bps_on_turnover",
        "positive_day_ratio",
        "daily_pnl_sharpe",
        "avg_daily_return_bps",
        "daily_return_bps_sharpe",
        "max_gross_exposure",
        "return_on_max_gross_exposure",
        "max_abs_net_exposure",
        "max_drawdown",
        "daily_cum_pnl_max_drawdown",
        "intraday_curve_max_drawdown",
        "num_short_events",
        "num_short_violations_if_no_short",
    ]:
        if metric in row:
            lines.append("| `" + metric + "` | " + fmt_num(row[metric]) + " |")

    lines.append("")
    lines.append("## Output Files")
    lines.append("")
    lines.append("| File | Path |")
    lines.append("|---|---|")
    for name, path in paths.items():
        if path and os.path.exists(path):
            lines.append("| " + name + " | `" + path + "` |")

    lines.append("")
    lines.append("## Charts")
    lines.append("")
    if os.path.exists(charts_dir):
        chart_files = sorted(glob.glob(os.path.join(charts_dir, "*.png")))
        for chart in chart_files:
            rel = os.path.relpath(chart, out_dir)
            lines.append("### " + os.path.basename(chart))
            lines.append("")
            lines.append("![](" + rel + ")")
            lines.append("")

    Path(report_path).write_text("\n".join(lines), encoding="utf-8")
    return report_path


def generate_pack(summary_path, portfolio_dir, out_root, annualization, rolling_window, no_plots):
    model_key = infer_model_key(summary_path)
    paths = build_paths(portfolio_dir, model_key)

    summary_df = read_csv_if_exists(summary_path)
    daily_df = read_csv_if_exists(paths["daily"])
    curve_df = read_csv_if_exists(paths["curve"])
    positions_df = read_csv_if_exists(paths["positions"])

    out_dir = os.path.join(out_root, model_key)
    charts_dir = os.path.join(out_dir, "charts")
    ensure_dir(out_dir)
    ensure_dir(charts_dir)

    daily_metrics = build_daily_metrics(daily_df, annualization, rolling_window)
    intraday_metrics = build_intraday_metrics(curve_df)
    summary_metrics = build_summary_metrics(
        model_key=model_key,
        summary_df=summary_df,
        daily_metrics=daily_metrics,
        intraday_metrics=intraday_metrics,
        positions_df=positions_df,
        annualization=annualization,
    )

    summary_out = os.path.join(out_dir, "summary_metrics.csv")
    daily_out = os.path.join(out_dir, "daily_metrics.csv")
    intraday_out = os.path.join(out_dir, "intraday_metrics.csv")
    positions_out = os.path.join(out_dir, "positions_snapshot.csv")

    save_table(summary_metrics, summary_out)

    if len(daily_metrics) > 0:
        save_table(daily_metrics, daily_out)

    if len(intraday_metrics) > 0:
        save_table(intraday_metrics, intraday_out)

    if len(positions_df) > 0:
        save_table(positions_df, positions_out)

    if not no_plots:
        generate_charts(
            model_key=model_key,
            daily_metrics=daily_metrics,
            intraday_metrics=intraday_metrics,
            positions_df=positions_df,
            charts_dir=charts_dir,
        )

    report_paths = {
        "summary_metrics": summary_out,
        "daily_metrics": daily_out if os.path.exists(daily_out) else "",
        "intraday_metrics": intraday_out if os.path.exists(intraday_out) else "",
        "positions_snapshot": positions_out if os.path.exists(positions_out) else "",
        "input_summary": summary_path,
        "input_daily": paths["daily"],
        "input_curve": paths["curve"],
        "input_positions": paths["positions"],
    }

    report_path = write_markdown_report(
        model_key=model_key,
        summary_metrics=summary_metrics,
        out_dir=out_dir,
        charts_dir=charts_dir,
        paths=report_paths,
    )

    return {
        "model_key": model_key,
        "out_dir": out_dir,
        "summary_metrics": summary_out,
        "daily_metrics": daily_out if os.path.exists(daily_out) else "",
        "intraday_metrics": intraday_out if os.path.exists(intraday_out) else "",
        "report": report_path,
    }


def select_summary_files(args):
    if args.list:
        paths = sorted(
            glob.glob(os.path.join(args.portfolio_dir, "portfolio_replay_*_summary.csv"))
        )
        print("available models:")
        for path in paths:
            print(infer_model_key(path))
        raise SystemExit

    if args.summary:
        return args.summary

    if args.model_key:
        paths = []
        for item in args.model_key:
            keys = [x.strip() for x in item.split(",") if x.strip()]
            for key in keys:
                paths.append(find_summary_by_model_key(args.portfolio_dir, key))
        return paths

    if args.pattern:
        return sorted(glob.glob(os.path.join(args.portfolio_dir, args.pattern)))

    if args.all:
        return sorted(
            glob.glob(os.path.join(args.portfolio_dir, "portfolio_replay_*_summary.csv"))
        )

    raise RuntimeError("Please specify one of: --model-key, --summary, --pattern, --all, --list")


def main():
    parser = argparse.ArgumentParser(
        description="Generate full visual performance pack for one or more backtest portfolio replay results."
    )

    parser.add_argument("--model-key", nargs="*", help="model key without portfolio_replay_ prefix")
    parser.add_argument("--summary", nargs="*", help="direct path(s) to portfolio_replay_*_summary.csv")
    parser.add_argument("--pattern", default="", help="glob pattern under portfolio dir, e.g. '*optimizer*_summary.csv'")
    parser.add_argument("--all", action="store_true", help="generate packs for all portfolio summary files")
    parser.add_argument("--list", action="store_true", help="list available model keys")

    parser.add_argument("--portfolio-dir", default="outputs/portfolio")
    parser.add_argument("--out-root", default="outputs/metrics/visual_performance_pack")
    parser.add_argument("--annualization", type=float, default=252.0)
    parser.add_argument("--rolling-window", type=int, default=5)
    parser.add_argument("--no-plots", action="store_true")

    args = parser.parse_args()

    ensure_dir(args.out_root)

    summary_files = select_summary_files(args)

    if len(summary_files) == 0:
        print("No summary files selected.")
        return

    results = []

    for summary_path in summary_files:
        print("generating performance pack:", summary_path)
        result = generate_pack(
            summary_path=summary_path,
            portfolio_dir=args.portfolio_dir,
            out_root=args.out_root,
            annualization=args.annualization,
            rolling_window=args.rolling_window,
            no_plots=args.no_plots,
        )
        results.append(result)

    index_df = pd.DataFrame(results)
    index_path = os.path.join(args.out_root, "index.csv")
    index_df.to_csv(index_path, index=False)

    pd.set_option("display.max_columns", 80)
    pd.set_option("display.width", 240)

    print("")
    print("===== generated packs =====")
    print(index_df.to_string(index=False))
    print("")
    print("saved index:", index_path)

    if (not HAS_MATPLOTLIB) and (not args.no_plots):
        print("matplotlib is not available, so charts were not generated.")


if __name__ == "__main__":
    main()
