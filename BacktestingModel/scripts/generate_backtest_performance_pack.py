import argparse
import glob
import math
import os

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except Exception:
    HAS_PLOT = False


def safe_div(a, b):
    if b is None or pd.isna(b) or b == 0:
        return np.nan
    return a / b


def calc_sharpe(values, annualization=252.0):
    s = pd.Series(values).dropna().astype(float)
    if len(s) < 2:
        return np.nan
    std = s.std(ddof=1)
    if pd.isna(std) or std == 0:
        return np.nan
    return s.mean() / std * math.sqrt(annualization)


def calc_max_drawdown(values):
    s = pd.Series(values).dropna().astype(float)
    if len(s) == 0:
        return np.nan
    running_max = s.cummax()
    return (s - running_max).min()


def infer_model_key(summary_path):
    base = os.path.basename(summary_path)
    prefix = "portfolio_replay_"
    suffix = "_summary.csv"
    if base.startswith(prefix) and base.endswith(suffix):
        return base[len(prefix):-len(suffix)]
    if base.endswith("_summary.csv"):
        return base[:-len("_summary.csv")]
    return os.path.splitext(base)[0]


def related_paths(portfolio_dir, model_key):
    prefix = os.path.join(portfolio_dir, "portfolio_replay_" + model_key)
    return {
        "summary": prefix + "_summary.csv",
        "daily": prefix + "_daily.csv",
        "curve": prefix + "_curve.csv",
        "positions": prefix + "_positions.csv",
    }


def find_time_col(df):
    for c in ["time", "datetime", "timestamp", "fill_time", "decision_time", "date"]:
        if c in df.columns:
            return c
    return None


def plot_line(df, x_col, y_col, title, out_path):
    if not HAS_PLOT:
        return
    if df is None or len(df) == 0 or y_col not in df.columns:
        return
    plt.figure(figsize=(12, 5))
    if x_col is not None and x_col in df.columns:
        plt.plot(df[x_col], df[y_col])
        plt.xlabel(x_col)
    else:
        plt.plot(df[y_col].values)
        plt.xlabel("index")
    plt.ylabel(y_col)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def build_daily_curve(daily_df):
    df = daily_df.copy()
    if "date" in df.columns:
        df = df.sort_values("date")
    if "daily_pnl" not in df.columns:
        return pd.DataFrame()
    df["daily_pnl"] = pd.to_numeric(df["daily_pnl"], errors="coerce")
    df["cum_pnl"] = df["daily_pnl"].cumsum()
    df["running_max_cum_pnl"] = df["cum_pnl"].cummax()
    df["drawdown"] = df["cum_pnl"] - df["running_max_cum_pnl"]
    if "turnover" in df.columns:
        df["turnover"] = pd.to_numeric(df["turnover"], errors="coerce")
        df["daily_return_bps_calc"] = df["daily_pnl"] / df["turnover"].replace(0, np.nan) * 10000.0
    if "max_gross_exposure" in df.columns:
        df["max_gross_exposure"] = pd.to_numeric(df["max_gross_exposure"], errors="coerce")
        df["daily_return_on_max_gross"] = df["daily_pnl"] / df["max_gross_exposure"].replace(0, np.nan)
    return df


def build_intraday_curve(curve_df):
    df = curve_df.copy()
    if len(df) == 0:
        return pd.DataFrame()
    time_col = find_time_col(df)
    if time_col is not None:
        if time_col != "date":
            df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        df = df.sort_values(time_col)
    value_col = None
    for c in ["equity", "total_equity", "portfolio_equity", "cum_pnl", "pnl", "total_pnl", "curve_value"]:
        if c in df.columns:
            value_col = c
            break
    if value_col is None:
        return pd.DataFrame()
    df["curve_value"] = pd.to_numeric(df[value_col], errors="coerce")
    df["running_max"] = df["curve_value"].cummax()
    df["drawdown"] = df["curve_value"] - df["running_max"]
    return df


def read_csv_if_exists(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


def analyze_one(summary_path, portfolio_dir, out_dir, annualization, make_plots):
    model_key = infer_model_key(summary_path)
    paths = related_paths(portfolio_dir, model_key)
    summary_df = pd.read_csv(summary_path)
    if len(summary_df) == 0:
        return None
    summary_row = summary_df.iloc[0].to_dict()
    daily_df = read_csv_if_exists(paths["daily"])
    curve_df = read_csv_if_exists(paths["curve"])
    positions_df = read_csv_if_exists(paths["positions"])
    curves_dir = os.path.join(out_dir, "curves")
    charts_dir = os.path.join(out_dir, "charts")
    os.makedirs(curves_dir, exist_ok=True)
    os.makedirs(charts_dir, exist_ok=True)
    result = {
        "model_key": model_key,
        "summary_file": summary_path,
        "daily_file": paths["daily"] if os.path.exists(paths["daily"]) else "",
        "curve_file": paths["curve"] if os.path.exists(paths["curve"]) else "",
        "positions_file": paths["positions"] if os.path.exists(paths["positions"]) else "",
    }
    summary_cols = [
        "model", "num_trades", "num_securities", "total_pnl", "total_turnover",
        "pnl_bps_on_turnover", "max_gross_exposure", "return_on_max_gross_exposure",
        "max_abs_net_exposure", "max_drawdown", "final_cash", "final_inventory_value",
        "final_equity", "num_long_symbols_end", "num_short_symbols_end", "num_short_events",
        "num_short_violations_if_no_short",
    ]
    for col in summary_cols:
        if col in summary_row:
            result[col] = summary_row[col]
    if len(daily_df) > 0:
        daily_curve = build_daily_curve(daily_df)
        if len(daily_curve) > 0:
            daily_curve["model_key"] = model_key
            daily_curve_path = os.path.join(curves_dir, model_key + "_daily_curve.csv")
            daily_curve.to_csv(daily_curve_path, index=False)
            result["daily_curve_file"] = daily_curve_path
            daily_pnl = pd.to_numeric(daily_curve["daily_pnl"], errors="coerce").dropna()
            result["num_days"] = len(daily_pnl)
            result["positive_days"] = int((daily_pnl > 0).sum())
            result["negative_days"] = int((daily_pnl < 0).sum())
            result["positive_day_ratio"] = safe_div(result["positive_days"], result["num_days"])
            result["avg_daily_pnl"] = daily_pnl.mean()
            result["median_daily_pnl"] = daily_pnl.median()
            result["std_daily_pnl"] = daily_pnl.std(ddof=1) if len(daily_pnl) >= 2 else np.nan
            result["min_daily_pnl"] = daily_pnl.min()
            result["max_daily_pnl"] = daily_pnl.max()
            result["daily_pnl_sharpe"] = calc_sharpe(daily_pnl, annualization)
            result["daily_cum_pnl_max_drawdown"] = calc_max_drawdown(daily_curve["cum_pnl"])
            if "turnover" in daily_curve.columns:
                result["avg_daily_turnover"] = daily_curve["turnover"].mean()
                result["total_turnover_from_daily"] = daily_curve["turnover"].sum()
            ret_col = None
            for col in ["daily_return_on_turnover_bps", "daily_return_bps_calc"]:
                if col in daily_curve.columns:
                    ret_col = col
                    break
            if ret_col is not None:
                daily_return = pd.to_numeric(daily_curve[ret_col], errors="coerce").dropna()
                result["avg_daily_return_bps"] = daily_return.mean()
                result["median_daily_return_bps"] = daily_return.median()
                result["daily_return_bps_sharpe"] = calc_sharpe(daily_return, annualization)
            if "max_gross_exposure" in daily_curve.columns:
                result["avg_daily_max_gross_exposure"] = daily_curve["max_gross_exposure"].mean()
                result["max_daily_max_gross_exposure"] = daily_curve["max_gross_exposure"].max()
            if "max_drawdown" in daily_curve.columns:
                result["worst_daily_drawdown"] = pd.to_numeric(daily_curve["max_drawdown"], errors="coerce").min()
            if make_plots:
                x_col = "date" if "date" in daily_curve.columns else None
                plot_line(daily_curve, x_col, "cum_pnl", "Daily Cumulative PnL - " + model_key, os.path.join(charts_dir, model_key + "_daily_cum_pnl.png"))
                plot_line(daily_curve, x_col, "drawdown", "Daily Drawdown - " + model_key, os.path.join(charts_dir, model_key + "_daily_drawdown.png"))
    if len(curve_df) > 0:
        intraday_curve = build_intraday_curve(curve_df)
        if len(intraday_curve) > 0:
            intraday_curve["model_key"] = model_key
            intraday_curve_path = os.path.join(curves_dir, model_key + "_intraday_curve.csv")
            intraday_curve.to_csv(intraday_curve_path, index=False)
            result["intraday_curve_file"] = intraday_curve_path
            result["intraday_curve_max_drawdown"] = intraday_curve["drawdown"].min()
            if make_plots:
                x_col = find_time_col(intraday_curve)
                plot_line(intraday_curve, x_col, "curve_value", "Intraday Equity / PnL Curve - " + model_key, os.path.join(charts_dir, model_key + "_intraday_curve.png"))
                plot_line(intraday_curve, x_col, "drawdown", "Intraday Drawdown - " + model_key, os.path.join(charts_dir, model_key + "_intraday_drawdown.png"))
    if len(positions_df) > 0:
        result["num_position_rows"] = len(positions_df)
        for col in ["ending_value", "abs_ending_value"]:
            if col in positions_df.columns:
                values = pd.to_numeric(positions_df[col], errors="coerce")
                result["total_" + col] = values.sum()
                result["max_" + col] = values.max()
    return result


def select_summary_files(args):
    portfolio_dir = args.portfolio_dir
    if args.list:
        paths = sorted(glob.glob(os.path.join(portfolio_dir, "portfolio_replay_*_summary.csv")))
        print("available models:")
        for path in paths:
            print(infer_model_key(path))
        raise SystemExit
    if args.summary:
        return args.summary
    if args.model_key:
        keys = []
        for item in args.model_key:
            keys.extend([x.strip() for x in item.split(",") if x.strip()])
        paths = []
        for key in keys:
            path = os.path.join(portfolio_dir, "portfolio_replay_" + key + "_summary.csv")
            if not os.path.exists(path):
                raise FileNotFoundError("summary not found for model_key=" + key + ": " + path)
            paths.append(path)
        return paths
    if args.pattern:
        return sorted(glob.glob(os.path.join(portfolio_dir, args.pattern)))
    if args.all:
        return sorted(glob.glob(os.path.join(portfolio_dir, "portfolio_replay_*_summary.csv")))
    raise RuntimeError("Please specify one of: --all, --model-key, --summary, --pattern, --list")


def write_output(results, out_dir):
    out = pd.DataFrame(results)
    preferred_cols = [
        "model_key", "model", "num_trades", "num_days", "num_securities", "total_pnl",
        "total_turnover", "pnl_bps_on_turnover", "positive_day_ratio", "avg_daily_pnl",
        "std_daily_pnl", "daily_pnl_sharpe", "avg_daily_return_bps", "daily_return_bps_sharpe",
        "max_gross_exposure", "return_on_max_gross_exposure", "max_abs_net_exposure",
        "max_drawdown", "daily_cum_pnl_max_drawdown", "intraday_curve_max_drawdown",
        "worst_daily_drawdown", "num_long_symbols_end", "num_short_symbols_end", "num_short_events",
        "num_short_violations_if_no_short", "summary_file", "daily_curve_file", "intraday_curve_file",
    ]
    cols = [col for col in preferred_cols if col in out.columns]
    other_cols = [col for col in out.columns if col not in cols]
    out = out[cols + other_cols]
    if "total_pnl" in out.columns:
        out = out.sort_values("total_pnl", ascending=False)
    if len(out) == 1:
        model_key = str(out.iloc[0]["model_key"])
        out_path = os.path.join(out_dir, model_key + "_performance_summary.csv")
    else:
        out_path = os.path.join(out_dir, "backtest_performance_summary.csv")
    out.to_csv(out_path, index=False)
    pd.set_option("display.max_columns", 160)
    pd.set_option("display.width", 300)
    print("\n===== performance summary =====")
    print(out.to_string(index=False))
    print("\nsaved:", out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--model-key", nargs="*")
    parser.add_argument("--summary", nargs="*")
    parser.add_argument("--pattern", default="")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--portfolio-dir", default="outputs/portfolio")
    parser.add_argument("--out-dir", default="outputs/metrics/performance_pack")
    parser.add_argument("--annualization", type=float, default=252.0)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    summary_files = select_summary_files(args)
    if len(summary_files) == 0:
        print("No summary files selected.")
        return
    results = []
    for summary_path in summary_files:
        print("analyzing:", summary_path)
        row = analyze_one(
            summary_path=summary_path,
            portfolio_dir=args.portfolio_dir,
            out_dir=args.out_dir,
            annualization=args.annualization,
            make_plots=(not args.no_plots),
        )
        if row is not None:
            results.append(row)
    if len(results) == 0:
        print("No valid results.")
        return
    write_output(results, args.out_dir)
    if (not HAS_PLOT) and (not args.no_plots):
        print("matplotlib unavailable; chart files were not generated.")


if __name__ == "__main__":
    main()
