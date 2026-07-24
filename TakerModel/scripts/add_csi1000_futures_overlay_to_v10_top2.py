# -*- coding: utf-8 -*-
from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/mnt/data1/fwz/HFT_010-dev_fwz_outputs/final_report")
FUTURES_CSV = Path("/mnt/data1/fwz/HFT_010-dev_fwz_outputs/futures/csi1000_im_minute_20241217_20250114.csv")
OUT_ROOT = ROOT / "v10_top2_with_csi1000_futures_overlay"

CANDIDATES = {
    "pure_cs_v10_mix_406000_csi2000_warmstart_noovernight": (
        ROOT
        / "v10_many_horizon_mix_csi2000_warmstart_noovernight"
        / "pure_cs_v10_mix_406000_csi2000_warmstart_noovernight"
        / "pure_cs_v10_mix_406000_csi2000_warmstart_noovernight_nav_curve_benchmark_warmstart_noovernight.csv"
    ),
    "pure_cs_v10_mix_007030_csi2000_warmstart_noovernight": (
        ROOT
        / "v10_many_horizon_mix_csi2000_warmstart_noovernight"
        / "pure_cs_v10_mix_007030_csi2000_warmstart_noovernight"
        / "pure_cs_v10_mix_007030_csi2000_warmstart_noovernight_nav_curve_benchmark_warmstart_noovernight.csv"
    ),
}

FUTURES_FEE_RATE = 0.000023   # 万分之0.23
FUTURES_LEVERAGE = 10.0       # 10x


def pick_existing(cols, candidates):
    s = set(cols)
    for c in candidates:
        if c in s:
            return c
    return None


def compound_curve(x):
    x = pd.Series(x).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return (1.0 + x).cumprod() - 1.0


def compound_return(x):
    x = pd.Series(x).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return float((1.0 + x).prod() - 1.0)


def daily_sharpe(daily_alpha):
    s = pd.Series(daily_alpha).dropna()
    if len(s) < 2:
        return np.nan
    sd = s.std(ddof=1)
    if not np.isfinite(sd) or sd == 0:
        return np.nan
    return float(s.mean() / sd * np.sqrt(252.0))


def get_step_return(df, step_candidates, cum_candidates, label):
    step_col = pick_existing(df.columns, step_candidates)
    if step_col is not None:
        s = pd.to_numeric(df[step_col], errors="coerce").fillna(0.0)
        return s, step_col, "step"

    cum_col = pick_existing(df.columns, cum_candidates)
    if cum_col is not None:
        c = pd.to_numeric(df[cum_col], errors="coerce").fillna(0.0)
        prev = (1.0 + c).shift(1)
        step = (1.0 + c) / prev - 1.0
        if len(step) > 0:
            step.iloc[0] = c.iloc[0]
        step = step.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return step, cum_col, "cum->step"

    raise KeyError(f"[{label}] cannot find return columns in {list(df.columns)}")


def load_stock_curve(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "datetime" not in df.columns:
        raise KeyError(f"{path} missing datetime; cols={list(df.columns)}")

    df["datetime"] = pd.to_datetime(df["datetime"])
    if "date" not in df.columns:
        df["date"] = df["datetime"].dt.strftime("%Y%m%d").astype(int)
    else:
        df["date"] = pd.to_numeric(df["date"], errors="coerce").astype("Int64")

    df = df.sort_values(["datetime"]).reset_index(drop=True)

    stock_ret, stock_src, stock_mode = get_step_return(
        df,
        step_candidates=["actual_ret", "actualret_step", "strategy_ret", "strategyret_step", "stock_ret"],
        cum_candidates=["nav_actualret", "actualret", "strategyret", "nav_strategyret"],
        label="stock actual return",
    )
    benchmark_ret, bench_src, bench_mode = get_step_return(
        df,
        step_candidates=["benchmark_ret", "benchmarkret_step", "bench_ret"],
        cum_candidates=["nav_benchmarkret", "benchmarkret", "nav_benchret"],
        label="benchmark return",
    )

    gross_col = pick_existing(
        df.columns,
        [
            "gross_prev_to_equity",
            "avg_gross_prev_to_equity",
            "gross_after_to_equity",
            "avg_gross_after_to_equity",
            "gross_prev_to_capital",
            "avg_gross_prev_to_capital",
            "actual_gross",
            "held_gross_prev",
            "gross",
        ],
    )
    if gross_col is None:
        raise KeyError(f"cannot find gross column in {list(df.columns)}")

    out = pd.DataFrame(
        {
            "date": df["date"].astype(int),
            "datetime": df["datetime"],
            "stock_ret": pd.to_numeric(stock_ret, errors="coerce").fillna(0.0),
            "benchmark_ret": pd.to_numeric(benchmark_ret, errors="coerce").fillna(0.0),
            "stock_gross": pd.to_numeric(df[gross_col], errors="coerce").fillna(0.0).clip(lower=0.0),
        }
    )

    out["stock_gross"] = out["stock_gross"].clip(upper=1.20)

    print(f"[load_stock_curve] {path.name}")
    print(f"  stock return source     : {stock_src} ({stock_mode})")
    print(f"  benchmark return source : {bench_src} ({bench_mode})")
    print(f"  gross source            : {gross_col}")
    print(f"  rows                    : {len(out)}")
    print(f"  dt range                : {out['datetime'].min()} -> {out['datetime'].max()}")

    return out


def load_futures(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "datetime" not in df.columns:
        raise KeyError(f"{path} missing datetime; cols={list(df.columns)}")
    df["datetime"] = pd.to_datetime(df["datetime"])
    ret_col = pick_existing(df.columns, ["futures_ret", "ret", "return"])
    if ret_col is None:
        price_col = pick_existing(df.columns, ["futures_price", "last_price", "lastprice", "price", "close", "mid_price", "tmid"])
        if price_col is None:
            raise KeyError(f"cannot find futures_ret or price columns; cols={list(df.columns)}")
        px = pd.to_numeric(df[price_col], errors="coerce")
        fut_ret = px.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    else:
        fut_ret = pd.to_numeric(df[ret_col], errors="coerce").fillna(0.0)

    out = df[["datetime"]].copy()
    out["futures_ret"] = fut_ret
    out = out.sort_values("datetime").drop_duplicates("datetime", keep="last").reset_index(drop=True)

    print(f"[load_futures] {path.name}")
    print(f"  rows     : {len(out)}")
    print(f"  dt range : {out['datetime'].min()} -> {out['datetime'].max()}")

    return out


def build_overlay_curve(stock_df: pd.DataFrame, fut_df: pd.DataFrame) -> pd.DataFrame:
    df = stock_df.merge(fut_df, on="datetime", how="left")
    df["futures_ret"] = df["futures_ret"].fillna(0.0)

    # 用股票剩余现金补期货多头：overlay = max(0, 1 - stock_gross)
    df["target_futures_overlay_gross"] = (1.0 - df["stock_gross"]).clip(lower=0.0, upper=1.0)
    df["margin_used"] = df["target_futures_overlay_gross"] / FUTURES_LEVERAGE

    prev_overlay = 0.0
    fut_pnl_list = []
    fut_fee_list = []
    fut_turnover_list = []
    overlay_prev_list = []

    for row in df.itertuples(index=False):
        overlay_prev_list.append(prev_overlay)

        fut_pnl = prev_overlay * float(row.futures_ret)
        target_overlay = float(row.target_futures_overlay_gross)
        fut_turnover = abs(target_overlay - prev_overlay)
        fut_fee = fut_turnover * FUTURES_FEE_RATE

        fut_pnl_list.append(fut_pnl)
        fut_fee_list.append(fut_fee)
        fut_turnover_list.append(fut_turnover)

        prev_overlay = target_overlay

    df["futures_overlay_prev_gross"] = overlay_prev_list
    df["futures_pnl_ret"] = fut_pnl_list
    df["futures_fee_ret"] = fut_fee_list
    df["futures_turnover_gross"] = fut_turnover_list

    df["actual_ret_with_futures"] = (
        df["stock_ret"] + df["futures_pnl_ret"] - df["futures_fee_ret"]
    )
    df["alpha_step_with_futures"] = df["actual_ret_with_futures"] - df["benchmark_ret"]

    df["nav_actualret_with_futures"] = compound_curve(df["actual_ret_with_futures"])
    df["nav_benchmarkret"] = compound_curve(df["benchmark_ret"])
    df["nav_alpharet_with_futures"] = compound_curve(df["alpha_step_with_futures"])

    return df


def make_daily(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.groupby("date", as_index=False)
        .agg(
            start_datetime=("datetime", "min"),
            end_datetime=("datetime", "max"),
            stock_ret=("stock_ret", compound_return),
            benchmark_ret=("benchmark_ret", compound_return),
            futures_pnl_ret=("futures_pnl_ret", "sum"),
            futures_fee_ret=("futures_fee_ret", "sum"),
            actual_ret_with_futures=("actual_ret_with_futures", compound_return),
            alpha_step_with_futures=("alpha_step_with_futures", compound_return),
            avg_stock_gross=("stock_gross", "mean"),
            avg_futures_overlay_gross=("target_futures_overlay_gross", "mean"),
            avg_margin_used=("margin_used", "mean"),
            total_futures_turnover_gross=("futures_turnover_gross", "sum"),
        )
    )
    return out


def make_summary(df: pd.DataFrame, daily: pd.DataFrame, version: str) -> pd.DataFrame:
    summary = {
        "version": version,
        "start_date": int(df["date"].min()),
        "end_date": int(df["date"].max()),
        "start_datetime": str(df["datetime"].min()),
        "end_datetime": str(df["datetime"].max()),
        "benchmark_name": "CSI2000 benchmark (stock benchmark), CSI1000 futures overlay",
        "actual_return_stock_plus_futures": compound_return(df["actual_ret_with_futures"]),
        "benchmark_return": compound_return(df["benchmark_ret"]),
        "alpha_return_stock_plus_futures": compound_return(df["alpha_step_with_futures"]),
        "actual_return_stock_only": compound_return(df["stock_ret"]),
        "alpha_return_stock_only": compound_return(df["stock_ret"] - df["benchmark_ret"]),
        "daily_excess_sharpe_stock_plus_futures": daily_sharpe(daily["alpha_step_with_futures"]),
        "avg_stock_gross": float(df["stock_gross"].mean()),
        "avg_futures_overlay_target_gross": float(df["target_futures_overlay_gross"].mean()),
        "avg_futures_overlay_prev_gross": float(df["futures_overlay_prev_gross"].mean()),
        "avg_margin_used": float(df["margin_used"].mean()),
        "total_futures_turnover_gross": float(df["futures_turnover_gross"].sum()),
        "total_futures_fee_return": float(df["futures_fee_ret"].sum()),
        "total_futures_pnl_return": float(df["futures_pnl_ret"].sum()),
        "futures_fee_rate": FUTURES_FEE_RATE,
        "futures_leverage": FUTURES_LEVERAGE,
        "n_minutes": int(len(df)),
        "n_days": int(df["date"].nunique()),
    }
    return pd.DataFrame([summary])


def pct(x):
    if pd.isna(x):
        return "NA"
    return f"{100.0 * float(x):.2f}%"


def num(x):
    if pd.isna(x):
        return "NA"
    return f"{float(x):.2f}"


def plot_report(df: pd.DataFrame, summary_row: pd.Series, out_png: Path, title_suffix: str):
    fig = plt.figure(figsize=(16, 9), dpi=160)
    ax = fig.add_axes([0.07, 0.12, 0.66, 0.78])

    x = np.arange(len(df))
    ax.plot(x, 100.0 * df["nav_actualret_with_futures"], label="actualret with futures", linewidth=1.5)
    ax.plot(x, 100.0 * df["nav_benchmarkret"], label="benchmarkret", linewidth=1.5)
    ax.plot(x, 100.0 * df["nav_alpharet_with_futures"], label="alpharet", linewidth=1.7)

    ax.axhline(0.0, linestyle="--", linewidth=0.8, alpha=0.8)

    xticks_idx = np.linspace(0, max(len(df) - 1, 0), 10, dtype=int)
    xticks_idx = np.unique(xticks_idx)
    xtick_labels = df.iloc[xticks_idx]["datetime"].dt.strftime("%Y%m%d").tolist()
    ax.set_xticks(xticks_idx)
    ax.set_xticklabels(xtick_labels, rotation=45, ha="right")

    ax.set_title(f"T+1-aware Pure-CS NAV with CSI1000 Futures Overlay\n{title_suffix}", fontsize=13, weight="bold")
    ax.set_xlabel("trading minute index, benchmark warm-start, no overnight PnL")
    ax.set_ylabel("cumulative return (%)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower left", fontsize=9)

    text = "\n".join(
        [
            "Summary",
            "",
            f"Strategy Return",
            pct(summary_row["actual_return_stock_plus_futures"]),
            "",
            f"Benchmark Return",
            pct(summary_row["benchmark_return"]),
            "",
            f"Alpha Return",
            pct(summary_row["alpha_return_stock_plus_futures"]),
            "",
            f"Daily Excess Sharpe",
            num(summary_row["daily_excess_sharpe_stock_plus_futures"]),
            "",
            f"Avg Stock Gross",
            pct(summary_row["avg_stock_gross"]),
            "",
            f"Avg Futures Overlay",
            pct(summary_row["avg_futures_overlay_target_gross"]),
        ]
    )

    fig.text(
        0.77, 0.62, text,
        fontsize=10.5,
        va="center",
        ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="0.6")
    )

    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def run_one(version: str, curve_path: Path, fut_df: pd.DataFrame):
    out_dir = OUT_ROOT / version
    out_dir.mkdir(parents=True, exist_ok=True)

    stock_df = load_stock_curve(curve_path)
    curve = build_overlay_curve(stock_df, fut_df)
    daily = make_daily(curve)
    summary = make_summary(curve, daily, version)

    curve_csv = out_dir / f"{version}_curve_with_csi1000_futures_overlay.csv"
    daily_csv = out_dir / f"{version}_daily_with_csi1000_futures_overlay.csv"
    summary_csv = out_dir / f"{version}_summary_with_csi1000_futures_overlay.csv"
    png = out_dir / f"{version}_nav_with_csi1000_futures_overlay.png"

    curve.to_csv(curve_csv, index=False)
    daily.to_csv(daily_csv, index=False)
    summary.to_csv(summary_csv, index=False)

    title_map = {
        "pure_cs_v10_mix_406000_csi2000_warmstart_noovernight": "Final: 40% h10 + 60% h20",
        "pure_cs_v10_mix_007030_csi2000_warmstart_noovernight": "Stable: 70% h20 + 30% h30",
    }
    plot_report(curve, summary.iloc[0], png, title_map.get(version, version))

    print(f"\n===== DONE {version} =====")
    print(summary.T.to_string(header=False))
    print(f"[saved curve ] {curve_csv}")
    print(f"[saved daily ] {daily_csv}")
    print(f"[saved summ  ] {summary_csv}")
    print(f"[saved png   ] {png}")

    return summary


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    fut_df = load_futures(FUTURES_CSV)

    all_summary = []
    for version, curve_path in CANDIDATES.items():
        if not curve_path.exists():
            raise FileNotFoundError(f"missing curve file: {curve_path}")
        summ = run_one(version, curve_path, fut_df)
        all_summary.append(summ)

    ranked = pd.concat(all_summary, ignore_index=True)
    ranked["score"] = (
        ranked["alpha_return_stock_plus_futures"] * 100.0
        + 0.2 * ranked["daily_excess_sharpe_stock_plus_futures"]
    )
    ranked = ranked.sort_values(
        ["score", "alpha_return_stock_plus_futures", "daily_excess_sharpe_stock_plus_futures"],
        ascending=False
    ).reset_index(drop=True)

    ranked_csv = OUT_ROOT / "top2_with_csi1000_futures_overlay_summary_ranked.csv"
    ranked.to_csv(ranked_csv, index=False)

    print("\n===== TOP2 WITH CSI1000 FUTURES OVERLAY RANKING =====")
    show_cols = [
        "version",
        "actual_return_stock_plus_futures",
        "benchmark_return",
        "alpha_return_stock_plus_futures",
        "daily_excess_sharpe_stock_plus_futures",
        "avg_stock_gross",
        "avg_futures_overlay_target_gross",
        "total_futures_turnover_gross",
        "total_futures_fee_return",
        "score",
    ]
    print(ranked[show_cols].to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"\n[saved ranking] {ranked_csv}")


if __name__ == "__main__":
    main()
