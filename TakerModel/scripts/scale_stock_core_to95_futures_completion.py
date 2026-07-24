# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def pick_col(df, candidates, name, required=True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"cannot find {name}; candidates={candidates}; columns={df.columns.tolist()}")
    return None


def compound(s):
    s = pd.to_numeric(s, errors="coerce").fillna(0.0)
    return float((1.0 + s).prod() - 1.0)


def daily_sharpe(df, actual_col, bench_col):
    x = df.copy()
    x["date"] = pd.to_datetime(x["datetime"]).dt.strftime("%Y%m%d").astype(int)
    d = x.groupby("date").agg(
        actual_day=(actual_col, compound),
        bench_day=(bench_col, compound),
    )
    e = d["actual_day"] - d["bench_day"]
    if len(e) < 2 or e.std(ddof=1) == 0:
        return np.nan
    return float(e.mean() / e.std(ddof=1) * np.sqrt(252))


def daily_turnover(target, date, charge_open=True, charge_close=True):
    target = pd.to_numeric(target, errors="coerce").fillna(0.0)
    prev = target.groupby(date).shift(1).fillna(0.0)
    turn = (target - prev).abs()

    bar = target.groupby(date).cumcount()
    if not charge_open:
        turn.loc[bar == 0] = 0.0

    if charge_close:
        max_bar = bar.groupby(date).transform("max")
        turn.loc[bar == max_bar] += target.loc[bar == max_bar].abs()

    return turn


def make_daily_table(df, actual_col, bench_col):
    x = df.copy()
    x["date"] = pd.to_datetime(x["datetime"]).dt.strftime("%Y%m%d").astype(int)
    d = x.groupby("date").agg(
        actual=(actual_col, compound),
        bench=(bench_col, compound),
        avg_raw_stock_gross=("raw_stock_gross", "mean"),
        avg_scaled_stock_gross=("scaled_stock_gross", "mean"),
        avg_extra_stock_gross=("extra_stock_gross", "mean"),
        avg_futures=("futures_target_weight", "mean"),
        avg_total_exposure=("total_target_exposure", "mean"),
        extra_stock_cost=("extra_stock_cost_ret", "sum"),
        futures_fee=("futures_fee_ret", "sum"),
    )
    d["excess"] = d["actual"] - d["bench"]
    return d.reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-curve", required=True, help="v19 curve, contains v18 base + stock_gross + futures_ret")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", required=True)

    ap.add_argument("--target-stock-gross", type=float, default=0.95)
    ap.add_argument("--target-total-exposure", type=float, default=1.00)
    ap.add_argument("--max-extra-stock-gross", type=float, default=0.12)
    ap.add_argument("--max-futures-gross", type=float, default=0.08)

    ap.add_argument("--stock-cost-bps", type=float, default=10.0)
    ap.add_argument("--futures-fee-rate", type=float, default=0.000023)
    ap.add_argument("--charge-open-turnover", type=int, default=1)
    ap.add_argument("--charge-close-turnover", type=int, default=1)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    x = pd.read_csv(args.base_curve)
    x["datetime"] = pd.to_datetime(x["datetime"])
    x = x.sort_values("datetime").reset_index(drop=True)
    x["date"] = x["datetime"].dt.strftime("%Y%m%d").astype(int)
    x["bar_in_day"] = x.groupby("date").cumcount()

    # v18 基础曲线：有隔夜
    base_actual_with_overnight = pick_col(x, ["actual_ret_with_overnight"], "v18 actual with overnight")
    bench_col = pick_col(x, ["benchmark_ret_with_overnight"], "benchmark with overnight")

    # 原 v16/v18 日内股票核心收益，不能用 with_overnight，否则会重复放大隔夜
    intraday_stock_ret_col = pick_col(
        x,
        ["actual_ret", "stock_ret", "strategy_ret"],
        "intraday stock core return",
    )

    if "stock_gross" not in x.columns:
        raise KeyError("base curve must contain stock_gross")
    if "futures_ret" not in x.columns:
        raise KeyError("base curve must contain futures_ret")

    x["raw_stock_gross"] = pd.to_numeric(x["stock_gross"], errors="coerce").fillna(0.0)
    x["intraday_stock_ret"] = pd.to_numeric(x[intraday_stock_ret_col], errors="coerce").fillna(0.0)
    x["futures_ret"] = pd.to_numeric(x["futures_ret"], errors="coerce").fillna(0.0)

    # 不放大隔夜第一根
    x.loc[x["bar_in_day"] == 0, "intraday_stock_ret"] = 0.0

    # 把原 stock core 从 raw gross 按比例补到 target-stock-gross
    x["extra_stock_gross"] = (
        float(args.target_stock_gross) - x["raw_stock_gross"]
    ).clip(lower=0.0, upper=float(args.max_extra_stock_gross))

    x["scaled_stock_gross"] = x["raw_stock_gross"] + x["extra_stock_gross"]

    # 额外股票仓位按原组合同比例放大，所以额外收益约为：
    # extra_gross / raw_gross * 原日内股票组合收益
    x["extra_stock_prev_gross"] = x.groupby("date")["extra_stock_gross"].shift(1).fillna(0.0)
    x["raw_stock_prev_gross"] = x.groupby("date")["raw_stock_gross"].shift(1).fillna(0.0)
    x.loc[x["bar_in_day"] == 0, "extra_stock_prev_gross"] = 0.0
    x.loc[x["bar_in_day"] == 0, "raw_stock_prev_gross"] = 0.0

    x["extra_scale"] = np.where(
        x["raw_stock_prev_gross"] > 1e-8,
        x["extra_stock_prev_gross"] / x["raw_stock_prev_gross"],
        0.0,
    )
    x["extra_stock_ret"] = x["extra_scale"] * x["intraday_stock_ret"]

    charge_open = bool(args.charge_open_turnover)
    charge_close = bool(args.charge_close_turnover)

    x["extra_stock_turnover"] = daily_turnover(
        x["extra_stock_gross"],
        x["date"],
        charge_open=charge_open,
        charge_close=charge_close,
    )
    x["extra_stock_cost_ret"] = x["extra_stock_turnover"] * float(args.stock_cost_bps) / 10000.0

    # 期货只机械补剩余 exposure
    x["futures_target_weight"] = (
        float(args.target_total_exposure) - x["scaled_stock_gross"]
    ).clip(lower=0.0, upper=float(args.max_futures_gross))

    x["futures_prev_weight"] = x.groupby("date")["futures_target_weight"].shift(1).fillna(0.0)
    x.loc[x["bar_in_day"] == 0, "futures_prev_weight"] = 0.0

    x["futures_turnover"] = daily_turnover(
        x["futures_target_weight"],
        x["date"],
        charge_open=charge_open,
        charge_close=charge_close,
    )
    x["futures_fee_ret"] = x["futures_turnover"] * float(args.futures_fee_rate)
    x["futures_overlay_ret"] = x["futures_prev_weight"] * x["futures_ret"]

    x["actual_ret_v23c"] = (
        pd.to_numeric(x[base_actual_with_overnight], errors="coerce").fillna(0.0)
        + x["extra_stock_ret"]
        + x["futures_overlay_ret"]
        - x["extra_stock_cost_ret"]
        - x["futures_fee_ret"]
    )
    x["benchmark_ret_v23c"] = pd.to_numeric(x[bench_col], errors="coerce").fillna(0.0)

    x["actualret_v23c"] = (1.0 + x["actual_ret_v23c"]).cumprod() - 1.0
    x["benchmarkret_v23c"] = (1.0 + x["benchmark_ret_v23c"]).cumprod() - 1.0
    x["alpharet_v23c"] = x["actualret_v23c"] - x["benchmarkret_v23c"]

    x["total_target_exposure"] = x["scaled_stock_gross"] + x["futures_target_weight"]

    summary = pd.DataFrame([{
        "tag": args.tag,
        "actual_return": float(x["actualret_v23c"].iloc[-1]),
        "benchmark_return": float(x["benchmarkret_v23c"].iloc[-1]),
        "alpha_return": float(x["alpharet_v23c"].iloc[-1]),
        "daily_excess_sharpe": daily_sharpe(x, "actual_ret_v23c", "benchmark_ret_v23c"),

        "avg_raw_stock_gross": float(x["raw_stock_gross"].mean()),
        "avg_extra_stock_gross": float(x["extra_stock_gross"].mean()),
        "avg_scaled_stock_gross": float(x["scaled_stock_gross"].mean()),
        "avg_futures_gross": float(x["futures_target_weight"].mean()),
        "avg_total_exposure": float(x["total_target_exposure"].mean()),

        "avg_extra_scale": float(x["extra_scale"].mean()),
        "avg_extra_stock_turnover": float(x["extra_stock_turnover"].mean()),
        "avg_futures_turnover": float(x["futures_turnover"].mean()),
        "total_extra_stock_cost_ret": float(x["extra_stock_cost_ret"].sum()),
        "total_futures_fee_ret": float(x["futures_fee_ret"].sum()),

        "target_stock_gross": float(args.target_stock_gross),
        "target_total_exposure": float(args.target_total_exposure),
        "max_extra_stock_gross": float(args.max_extra_stock_gross),
        "max_futures_gross": float(args.max_futures_gross),
        "stock_cost_bps": float(args.stock_cost_bps),
        "futures_fee_rate": float(args.futures_fee_rate),
        "intraday_stock_ret_col": intraday_stock_ret_col,
        "charge_open_turnover": int(args.charge_open_turnover),
        "charge_close_turnover": int(args.charge_close_turnover),
    }])

    daily = make_daily_table(x, "actual_ret_v23c", "benchmark_ret_v23c")

    curve_out = out / f"{args.tag}_curve.csv"
    daily_out = out / f"{args.tag}_daily.csv"
    summary_out = out / f"{args.tag}_summary.csv"
    png_out = out / f"{args.tag}_nav.png"

    x.to_csv(curve_out, index=False)
    daily.to_csv(daily_out, index=False)
    summary.to_csv(summary_out, index=False)

    x["plot_index"] = np.arange(len(x))
    first_idx = x.groupby("date")["plot_index"].first()
    tick_dates = list(first_idx.index)[::2]
    tick_pos = [int(first_idx.loc[d]) for d in tick_dates]

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(x["plot_index"], x["actualret_v23c"] * 100.0, label="strategy actualret")
    ax.plot(x["plot_index"], x["benchmarkret_v23c"] * 100.0, label="CSI2000 benchmarkret")
    ax.plot(x["plot_index"], x["alpharet_v23c"] * 100.0, label="alpharet")
    ax.axhline(0, linewidth=1, linestyle="--")
    ax.set_title("V23c Scale Pure-CS Stock Core to 95% + Passive Futures Completion")
    ax.set_xlabel("trading minute index")
    ax.set_ylabel("cumulative return (%)")
    ax.set_xticks(tick_pos)
    ax.set_xticklabels([str(d) for d in tick_dates], rotation=45, ha="right")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left")

    s = summary.iloc[0]
    txt = (
        "Summary\n\n"
        f"Strategy Return\n{s['actual_return']*100:.2f}%\n\n"
        f"Benchmark Return\n{s['benchmark_return']*100:.2f}%\n\n"
        f"Alpha Return\n{s['alpha_return']*100:.2f}%\n\n"
        f"Daily Excess Sharpe\n{s['daily_excess_sharpe']:.2f}"
    )
    ax.text(1.05, 0.95, txt, transform=ax.transAxes, va="top", ha="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

    fig.tight_layout()
    fig.savefig(png_out, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(summary.T.to_string())
    print("curve:", curve_out)
    print("daily:", daily_out)
    print("summary:", summary_out)
    print("png:", png_out)


if __name__ == "__main__":
    main()
