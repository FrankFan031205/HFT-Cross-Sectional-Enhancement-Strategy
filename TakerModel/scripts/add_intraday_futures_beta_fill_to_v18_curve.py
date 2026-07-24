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
    return float((1 + s).prod() - 1)


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


def load_futures_ret(path):
    f = pd.read_csv(path) if str(path).endswith(".csv") else pd.read_parquet(path)
    dt_col = pick_col(f, ["datetime", "ts_real", "time", "trade_time"], "futures datetime")
    f["datetime"] = pd.to_datetime(f[dt_col])
    f = f.sort_values("datetime").drop_duplicates("datetime", keep="last")
    f["date"] = f["datetime"].dt.strftime("%Y%m%d").astype(int)

    ret_col = pick_col(f, ["futures_ret", "ret", "return"], "futures ret", required=False)
    if ret_col is not None:
        f["futures_ret"] = pd.to_numeric(f[ret_col], errors="coerce").fillna(0.0)
    else:
        px_col = pick_col(
            f,
            ["close", "lastprice", "last_price", "price", "mid_price", "tmid"],
            "futures price",
        )
        f["px"] = pd.to_numeric(f[px_col], errors="coerce")
        f["futures_ret"] = f.groupby("date")["px"].pct_change().fillna(0.0)

    # 不持有隔夜 futures，日内第一根 futures_ret 设为 0
    f.loc[f.groupby("date").cumcount() == 0, "futures_ret"] = 0.0
    return f[["datetime", "futures_ret"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curve-csv", required=True)
    ap.add_argument("--positions-csv", required=True)
    ap.add_argument("--futures-csv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--capital", type=float, default=200000000.0)
    ap.add_argument("--target-total-exposure", type=float, default=1.0)
    ap.add_argument("--max-overlay-gross", type=float, default=0.15)
    ap.add_argument("--futures-fee-rate", type=float, default=0.000023)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    curve = pd.read_csv(args.curve_csv)
    curve["datetime"] = pd.to_datetime(curve["datetime"])
    curve = curve.sort_values("datetime").reset_index(drop=True)
    curve["date"] = curve["datetime"].dt.strftime("%Y%m%d").astype(int)

    pos = pd.read_csv(args.positions_csv)
    pos["datetime"] = pd.to_datetime(pos["datetime"])
    pos["gross"] = (
        pd.to_numeric(pos["actual_shares_after"], errors="coerce").fillna(0.0).abs()
        * pd.to_numeric(pos["mid_price"], errors="coerce").fillna(0.0)
        / float(args.capital)
    )
    gross = pos.groupby("datetime", as_index=False)["gross"].sum().sort_values("datetime")

    x = pd.merge_asof(
        curve.sort_values("datetime"),
        gross.sort_values("datetime"),
        on="datetime",
        direction="backward",
    )
    x["stock_gross"] = x["gross"].ffill().fillna(0.0)
    x = x.drop(columns=["gross"], errors="ignore")

    fut = load_futures_ret(args.futures_csv)
    x = x.merge(fut, on="datetime", how="left")
    x["futures_ret"] = x["futures_ret"].fillna(0.0)

    # overlay 用上一分钟持有；第一根不持有，避免隔夜 futures
    x["target_futures_overlay_gross"] = (
        float(args.target_total_exposure) - x["stock_gross"]
    ).clip(lower=0.0, upper=float(args.max_overlay_gross))

    x["futures_overlay_prev_gross"] = x.groupby("date")["target_futures_overlay_gross"].shift(1).fillna(0.0)

    x["overlay_turnover"] = (
        x.groupby("date")["target_futures_overlay_gross"].diff().abs().fillna(x["target_futures_overlay_gross"].abs())
    )
    # 第一根不建 futures overnight；把第一根 turnover 也设 0
    x.loc[x.groupby("date").cumcount() == 0, "overlay_turnover"] = 0.0

    x["futures_fee_ret"] = x["overlay_turnover"] * float(args.futures_fee_rate)

    x["actual_ret_v18_futures"] = (
        pd.to_numeric(x["actual_ret_with_overnight"], errors="coerce").fillna(0.0)
        + x["futures_overlay_prev_gross"] * x["futures_ret"]
        - x["futures_fee_ret"]
    )

    x["benchmark_ret_v18_futures"] = pd.to_numeric(x["benchmark_ret_with_overnight"], errors="coerce").fillna(0.0)

    x["actualret_v18_futures"] = (1 + x["actual_ret_v18_futures"]).cumprod() - 1
    x["benchmarkret_v18_futures"] = (1 + x["benchmark_ret_v18_futures"]).cumprod() - 1
    x["alpharet_v18_futures"] = x["actualret_v18_futures"] - x["benchmarkret_v18_futures"]

    summary = pd.DataFrame([{
        "tag": args.tag,
        "actual_return": float(x["actualret_v18_futures"].iloc[-1]),
        "benchmark_return": float(x["benchmarkret_v18_futures"].iloc[-1]),
        "alpha_return": float(x["alpharet_v18_futures"].iloc[-1]),
        "daily_excess_sharpe": daily_sharpe(x, "actual_ret_v18_futures", "benchmark_ret_v18_futures"),
        "avg_stock_gross": float(x["stock_gross"].mean()),
        "avg_futures_overlay_gross": float(x["target_futures_overlay_gross"].mean()),
        "avg_total_exposure": float((x["stock_gross"] + x["target_futures_overlay_gross"]).mean()),
        "avg_overlay_turnover": float(x["overlay_turnover"].mean()),
        "total_futures_fee_ret": float(x["futures_fee_ret"].sum()),
        "target_total_exposure": float(args.target_total_exposure),
        "max_overlay_gross": float(args.max_overlay_gross),
        "futures_fee_rate": float(args.futures_fee_rate),
    }])

    curve_out = out / f"{args.tag}_curve.csv"
    summary_out = out / f"{args.tag}_summary.csv"
    png_out = out / f"{args.tag}_nav.png"

    x.to_csv(curve_out, index=False)
    summary.to_csv(summary_out, index=False)

    x["bar_index"] = np.arange(len(x))
    first_idx = x.groupby("date")["bar_index"].first()
    tick_dates = list(first_idx.index)[::2]
    tick_pos = [int(first_idx.loc[d]) for d in tick_dates]

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(x["bar_index"], x["actualret_v18_futures"] * 100, label="strategy actualret")
    ax.plot(x["bar_index"], x["benchmarkret_v18_futures"] * 100, label="CSI2000 benchmarkret")
    ax.plot(x["bar_index"], x["alpharet_v18_futures"] * 100, label="alpharet")
    ax.axhline(0, linewidth=1, linestyle="--")
    ax.set_title("V18 + Intraday CSI1000 Futures Beta Fill")
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
    print("summary:", summary_out)
    print("png:", png_out)


if __name__ == "__main__":
    main()
