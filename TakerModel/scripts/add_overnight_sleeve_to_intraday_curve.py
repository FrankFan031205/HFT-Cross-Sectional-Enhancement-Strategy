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
        raise KeyError(f"cannot find {name}. candidates={candidates}, columns={df.columns.tolist()}")
    return None


def norm_sid(df):
    x = df.copy()
    if "sid" in x.columns:
        x["sid_norm"] = pd.to_numeric(x["sid"], errors="coerce").astype("Int64")
    elif "securityid" in x.columns:
        x["sid_norm"] = pd.to_numeric(
            x["securityid"].astype(str).str.extract(r"(\d+)")[0],
            errors="coerce"
        ).astype("Int64")
    elif "SecurityID" in x.columns:
        x["sid_norm"] = pd.to_numeric(
            x["SecurityID"].astype(str).str.extract(r"(\d+)")[0],
            errors="coerce"
        ).astype("Int64")
    else:
        raise KeyError(f"cannot find sid/securityid columns: {x.columns.tolist()}")
    return x


def compound_ret(s):
    s = pd.to_numeric(s, errors="coerce").fillna(0.0)
    return float((1.0 + s).prod() - 1.0)


def daily_sharpe(df, actual_col, bench_col):
    x = df.copy()
    x["date_int"] = pd.to_datetime(x["datetime"]).dt.strftime("%Y%m%d").astype(int)
    d = x.groupby("date_int").agg(
        actual_day=(actual_col, compound_ret),
        bench_day=(bench_col, compound_ret),
    )
    e = d["actual_day"] - d["bench_day"]
    if len(e) < 2 or e.std(ddof=1) == 0:
        return np.nan
    return float(e.mean() / e.std(ddof=1) * np.sqrt(252))


def build_base_weights(pos_snap, capital):
    x = norm_sid(pos_snap)

    w_col = pick_col(
        x,
        [
            "actual_weight_after",
            "actual_weight",
            "weight_after",
            "position_weight",
            "target_weight",
            "weight",
        ],
        "position weight",
        required=False,
    )

    if w_col is not None:
        x["base_weight"] = pd.to_numeric(x[w_col], errors="coerce").fillna(0.0)
    else:
        sh_col = pick_col(
            x,
            [
                "actual_shares_after",
                "actual_shares",
                "target_shares",
                "shares",
                "position_shares",
            ],
            "shares",
        )
        px_col = pick_col(
            x,
            [
                "mid_price",
                "mark_price",
                "tmid",
                "price",
                "bid1",
                "ask1",
            ],
            "price",
        )
        x["base_weight"] = (
            pd.to_numeric(x[sh_col], errors="coerce").fillna(0.0)
            * pd.to_numeric(x[px_col], errors="coerce").fillna(0.0)
            / float(capital)
        )

    out = (
        x[["sid_norm", "base_weight"]]
        .dropna()
        .rename(columns={"sid_norm": "sid"})
        .groupby("sid", as_index=False)["base_weight"]
        .sum()
    )
    return out


def build_benchmark_weights(pos_snap):
    x = norm_sid(pos_snap)

    if "benchmark_weight" not in x.columns:
        raise KeyError(
            "positions snapshot has no benchmark_weight. "
            "当前脚本需要从 target_positions.csv 里取 benchmark_weight。"
        )

    x["benchmark_weight"] = pd.to_numeric(x["benchmark_weight"], errors="coerce").fillna(0.0)
    out = (
        x[["sid_norm", "benchmark_weight"]]
        .dropna()
        .rename(columns={"sid_norm": "sid"})
        .groupby("sid", as_index=False)["benchmark_weight"]
        .sum()
    )
    return out


def make_overlay(day_df, sleeve_gross, top_frac, bottom_frac, pred_col):
    x = day_df.copy()
    x["base_weight"] = pd.to_numeric(x["base_weight"], errors="coerce").fillna(0.0)
    x[pred_col] = pd.to_numeric(x[pred_col], errors="coerce").fillna(0.0)

    n = len(x)
    top_n = max(1, int(n * top_frac))
    bottom_n = max(1, int(n * bottom_frac))

    x["target_weight"] = x["base_weight"]
    x["overlay_buy_weight"] = 0.0
    x["overlay_sell_weight"] = 0.0

    top = x.sort_values(pred_col, ascending=False).head(top_n).copy()
    bottom = x[x["base_weight"] > 1e-12].sort_values(pred_col, ascending=True).head(bottom_n).copy()

    sell_budget = min(float(sleeve_gross), float(bottom["base_weight"].sum()))
    if sell_budget <= 1e-12:
        return x

    sell = bottom[["sid", "base_weight"]].copy()
    sell["overlay_sell_weight_new"] = sell_budget * sell["base_weight"] / sell["base_weight"].sum()

    buy = top[["sid"]].copy()
    buy["overlay_buy_weight_new"] = sell_budget / len(buy)

    x = x.merge(sell[["sid", "overlay_sell_weight_new"]], on="sid", how="left")
    x = x.merge(buy[["sid", "overlay_buy_weight_new"]], on="sid", how="left")

    x["overlay_sell_weight_new"] = x["overlay_sell_weight_new"].fillna(0.0)
    x["overlay_buy_weight_new"] = x["overlay_buy_weight_new"].fillna(0.0)

    x["overlay_sell_weight"] = x["overlay_sell_weight_new"]
    x["overlay_buy_weight"] = x["overlay_buy_weight_new"]
    x["target_weight"] = x["base_weight"] - x["overlay_sell_weight"] + x["overlay_buy_weight"]
    x["target_weight"] = x["target_weight"].clip(lower=0.0)

    return x.drop(columns=["overlay_sell_weight_new", "overlay_buy_weight_new"], errors="ignore")


def ts_to_trading_datetime(date_int, ts):
    # ts 是交易秒：09:30 开始，午休不计入；7200 秒后进入 13:00 午后
    day = pd.to_datetime(str(int(date_int)))
    ts = int(ts)
    if ts < 7200:
        return day + pd.Timedelta(hours=9, minutes=30, seconds=ts)
    return day + pd.Timedelta(hours=13, seconds=ts - 7200)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--intraday-curve", required=True)
    ap.add_argument("--positions-csv", required=True)
    ap.add_argument("--overnight-pred", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", required=True)

    ap.add_argument("--capital", type=float, default=200000000.0)
    ap.add_argument("--entry-ts", type=int, default=12600)
    ap.add_argument("--sleeve-gross", type=float, default=0.05)
    ap.add_argument("--top-frac", type=float, default=0.10)
    ap.add_argument("--bottom-frac", type=float, default=0.10)
    ap.add_argument("--pred-col", default="pred_z")
    ap.add_argument("--pred-mult", type=float, default=1.0)
    ap.add_argument("--ret-col", default="y_raw")
    ap.add_argument("--stock-cost-bps", type=float, default=10.0)
    ap.add_argument("--roundtrip-cost-mult", type=float, default=2.0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    curve = pd.read_csv(args.intraday_curve)
    curve["datetime"] = pd.to_datetime(curve["datetime"])
    curve = curve.sort_values("datetime").reset_index(drop=True)
    curve["date_int"] = curve["datetime"].dt.strftime("%Y%m%d").astype(int)

    actual_col = pick_col(curve, ["actual_ret", "stock_ret", "strategy_ret"], "actual return")
    bench_col = pick_col(curve, ["benchmark_ret", "bench_ret"], "benchmark return")

    pos = pd.read_csv(args.positions_csv)
    pos["datetime"] = pd.to_datetime(pos["datetime"])
    if "date" in pos.columns:
        pos["date_int"] = pd.to_numeric(pos["date"], errors="coerce").astype("Int64")
    else:
        pos["date_int"] = pos["datetime"].dt.strftime("%Y%m%d").astype(int)

    ov = pd.read_parquet(args.overnight_pred)
    ov["date_int"] = pd.to_numeric(ov["date"], errors="coerce").astype("Int64")
    ov["sid"] = pd.to_numeric(ov["sid"], errors="coerce").astype("Int64")
    ov["ts"] = pd.to_numeric(ov["ts"], errors="coerce").astype("Int64")

    if args.pred_col not in ov.columns:
        raise KeyError(f"overnight pred missing {args.pred_col}. columns={ov.columns.tolist()}")
    if args.ret_col not in ov.columns:
        raise KeyError(f"overnight pred missing {args.ret_col}. columns={ov.columns.tolist()}")

    ov[args.pred_col] = pd.to_numeric(ov[args.pred_col], errors="coerce").fillna(0.0) * float(args.pred_mult)

    ov = ov[ov["ts"] == int(args.entry_ts)].copy()
    if ov.empty:
        raise RuntimeError(f"no overnight rows at entry_ts={args.entry_ts}")

    dates = sorted(curve["date_int"].unique())
    next_date = {int(dates[i]): int(dates[i + 1]) for i in range(len(dates) - 1)}

    daily_rows = []

    for d in sorted(ov["date_int"].dropna().unique()):
        d = int(d)
        if d not in next_date:
            continue

        nd = next_date[d]
        day_pos = pos[pos["date_int"] == d].copy()
        if day_pos.empty:
            continue

        entry_dt = ts_to_trading_datetime(d, args.entry_ts)
        cand = day_pos[day_pos["datetime"] <= entry_dt]
        if cand.empty:
            cand = day_pos
        snap_dt = cand["datetime"].max()
        snap = cand[cand["datetime"] == snap_dt].copy()

        base_w = build_base_weights(snap, args.capital)
        bench_w = build_benchmark_weights(snap)

        z = ov[ov["date_int"] == d][["sid", args.pred_col, args.ret_col]].copy()
        z = z.rename(columns={args.ret_col: "overnight_ret"})

        x = z.merge(base_w, on="sid", how="left")
        x = x.merge(bench_w, on="sid", how="left")
        x["base_weight"] = x["base_weight"].fillna(0.0)
        x["benchmark_weight"] = x["benchmark_weight"].fillna(0.0)
        x["overnight_ret"] = pd.to_numeric(x["overnight_ret"], errors="coerce").fillna(0.0)

        x = make_overlay(
            x,
            sleeve_gross=args.sleeve_gross,
            top_frac=args.top_frac,
            bottom_frac=args.bottom_frac,
            pred_col=args.pred_col,
        )

        one_way_turnover = float(x["overlay_buy_weight"].sum() + x["overlay_sell_weight"].sum())
        cost_ret = one_way_turnover * float(args.roundtrip_cost_mult) * float(args.stock_cost_bps) / 10000.0

        strategy_ov_gross = float((x["target_weight"] * x["overnight_ret"]).sum())
        strategy_ov_net = strategy_ov_gross - cost_ret
        benchmark_ov = float((x["benchmark_weight"] * x["overnight_ret"]).sum())
        base_ov = float((x["base_weight"] * x["overnight_ret"]).sum())

        daily_rows.append({
            "entry_date": d,
            "next_date": nd,
            "snap_datetime": snap_dt,
            "base_overnight_ret": base_ov,
            "strategy_overnight_ret_gross": strategy_ov_gross,
            "strategy_overnight_ret": strategy_ov_net,
            "benchmark_overnight_ret": benchmark_ov,
            "overnight_alpha_ret": strategy_ov_net - benchmark_ov,
            "base_gross": float(x["base_weight"].sum()),
            "target_gross": float(x["target_weight"].sum()),
            "benchmark_gross": float(x["benchmark_weight"].sum()),
            "overlay_buy_weight": float(x["overlay_buy_weight"].sum()),
            "overlay_sell_weight": float(x["overlay_sell_weight"].sum()),
            "one_way_turnover": one_way_turnover,
            "cost_ret": cost_ret,
            "n_names": len(x),
        })

    daily = pd.DataFrame(daily_rows)
    if daily.empty:
        raise RuntimeError("no overnight daily rows built")

    res = curve.copy()
    res["actual_ret_with_overnight"] = pd.to_numeric(res[actual_col], errors="coerce").fillna(0.0)
    res["benchmark_ret_with_overnight"] = pd.to_numeric(res[bench_col], errors="coerce").fillna(0.0)
    res["overnight_strategy_ret_added"] = 0.0
    res["overnight_benchmark_ret_added"] = 0.0

    for r in daily.itertuples(index=False):
        idx = res.index[res["date_int"] == int(r.next_date)]
        if len(idx) == 0:
            continue
        i = int(idx.min())

        res.loc[i, "actual_ret_with_overnight"] = (
            (1.0 + res.loc[i, "actual_ret_with_overnight"])
            * (1.0 + float(r.strategy_overnight_ret))
            - 1.0
        )
        res.loc[i, "benchmark_ret_with_overnight"] = (
            (1.0 + res.loc[i, "benchmark_ret_with_overnight"])
            * (1.0 + float(r.benchmark_overnight_ret))
            - 1.0
        )
        res.loc[i, "overnight_strategy_ret_added"] = float(r.strategy_overnight_ret)
        res.loc[i, "overnight_benchmark_ret_added"] = float(r.benchmark_overnight_ret)

    res["actualret_with_overnight"] = (1.0 + res["actual_ret_with_overnight"]).cumprod() - 1.0
    res["benchmarkret_with_overnight"] = (1.0 + res["benchmark_ret_with_overnight"]).cumprod() - 1.0
    res["alpharet_with_overnight"] = res["actualret_with_overnight"] - res["benchmarkret_with_overnight"]

    res["actualret_intraday_only"] = (1.0 + pd.to_numeric(res[actual_col], errors="coerce").fillna(0.0)).cumprod() - 1.0
    res["benchmarkret_intraday_only"] = (1.0 + pd.to_numeric(res[bench_col], errors="coerce").fillna(0.0)).cumprod() - 1.0
    res["alpharet_intraday_only"] = res["actualret_intraday_only"] - res["benchmarkret_intraday_only"]

    summary = pd.DataFrame([{
        "tag": args.tag,
        "actual_return_with_overnight": float(res["actualret_with_overnight"].iloc[-1]),
        "benchmark_return_with_overnight": float(res["benchmarkret_with_overnight"].iloc[-1]),
        "alpha_return_with_overnight": float(res["alpharet_with_overnight"].iloc[-1]),
        "daily_excess_sharpe_with_overnight": daily_sharpe(res, "actual_ret_with_overnight", "benchmark_ret_with_overnight"),
        "actual_return_intraday_only": float(res["actualret_intraday_only"].iloc[-1]),
        "benchmark_return_intraday_only": float(res["benchmarkret_intraday_only"].iloc[-1]),
        "alpha_return_intraday_only": float(res["alpharet_intraday_only"].iloc[-1]),
        "avg_base_overnight_ret": float(daily["base_overnight_ret"].mean()),
        "avg_strategy_overnight_ret": float(daily["strategy_overnight_ret"].mean()),
        "avg_benchmark_overnight_ret": float(daily["benchmark_overnight_ret"].mean()),
        "avg_overnight_alpha_ret": float(daily["overnight_alpha_ret"].mean()),
        "sum_overnight_alpha_ret": float(daily["overnight_alpha_ret"].sum()),
        "avg_base_gross": float(daily["base_gross"].mean()),
        "avg_target_gross": float(daily["target_gross"].mean()),
        "avg_benchmark_gross": float(daily["benchmark_gross"].mean()),
        "avg_overlay_buy_weight": float(daily["overlay_buy_weight"].mean()),
        "avg_overlay_sell_weight": float(daily["overlay_sell_weight"].mean()),
        "overnight_days": int(len(daily)),
        "entry_ts": int(args.entry_ts),
        "sleeve_gross": float(args.sleeve_gross),
        "top_frac": float(args.top_frac),
        "bottom_frac": float(args.bottom_frac),
        "stock_cost_bps": float(args.stock_cost_bps),
        "pred_mult": float(args.pred_mult),
        "roundtrip_cost_mult": float(args.roundtrip_cost_mult),
    }])

    curve_out = out_dir / f"{args.tag}_intraday_plus_overnight_curve.csv"
    daily_out = out_dir / f"{args.tag}_overnight_daily.csv"
    summary_out = out_dir / f"{args.tag}_intraday_plus_overnight_summary.csv"
    png_out = out_dir / f"{args.tag}_intraday_plus_overnight_nav.png"

    res.to_csv(curve_out, index=False)
    daily.to_csv(daily_out, index=False)
    summary.to_csv(summary_out, index=False)

    res["bar_index"] = np.arange(len(res))
    first_idx = res.groupby("date_int")["bar_index"].first()
    tick_dates = list(first_idx.index)[::2]
    tick_pos = [int(first_idx.loc[d]) for d in tick_dates]

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(res["bar_index"], res["actualret_with_overnight"] * 100, label="strategy actualret")
    ax.plot(res["bar_index"], res["benchmarkret_with_overnight"] * 100, label="CSI2000 benchmarkret")
    ax.plot(res["bar_index"], res["alpharet_with_overnight"] * 100, label="alpharet")
    ax.axhline(0, linewidth=1, linestyle="--")
    ax.set_title("V16 Intraday + Overnight Sleeve, CSI2000 Benchmark, With Overnight PnL")
    ax.set_xlabel("trading minute index; overnight return added to next-day open")
    ax.set_ylabel("cumulative return (%)")
    ax.set_xticks(tick_pos)
    ax.set_xticklabels([str(d) for d in tick_dates], rotation=45, ha="right")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left")

    s = summary.iloc[0]
    text = (
        "Summary\n\n"
        f"Strategy Return\n{s['actual_return_with_overnight']*100:.2f}%\n\n"
        f"Benchmark Return\n{s['benchmark_return_with_overnight']*100:.2f}%\n\n"
        f"Alpha Return\n{s['alpha_return_with_overnight']*100:.2f}%\n\n"
        f"Daily Excess Sharpe\n{s['daily_excess_sharpe_with_overnight']:.2f}"
    )
    ax.text(
        1.05, 0.95, text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )

    fig.tight_layout()
    fig.savefig(png_out, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print("curve:", curve_out)
    print("daily:", daily_out)
    print("summary:", summary_out)
    print("png:", png_out)
    print()
    print(summary.T.to_string())


if __name__ == "__main__":
    main()
