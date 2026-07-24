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


def norm_date(s):
    return (
        s.astype(str)
        .str.replace(r"\D", "", regex=True)
        .str.slice(0, 8)
        .astype(int)
    )


def norm_sid(df):
    if "sid" in df.columns:
        return pd.to_numeric(df["sid"], errors="coerce").astype("Int64")
    if "securityid" in df.columns:
        return pd.to_numeric(df["securityid"].astype(str).str.extract(r"(\d+)")[0], errors="coerce").astype("Int64")
    if "SecurityID" in df.columns:
        return pd.to_numeric(df["SecurityID"].astype(str).str.extract(r"(\d+)")[0], errors="coerce").astype("Int64")
    raise KeyError(f"cannot find sid/securityid columns: {df.columns.tolist()}")


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


def choose_signal_col(df, requested):
    if requested and requested != "auto":
        if requested not in df.columns:
            raise KeyError(f"signal_col={requested} not found; columns={df.columns.tolist()}")
        return requested

    for c in ["pred_ret_h20", "pred_ret", "predz_h20", "pred_z", "predz", "score_h20", "alpha"]:
        if c in df.columns:
            return c
    raise KeyError(f"cannot auto-detect signal column; columns={df.columns.tolist()}")


def daily_turnover(series, date, charge_open=True, charge_close=True):
    s = pd.to_numeric(series, errors="coerce").fillna(0.0)
    prev = s.groupby(date).shift(1).fillna(0.0)
    turn = (s - prev).abs()

    bar = s.groupby(date).cumcount()

    if not charge_open:
        turn.loc[bar == 0] = 0.0

    if charge_close:
        last_bar = bar.groupby(date).transform("max")
        turn.loc[bar == last_bar] += s.loc[bar == last_bar].abs()

    return turn


def build_completion_day(
    market_file,
    curve_day,
    signal_col_req,
    target_col,
    top_frac,
    weighting,
    rebal_every_bars,
    stock_cost_bps,
    charge_open,
    charge_close,
):
    m = pd.read_parquet(market_file)

    if "date" in m.columns:
        m["date"] = norm_date(m["date"])
    else:
        d = int(Path(market_file).stem.replace("optimizer_input_", ""))
        m["date"] = d

    m["datetime"] = pd.to_datetime(m["datetime"])
    m["sid"] = norm_sid(m)

    px_col = pick_col(m, ["mid_price", "tmid", "price", "mark_price", "bid1", "ask1"], "price")
    sig_col = choose_signal_col(m, signal_col_req)

    if "benchmark_weight" not in m.columns:
        m["benchmark_weight"] = 1.0

    m = m[["date", "datetime", "sid", px_col, sig_col, "benchmark_weight"]].copy()
    m = m.rename(columns={px_col: "mid_price", sig_col: "signal"})
    m["mid_price"] = pd.to_numeric(m["mid_price"], errors="coerce")
    m["signal"] = pd.to_numeric(m["signal"], errors="coerce").fillna(0.0)
    m["benchmark_weight"] = pd.to_numeric(m["benchmark_weight"], errors="coerce").fillna(0.0)

    m = m.dropna(subset=["datetime", "sid", "mid_price"])
    m = m.sort_values(["sid", "datetime"])
    m["stock_bar_ret"] = m.groupby("sid")["mid_price"].pct_change().fillna(0.0)

    curve_day = curve_day[["datetime", target_col]].copy()
    curve_day = curve_day.drop_duplicates("datetime").sort_values("datetime")
    curve_day["bar_in_day"] = np.arange(len(curve_day))

    # 只在每 rebal_every_bars 个 bar 更新一次 completion basket
    rebal_curve = curve_day[curve_day["bar_in_day"] % int(rebal_every_bars) == 0].copy()

    if rebal_curve.empty:
        raise RuntimeError(f"no rebalance timestamps for {market_file}")

    weights_pieces = []
    for r in rebal_curve.itertuples(index=False):
        dt = r.datetime
        gross = float(getattr(r, target_col))

        g = m[m["datetime"] == dt].copy()
        if g.empty:
            continue

        base = g[["datetime", "sid", "signal", "benchmark_weight"]].copy()
        base["completion_weight"] = 0.0

        if gross > 1e-12:
            n = max(1, int(len(base) * float(top_frac)))
            top = base.sort_values("signal", ascending=False).head(n).copy()

            if weighting == "benchmark":
                w = top["benchmark_weight"].clip(lower=0.0)
                if w.sum() > 1e-12:
                    alloc = gross * w / w.sum()
                else:
                    alloc = pd.Series(gross / len(top), index=top.index)
            elif weighting == "score":
                s = top["signal"] - top["signal"].min()
                s = s.clip(lower=0.0) + 1e-12
                alloc = gross * s / s.sum()
            else:
                alloc = pd.Series(gross / len(top), index=top.index)

            base.loc[top.index, "completion_weight"] = alloc.values

        weights_pieces.append(base[["datetime", "sid", "completion_weight"]])

    if not weights_pieces:
        raise RuntimeError(f"no target weights built for {market_file}")

    weights = pd.concat(weights_pieces, ignore_index=True)

    # 全市场 minute grid
    grid = m[["date", "datetime", "sid", "stock_bar_ret"]].copy()
    grid = grid.merge(weights, on=["datetime", "sid"], how="left")
    grid = grid.sort_values(["sid", "datetime"])

    # 在 rebalance 之间持有不动
    grid["completion_weight"] = grid.groupby("sid")["completion_weight"].ffill().fillna(0.0)

    # 收益用上一 bar 权重，避免 lookahead
    grid["prev_weight"] = grid.groupby("sid")["completion_weight"].shift(1).fillna(0.0)
    first_dt = grid["datetime"].min()
    grid.loc[grid["datetime"] == first_dt, "prev_weight"] = 0.0
    grid["ret_contrib"] = grid["prev_weight"] * grid["stock_bar_ret"]

    # 换手只在 target 改变时产生；持有期间 diff=0
    grid["prev_w_for_turn"] = grid.groupby("sid")["completion_weight"].shift(1).fillna(0.0)
    grid["turn_contrib"] = (grid["completion_weight"] - grid["prev_w_for_turn"]).abs()

    if not charge_open:
        grid.loc[grid["datetime"] == first_dt, "turn_contrib"] = 0.0

    if charge_close:
        last_dt = grid["datetime"].max()
        grid.loc[grid["datetime"] == last_dt, "turn_contrib"] += grid.loc[grid["datetime"] == last_dt, "completion_weight"].abs()

    ret = grid.groupby("datetime", as_index=False)["ret_contrib"].sum()
    ret = ret.rename(columns={"ret_contrib": "alpha_completion_ret"})

    turn = grid.groupby("datetime", as_index=False)["turn_contrib"].sum()
    turn = turn.rename(columns={"turn_contrib": "alpha_completion_turnover"})
    turn["alpha_completion_cost_ret"] = turn["alpha_completion_turnover"] * float(stock_cost_bps) / 10000.0

    gross = grid.groupby("datetime", as_index=False)["completion_weight"].sum()
    gross = gross.rename(columns={"completion_weight": "realized_completion_gross"})

    out = ret.merge(turn, on="datetime", how="outer").merge(gross, on="datetime", how="outer")
    out["signal_col"] = sig_col
    return out.fillna(0.0)


def make_daily_table(df, actual_col, bench_col):
    x = df.copy()
    x["date"] = pd.to_datetime(x["datetime"]).dt.strftime("%Y%m%d").astype(int)
    d = x.groupby("date").agg(
        actual=(actual_col, compound),
        bench=(bench_col, compound),
        avg_raw_stock_gross=("raw_stock_gross", "mean"),
        avg_alpha_completion=("realized_completion_gross", "mean"),
        avg_stock_core=("stock_core_gross", "mean"),
        avg_futures=("futures_target_weight", "mean"),
        avg_total_exposure=("total_target_exposure", "mean"),
        completion_cost=("alpha_completion_cost_ret", "sum"),
        futures_fee=("futures_fee_ret", "sum"),
    )
    d["excess"] = d["actual"] - d["bench"]
    return d.reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-curve", required=True)
    ap.add_argument("--market-input-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", required=True)

    ap.add_argument("--target-stock-gross", type=float, default=0.95)
    ap.add_argument("--target-total-exposure", type=float, default=1.00)
    ap.add_argument("--max-stock-completion", type=float, default=0.12)
    ap.add_argument("--max-futures-gross", type=float, default=0.08)

    ap.add_argument("--signal-col", default="auto")
    ap.add_argument("--top-frac", type=float, default=0.10)
    ap.add_argument("--weighting", choices=["benchmark", "equal", "score"], default="benchmark")
    ap.add_argument("--rebal-every-bars", type=int, default=10)

    ap.add_argument("--stock-cost-bps", type=float, default=10.0)
    ap.add_argument("--futures-fee-rate", type=float, default=0.000023)
    ap.add_argument("--charge-open-turnover", type=int, default=1)
    ap.add_argument("--charge-close-turnover", type=int, default=1)

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    x = pd.read_csv(args.base_curve)
    x["datetime"] = pd.to_datetime(x["datetime"])
    x = x.sort_values("datetime").reset_index(drop=True)
    x["date"] = x["datetime"].dt.strftime("%Y%m%d").astype(int)
    x["bar_in_day"] = x.groupby("date").cumcount()

    actual_base_col = pick_col(x, ["actual_ret_with_overnight"], "v18 actual")
    bench_col = pick_col(x, ["benchmark_ret_with_overnight"], "benchmark")
    if "stock_gross" not in x.columns:
        raise KeyError("base curve must contain stock_gross. Use v19 curve.")
    if "futures_ret" not in x.columns:
        raise KeyError("base curve must contain futures_ret. Use v19 curve.")

    x["raw_stock_gross"] = pd.to_numeric(x["stock_gross"], errors="coerce").fillna(0.0)
    x["futures_ret"] = pd.to_numeric(x["futures_ret"], errors="coerce").fillna(0.0)

    x["stock_completion_target_gross"] = (
        float(args.target_stock_gross) - x["raw_stock_gross"]
    ).clip(lower=0.0, upper=float(args.max_stock_completion))

    parts = []
    for d, day_curve in x.groupby("date", sort=True):
        f = Path(args.market_input_dir) / f"optimizer_input_{int(d)}.parquet"
        if not f.exists():
            print("[WARN] missing market:", f)
            continue

        print(f"===== build completion {d}, rebal_every_bars={args.rebal_every_bars}, file={f.name} =====")
        y = build_completion_day(
            market_file=f,
            curve_day=day_curve,
            signal_col_req=args.signal_col,
            target_col="stock_completion_target_gross",
            top_frac=args.top_frac,
            weighting=args.weighting,
            rebal_every_bars=args.rebal_every_bars,
            stock_cost_bps=args.stock_cost_bps,
            charge_open=bool(args.charge_open_turnover),
            charge_close=bool(args.charge_close_turnover),
        )
        parts.append(y)

    if not parts:
        raise RuntimeError("no completion outputs")

    comp = pd.concat(parts, ignore_index=True)

    x = x.merge(comp, on="datetime", how="left")
    for c in ["alpha_completion_ret", "alpha_completion_turnover", "alpha_completion_cost_ret", "realized_completion_gross"]:
        x[c] = pd.to_numeric(x[c], errors="coerce").fillna(0.0)

    x["stock_core_gross"] = x["raw_stock_gross"] + x["realized_completion_gross"]

    x["futures_target_weight"] = (
        float(args.target_total_exposure) - x["stock_core_gross"]
    ).clip(lower=0.0, upper=float(args.max_futures_gross))

    x["futures_prev_weight"] = x.groupby("date")["futures_target_weight"].shift(1).fillna(0.0)
    x.loc[x["bar_in_day"] == 0, "futures_prev_weight"] = 0.0

    x["futures_turnover"] = daily_turnover(
        x["futures_target_weight"],
        x["date"],
        charge_open=bool(args.charge_open_turnover),
        charge_close=bool(args.charge_close_turnover),
    )
    x["futures_fee_ret"] = x["futures_turnover"] * float(args.futures_fee_rate)
    x["futures_overlay_ret"] = x["futures_prev_weight"] * x["futures_ret"]

    x["actual_ret_v23b"] = (
        pd.to_numeric(x[actual_base_col], errors="coerce").fillna(0.0)
        + x["alpha_completion_ret"]
        + x["futures_overlay_ret"]
        - x["alpha_completion_cost_ret"]
        - x["futures_fee_ret"]
    )
    x["benchmark_ret_v23b"] = pd.to_numeric(x[bench_col], errors="coerce").fillna(0.0)

    x["actualret_v23b"] = (1.0 + x["actual_ret_v23b"]).cumprod() - 1.0
    x["benchmarkret_v23b"] = (1.0 + x["benchmark_ret_v23b"]).cumprod() - 1.0
    x["alpharet_v23b"] = x["actualret_v23b"] - x["benchmarkret_v23b"]

    x["total_target_exposure"] = x["raw_stock_gross"] + x["realized_completion_gross"] + x["futures_target_weight"]

    sig_used = "|".join(sorted(set(str(v) for v in x.get("signal_col", pd.Series(dtype=str)).dropna().unique())))

    summary = pd.DataFrame([{
        "tag": args.tag,
        "actual_return": float(x["actualret_v23b"].iloc[-1]),
        "benchmark_return": float(x["benchmarkret_v23b"].iloc[-1]),
        "alpha_return": float(x["alpharet_v23b"].iloc[-1]),
        "daily_excess_sharpe": daily_sharpe(x, "actual_ret_v23b", "benchmark_ret_v23b"),

        "avg_raw_stock_gross": float(x["raw_stock_gross"].mean()),
        "avg_alpha_completion_gross": float(x["realized_completion_gross"].mean()),
        "avg_stock_core_gross": float(x["stock_core_gross"].mean()),
        "avg_futures_gross": float(x["futures_target_weight"].mean()),
        "avg_total_exposure": float(x["total_target_exposure"].mean()),

        "avg_alpha_completion_turnover": float(x["alpha_completion_turnover"].mean()),
        "avg_futures_turnover": float(x["futures_turnover"].mean()),
        "total_alpha_completion_cost_ret": float(x["alpha_completion_cost_ret"].sum()),
        "total_futures_fee_ret": float(x["futures_fee_ret"].sum()),

        "target_stock_gross": float(args.target_stock_gross),
        "target_total_exposure": float(args.target_total_exposure),
        "max_stock_completion": float(args.max_stock_completion),
        "max_futures_gross": float(args.max_futures_gross),
        "top_frac": float(args.top_frac),
        "weighting": args.weighting,
        "rebal_every_bars": int(args.rebal_every_bars),
        "signal_col_requested": args.signal_col,
        "signal_cols_used": sig_used,
        "stock_cost_bps": float(args.stock_cost_bps),
        "futures_fee_rate": float(args.futures_fee_rate),
        "charge_open_turnover": int(args.charge_open_turnover),
        "charge_close_turnover": int(args.charge_close_turnover),
    }])

    daily = make_daily_table(x, "actual_ret_v23b", "benchmark_ret_v23b")

    curve_out = out_dir / f"{args.tag}_curve.csv"
    daily_out = out_dir / f"{args.tag}_daily.csv"
    summary_out = out_dir / f"{args.tag}_summary.csv"
    png_out = out_dir / f"{args.tag}_nav.png"

    x.to_csv(curve_out, index=False)
    daily.to_csv(daily_out, index=False)
    summary.to_csv(summary_out, index=False)

    x["plot_index"] = np.arange(len(x))
    first_idx = x.groupby("date")["plot_index"].first()
    tick_dates = list(first_idx.index)[::2]
    tick_pos = [int(first_idx.loc[d]) for d in tick_dates]

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(x["plot_index"], x["actualret_v23b"] * 100.0, label="strategy actualret")
    ax.plot(x["plot_index"], x["benchmarkret_v23b"] * 100.0, label="CSI2000 benchmarkret")
    ax.plot(x["plot_index"], x["alpharet_v23b"] * 100.0, label="alpharet")
    ax.axhline(0, linewidth=1, linestyle="--")
    ax.set_title("V23b Rebalanced Alpha-Aware Stock Completion + Passive Futures")
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
