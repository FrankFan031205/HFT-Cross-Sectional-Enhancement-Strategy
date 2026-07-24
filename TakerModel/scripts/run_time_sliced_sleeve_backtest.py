# -*- coding: utf-8 -*-
"""
Time-sliced / sleeve-based pure-CS backtest.

Mentor mode
-----------
一天内有 K 个调仓时间步，就把总股票仓位切成 K 个 sleeve。
每个时间步只更新属于这个 slot 的 1/K 仓位；其他 sleeve 不动。

Example:
  target_total_gross = 0.95
  K = 11
  each sleeve gross ~= 0.95 / 11

At each rebalance timestamp:
  - read full-book target weights at that timestamp
  - normalize to target_total_gross
  - scale by 1/K
  - replace only this slot's sleeve holdings
  - old sleeve was last traded on the previous trading day at the same slot,
    so it is naturally sellable under A-share T+1

This is NOT the full-book T+1-aware optimizer. It is the mentor's
time-sliced execution model.
"""

import argparse
import glob
import math
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_any(path):
    path = Path(path)
    suf = "".join(path.suffixes).lower()
    if suf.endswith(".parquet"):
        return pd.read_parquet(path)
    if suf.endswith(".csv") or suf.endswith(".csv.gz"):
        return pd.read_csv(path, low_memory=False)
    raise ValueError(f"unsupported file: {path}")


def read_glob(path_glob):
    files = sorted(glob.glob(path_glob))
    if not files:
        raise FileNotFoundError(path_glob)
    parts = []
    for f in files:
        print("[read market]", f)
        parts.append(read_any(f))
    return pd.concat(parts, ignore_index=True)


def pick_col(df, candidates, required=True, name="column"):
    for c in candidates:
        if c in df.columns:
            return c
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    if required:
        raise KeyError(f"cannot find {name}; candidates={candidates}; columns={list(df.columns)}")
    return None


def floor_lot(x, lot_size):
    if not np.isfinite(x) or x <= 0:
        return 0
    if lot_size <= 1:
        return int(math.floor(x))
    return int(math.floor(x / lot_size) * lot_size)


def normalize_market(df):
    date_col = pick_col(df, ["date", "execution_date"], name="date")
    dt_col = pick_col(df, ["datetime", "execution_datetime", "tsminute", "timestamp"], name="datetime")
    sid_col = pick_col(df, ["securityid", "SecurityID", "sid", "symbol"], name="symbol")

    bid_col = pick_col(df, ["bid_price", "bid1", "bid", "tbid"], required=False, name="bid")
    ask_col = pick_col(df, ["ask_price", "ask1", "ask", "task"], required=False, name="ask")
    mid_col = pick_col(df, ["mid_price", "price", "tmid"], required=False, name="mid")
    bench_col = pick_col(df, ["benchmark_weight", "bench_weight", "index_weight"], required=False, name="benchmark")

    if bid_col is None and mid_col is None:
        raise KeyError("need bid_price/bid1/tbid or mid_price/price/tmid")

    out = pd.DataFrame({
        "date": df[date_col].astype(int),
        "datetime": pd.to_datetime(df[dt_col]),
        "securityid": df[sid_col].astype(str).str.zfill(6),
    })

    if bid_col is not None:
        out["bid_price"] = pd.to_numeric(df[bid_col], errors="coerce")
    else:
        out["bid_price"] = pd.to_numeric(df[mid_col], errors="coerce")

    if ask_col is not None:
        out["ask_price"] = pd.to_numeric(df[ask_col], errors="coerce")
    else:
        out["ask_price"] = out["bid_price"]

    if mid_col is not None:
        out["mid_price"] = pd.to_numeric(df[mid_col], errors="coerce")
    else:
        out["mid_price"] = (out["bid_price"] + out["ask_price"]) / 2.0

    if bench_col is not None:
        out["benchmark_weight"] = pd.to_numeric(df[bench_col], errors="coerce")
    else:
        out["benchmark_weight"] = np.nan

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=["date", "datetime", "securityid", "bid_price", "ask_price"])
    out = out[(out["bid_price"] > 0) & (out["ask_price"] > 0)].copy()

    if out["benchmark_weight"].isna().all():
        out["benchmark_weight"] = 1.0 / out.groupby(["date", "datetime"])["securityid"].transform("count")
    else:
        out["benchmark_weight"] = out["benchmark_weight"].fillna(0.0)

    out = out.sort_values(["date", "datetime", "securityid"]).drop_duplicates(
        ["date", "datetime", "securityid"], keep="last"
    ).reset_index(drop=True)

    print("[market]", out.shape)
    print("[market dates]", out["date"].min(), "->", out["date"].max(), "n=", out["date"].nunique())
    print("[market minutes]", out[["date", "datetime"]].drop_duplicates().shape[0])
    print("[market symbols]", out["securityid"].nunique())
    return out


def normalize_targets(df, args):
    date_col = pick_col(df, ["date", "execution_date"], name="date")
    dt_col = pick_col(df, ["datetime", "execution_datetime", "tsminute", "timestamp"], name="datetime")
    sid_col = pick_col(df, ["securityid", "SecurityID", "sid", "symbol"], name="symbol")

    if args.weight_col:
        w_col = args.weight_col
        if w_col not in df.columns:
            raise KeyError(f"--weight-col {w_col} not in target file")
    else:
        w_col = pick_col(
            df,
            ["target_weight", "effective_target_weight", "weight", "w", "opt_weight", "optimized_weight"],
            name="target weight",
        )

    out = pd.DataFrame({
        "date": df[date_col].astype(int),
        "datetime": pd.to_datetime(df[dt_col]),
        "securityid": df[sid_col].astype(str).str.zfill(6),
        "raw_weight": pd.to_numeric(df[w_col], errors="coerce").fillna(0.0),
    })

    out = out.sort_values(["date", "datetime", "securityid"]).drop_duplicates(
        ["date", "datetime", "securityid"], keep="last"
    ).reset_index(drop=True)

    if args.long_only:
        out["raw_weight"] = out["raw_weight"].clip(lower=0.0)

    # slot id within each trading day
    times = out[["date", "datetime"]].drop_duplicates().sort_values(["date", "datetime"]).reset_index(drop=True)
    times["slot_id"] = times.groupby("date").cumcount()
    slots_per_day = times.groupby("date")["slot_id"].max() + 1
    k_mode = int(slots_per_day.mode().iloc[0])

    if slots_per_day.nunique() > 1:
        print("[WARN] different slots per day detected:")
        print(slots_per_day.value_counts().sort_index().to_string())
        print("[INFO] using K mode =", k_mode)

    out = out.merge(times, on=["date", "datetime"], how="left")

    # normalize each timestamp's full target gross to target_total_gross, then divide by K
    g = out.groupby(["date", "datetime"])["raw_weight"].transform(lambda x: float(np.abs(x).sum()))
    out["full_target_weight"] = np.where(
        g > 1e-12,
        out["raw_weight"] / g * float(args.target_total_gross),
        0.0,
    )
    out["sleeve_target_weight"] = out["full_target_weight"] / float(k_mode)

    print("[targets]", out.shape)
    print("[target dates]", out["date"].min(), "->", out["date"].max(), "n=", out["date"].nunique())
    print("[target rebalances]", out[["date", "datetime"]].drop_duplicates().shape[0])
    print("[target symbols]", out["securityid"].nunique())
    print("[target weight col]", w_col)
    print("[slots per day K]", k_mode)

    return out, k_mode


def compound_return(x):
    x = pd.Series(x).replace([np.inf, -np.inf], np.nan).dropna()
    if x.empty:
        return np.nan
    return float((1.0 + x).prod() - 1.0)


def daily_sharpe(curve):
    daily = curve.groupby("date", as_index=False).agg(
        actual_day=("actual_ret", compound_return),
        bench_day=("benchmark_ret", compound_return),
    )
    daily["excess_day"] = daily["actual_day"] - daily["bench_day"]
    x = daily["excess_day"].dropna()
    if len(x) <= 1 or x.std(ddof=1) <= 1e-12:
        return np.nan
    return float(x.mean() / x.std(ddof=1) * np.sqrt(252))


def simulate_sleeves(market, targets, k_slots, args):
    target_groups = {
        (int(d), pd.to_datetime(t)): g.copy()
        for (d, t), g in targets.groupby(["date", "datetime"], sort=True)
    }

    market_groups = list(market.groupby(["date", "datetime"], sort=True))

    sleeve_pos = {i: {} for i in range(k_slots)}  # slot_id -> {sid: shares}
    initialized_slots = set()

    cash = float(args.capital)
    prev_equity = float(args.capital)

    prev_date = None
    prev_price = {}
    prev_bench_price = {}
    prev_bench_weight = {}

    rows = []
    trade_rows = []
    position_rows = []

    for (date, dt), cur in market_groups:
        date = int(date)
        dt = pd.to_datetime(dt)
        cur = cur.sort_values("securityid").reset_index(drop=True)

        px_bid = dict(zip(cur["securityid"], cur["bid_price"]))
        px_ask = dict(zip(cur["securityid"], cur["ask_price"]))
        px_mark = px_bid

        new_day = (prev_date is None) or (date != prev_date)

        # aggregate previous positions before trading
        total_pos_before = {}
        for sid_map in sleeve_pos.values():
            for sid, sh in sid_map.items():
                if sh:
                    total_pos_before[sid] = total_pos_before.get(sid, 0) + sh

        # mark equity before interval PnL
        equity_before = cash
        gross_before_currency = 0.0
        n_hold_before = 0
        for sid, sh in total_pos_before.items():
            px = px_mark.get(sid, prev_price.get(sid, np.nan))
            if np.isfinite(px) and px > 0:
                equity_before += sh * px
                gross_before_currency += abs(sh * px)
                n_hold_before += 1

        if len(rows) == 0:
            equity_before = float(args.capital)

        # PnL from previous minute to current minute, no overnight
        pnl_currency = 0.0
        if not new_day:
            for sid, sh in total_pos_before.items():
                if sh == 0:
                    continue
                if sid in px_mark and sid in prev_price and prev_price[sid] > 0:
                    pnl_currency += sh * (px_mark[sid] - prev_price[sid])

        # benchmark return, no overnight
        benchmark_ret = 0.0
        if not new_day:
            for r in cur.itertuples(index=False):
                sid = r.securityid
                px = float(r.bid_price)
                old_px = prev_bench_price.get(sid, np.nan)
                old_w = prev_bench_weight.get(sid, 0.0)
                if np.isfinite(old_px) and old_px > 0:
                    benchmark_ret += old_w * (px / old_px - 1.0)

        # Apply interval PnL before trades.
        cash += pnl_currency

        # Rebalance current sleeve if target exists at this timestamp.
        fee_total = 0.0
        turnover_notional = 0.0
        n_buy = 0
        n_sell = 0
        slot_id = -1

        key = (date, dt)
        if key in target_groups:
            tg = target_groups[key]
            slot_id = int(tg["slot_id"].iloc[0])
            old_pos = sleeve_pos.get(slot_id, {}).copy()

            target_shares = {}
            for r in tg.itertuples(index=False):
                sid = r.securityid
                w = float(r.sleeve_target_weight)
                px = px_mark.get(sid, np.nan)
                if np.isfinite(px) and px > 0 and w > 0:
                    sh = floor_lot(float(args.capital) * w / px, int(args.lot_size))
                    if sh > 0:
                        target_shares[sid] = sh

            all_sids = sorted(set(old_pos) | set(target_shares))

            initial_build_for_slot = slot_id not in initialized_slots
            charge_cost = not (initial_build_for_slot and args.zero_initial_build_cost)

            # Sell first. Under sleeve mode, old slot holdings are from previous day's same slot,
            # so they are sellable by construction after the first initialization.
            for sid in all_sids:
                cur_sh = int(old_pos.get(sid, 0))
                tgt_sh = int(target_shares.get(sid, 0))
                if tgt_sh >= cur_sh:
                    continue
                qty = cur_sh - tgt_sh
                qty = floor_lot(qty, int(args.lot_size))
                if qty <= 0 or sid not in px_bid:
                    continue
                px = float(px_bid[sid])
                notional = qty * px
                if notional < float(args.min_trade_notional) and not args.allow_small_exit:
                    continue
                fee = notional * float(args.fee_bps) / 10000.0 if charge_cost else 0.0
                cash += notional - fee
                turnover_notional += notional
                fee_total += fee
                n_sell += 1

                new_sh = cur_sh - qty
                if new_sh > 0:
                    old_pos[sid] = new_sh
                else:
                    old_pos.pop(sid, None)

                trade_rows.append({
                    "date": date, "datetime": dt, "slot_id": slot_id,
                    "securityid": sid, "side": "SELL", "shares": qty,
                    "price": px, "notional": notional, "fee": fee,
                    "initial_build_for_slot": int(initial_build_for_slot),
                })

            # Buy.
            for sid in all_sids:
                cur_sh = int(old_pos.get(sid, 0))
                tgt_sh = int(target_shares.get(sid, 0))
                if tgt_sh <= cur_sh:
                    continue
                qty = tgt_sh - cur_sh
                qty = floor_lot(qty, int(args.lot_size))
                if qty <= 0 or sid not in px_ask:
                    continue
                px = float(px_ask[sid])
                notional = qty * px
                if notional < float(args.min_trade_notional):
                    continue
                fee_rate = float(args.fee_bps) / 10000.0 if charge_cost else 0.0
                cash_need = notional * (1.0 + fee_rate)
                if cash_need > cash:
                    max_qty = floor_lot(cash / (px * (1.0 + fee_rate)), int(args.lot_size))
                    max_qty = min(max_qty, qty)
                    if max_qty <= 0:
                        continue
                    qty = max_qty
                    notional = qty * px
                    cash_need = notional * (1.0 + fee_rate)

                fee = notional * fee_rate
                cash -= notional + fee
                old_pos[sid] = cur_sh + qty
                turnover_notional += notional
                fee_total += fee
                n_buy += 1

                trade_rows.append({
                    "date": date, "datetime": dt, "slot_id": slot_id,
                    "securityid": sid, "side": "BUY", "shares": qty,
                    "price": px, "notional": notional, "fee": fee,
                    "initial_build_for_slot": int(initial_build_for_slot),
                })

            sleeve_pos[slot_id] = {sid: int(sh) for sid, sh in old_pos.items() if int(sh) > 0}
            initialized_slots.add(slot_id)

        # Aggregate positions after trades.
        total_pos_after = {}
        for sid_map in sleeve_pos.values():
            for sid, sh in sid_map.items():
                if sh:
                    total_pos_after[sid] = total_pos_after.get(sid, 0) + sh

        equity_after = cash
        gross_after_currency = 0.0
        n_hold_after = 0
        for sid, sh in total_pos_after.items():
            px = px_mark.get(sid, prev_price.get(sid, np.nan))
            if np.isfinite(px) and px > 0:
                equity_after += sh * px
                gross_after_currency += abs(sh * px)
                n_hold_after += 1

        actual_ret = equity_after / prev_equity - 1.0 if prev_equity > 0 else 0.0
        actualret = equity_after / float(args.capital) - 1.0

        # cumulative benchmark from returns
        if rows:
            benchmarkret = (1.0 + rows[-1]["benchmarkret"]) * (1.0 + benchmark_ret) - 1.0
        else:
            benchmarkret = benchmark_ret

        alpharet = actualret - benchmarkret

        rows.append({
            "date": date,
            "datetime": dt,
            "slot_id_rebalanced": slot_id,
            "all_slots_initialized": int(len(initialized_slots) >= k_slots),
            "num_initialized_slots": int(len(initialized_slots)),
            "cash": cash,
            "equity": equity_after,
            "pnl_currency": pnl_currency,
            "fee": fee_total,
            "total_cost": fee_total,
            "turnover_notional": turnover_notional,
            "turnover_weight": turnover_notional / float(args.capital),
            "actual_ret": actual_ret,
            "benchmark_ret": benchmark_ret,
            "alpha_ret": actual_ret - benchmark_ret,
            "actualret": actualret,
            "benchmarkret": benchmarkret,
            "alpharet": alpharet,
            "gross_prev_to_capital": gross_before_currency / float(args.capital),
            "gross_after_to_capital": gross_after_currency / float(args.capital),
            "gross_prev_to_equity": gross_before_currency / equity_before if equity_before > 0 else 0.0,
            "gross_after_to_equity": gross_after_currency / equity_after if equity_after > 0 else 0.0,
            "n_hold": n_hold_after,
            "n_buy": n_buy,
            "n_sell": n_sell,
            "n_trade": n_buy + n_sell,
        })

        # Optional position snapshots at rebalance times only to keep file size reasonable.
        if slot_id >= 0:
            for sid, sh in total_pos_after.items():
                px = px_mark.get(sid, np.nan)
                if np.isfinite(px) and px > 0:
                    position_rows.append({
                        "date": date, "datetime": dt, "securityid": sid,
                        "shares": int(sh), "mark_price": px,
                        "weight_to_capital": sh * px / float(args.capital),
                        "weight_to_equity": sh * px / equity_after if equity_after > 0 else 0.0,
                    })

        # update previous prices and benchmark state
        prev_price.update(px_mark)
        prev_bench_price = dict(zip(cur["securityid"], cur["bid_price"]))
        prev_bench_weight = dict(zip(cur["securityid"], cur["benchmark_weight"]))
        prev_date = date
        prev_equity = equity_after

    curve = pd.DataFrame(rows)
    trades = pd.DataFrame(trade_rows)
    positions = pd.DataFrame(position_rows)
    curve["bar_index"] = np.arange(len(curve))
    return curve, trades, positions


def rebase_curve(df):
    df = df.copy().reset_index(drop=True)
    df["actualret"] = (1.0 + df["actual_ret"].fillna(0.0)).cumprod() - 1.0
    df["benchmarkret"] = (1.0 + df["benchmark_ret"].fillna(0.0)).cumprod() - 1.0
    df["alpharet"] = df["actualret"] - df["benchmarkret"]
    df["bar_index"] = np.arange(len(df))
    return df


def make_ticks(curve):
    first = curve.groupby("date", as_index=False)["bar_index"].min()
    step = 1 if len(first) <= 12 else 2 if len(first) <= 24 else max(1, len(first) // 12)
    ticks = first.iloc[::step]
    return ticks["bar_index"].tolist(), ticks["date"].astype(str).tolist()


def plot_and_save(curve, out_dir, tag, suffix, title):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    curve = rebase_curve(curve)

    daily = curve.groupby("date", as_index=False).agg(
        actual_day=("actual_ret", compound_return),
        bench_day=("benchmark_ret", compound_return),
    )
    daily["excess_day"] = daily["actual_day"] - daily["bench_day"]
    sharpe = np.nan
    if len(daily) > 1 and daily["excess_day"].std(ddof=1) > 1e-12:
        sharpe = float(daily["excess_day"].mean() / daily["excess_day"].std(ddof=1) * np.sqrt(252))

    actual = compound_return(curve["actual_ret"])
    bench = compound_return(curve["benchmark_ret"])
    alpha = actual - bench

    summary = pd.DataFrame([{
        "start_datetime": str(curve["datetime"].iloc[0]),
        "end_datetime": str(curve["datetime"].iloc[-1]),
        "actual_return": actual,
        "benchmark_return": bench,
        "alpha_return": alpha,
        "daily_excess_sharpe": sharpe,
        "avg_gross_prev_to_equity": float(curve["gross_prev_to_equity"].mean()),
        "avg_gross_after_to_equity": float(curve["gross_after_to_equity"].mean()),
        "turnover_weight": float(curve["turnover_weight"].sum()),
        "total_cost": float(curve["total_cost"].sum()),
        "n_minutes": int(len(curve)),
        "n_days": int(curve["date"].nunique()),
    }])

    curve_path = out_dir / f"{tag}_{suffix}_curve.csv"
    summary_path = out_dir / f"{tag}_{suffix}_summary.csv"
    png_path = out_dir / f"{tag}_{suffix}_nav.png"

    curve.to_csv(curve_path, index=False)
    summary.to_csv(summary_path, index=False)

    ticks, labels = make_ticks(curve)

    fig = plt.figure(figsize=(18, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=[5.5, 1.3], wspace=0.08)
    ax = fig.add_subplot(gs[0, 0])
    box = fig.add_subplot(gs[0, 1])
    box.axis("off")

    ax.plot(curve["bar_index"], curve["actualret"] * 100, label="sleeve actualret")
    ax.plot(curve["bar_index"], curve["benchmarkret"] * 100, label="benchmarkret")
    ax.plot(curve["bar_index"], curve["alpharet"] * 100, label="sleeve alpharet")
    ax.axhline(0.0, linestyle="--", linewidth=0.8)
    ax.grid(True, alpha=0.3)
    ax.set_title(title)
    ax.set_ylabel("cumulative return (%)")
    ax.set_xlabel("trading minute index")
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend(loc="lower left")

    r = summary.iloc[0]
    info = (
        "Summary\n\n"
        f"Strategy Return\n{r['actual_return']:.2%}\n\n"
        f"Benchmark Return\n{r['benchmark_return']:.2%}\n\n"
        f"Alpha Return\n{r['alpha_return']:.2%}\n\n"
        f"Daily Excess Sharpe\n{r['daily_excess_sharpe']:.2f}"
    )
    box.text(0.02, 0.95, info, va="top", ha="left", fontsize=12,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="0.6"))

    fig.tight_layout()
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"\n===== {suffix} summary =====")
    print(summary.T.to_string(header=False))
    print("[saved curve]", curve_path)
    print("[saved summ ]", summary_path)
    print("[saved png  ]", png_path)

    return curve_path, summary_path, png_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market-glob", required=True)
    ap.add_argument("--source-target-positions", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", required=True)

    ap.add_argument("--capital", type=float, default=200_000_000.0)
    ap.add_argument("--target-total-gross", type=float, default=0.95)
    ap.add_argument("--fee-bps", type=float, default=10.0)
    ap.add_argument("--lot-size", type=int, default=100)
    ap.add_argument("--min-trade-notional", type=float, default=5000.0)
    ap.add_argument("--allow-small-exit", type=int, default=1)
    ap.add_argument("--zero-initial-build-cost", type=int, default=1)
    ap.add_argument("--long-only", type=int, default=1)
    ap.add_argument("--weight-col", default="")

    args = ap.parse_args()
    args.allow_small_exit = bool(args.allow_small_exit)
    args.zero_initial_build_cost = bool(args.zero_initial_build_cost)
    args.long_only = bool(args.long_only)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    market = normalize_market(read_glob(args.market_glob))
    targets, k_slots = normalize_targets(read_any(args.source_target_positions), args)

    curve, trades, positions = simulate_sleeves(market, targets, k_slots, args)

    all_curve = out_dir / f"{args.tag}_all_minutes_raw_curve.csv"
    trades_path = out_dir / f"{args.tag}_trades.csv"
    pos_path = out_dir / f"{args.tag}_positions_at_rebalance.csv"
    curve.to_csv(all_curve, index=False)
    trades.to_csv(trades_path, index=False)
    positions.to_csv(pos_path, index=False)
    print("[saved raw curve]", all_curve)
    print("[saved trades   ]", trades_path)
    print("[saved pos      ]", pos_path)

    # 1) raw gradual-build curve from the sample start
    plot_and_save(
        curve,
        out_dir,
        args.tag,
        "raw_gradual_build",
        "Time-sliced Sleeve Pure-CS, Raw Gradual Build",
    )

    # 2) valid steady-state curve after first day, all sleeves have been built once
    first_date = int(curve["date"].min())
    valid = curve[curve["date"] > first_date].copy()
    if len(valid) > 0:
        plot_and_save(
            valid,
            out_dir,
            args.tag,
            "after_first_day",
            "Time-sliced Sleeve Pure-CS, After First Full Sleeve Cycle",
        )


if __name__ == "__main__":
    main()
