import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--positions", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--tag", default="actual_qty")
    ap.add_argument("--capital", type=float, default=200_000_000)
    ap.add_argument("--fee_bps", type=float, default=0.5)
    ap.add_argument("--slippage_bps", type=float, default=0.0)
    ap.add_argument("--clip_short_sell", action="store_true")
    args = ap.parse_args()

    capital = float(args.capital)
    pos_path = Path(args.positions)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_minute = out_dir / f"share_level_actual_qty_minute_{args.tag}.csv"
    out_daily = out_dir / f"share_level_actual_qty_daily_{args.tag}.csv"
    out_summary = out_dir / f"share_level_actual_qty_summary_{args.tag}.csv"
    out_audit = out_dir / f"share_level_actual_qty_trade_audit_{args.tag}.csv"
    out_negative = out_dir / f"share_level_actual_qty_negative_qty_{args.tag}.csv"

    print("reading:", pos_path)

    head = pd.read_csv(pos_path, nrows=0).columns.tolist()

    need_cols = [
        "execution_date",
        "execution_datetime",
        "securityid",
        "mid_price",
        "bid_price",
        "ask_price",
        "exec_price",
        "trade_side",
        "trade_notional",
        "current_qty",
        "target_qty",
        "effective_target_qty",
        "delta_qty_raw",
        "delta_qty_executable",
        "executed_delta_weight",
        "executed_weight",
        "fee",
        "slippage",
        "skip_reason",
    ]

    usecols = [c for c in need_cols if c in head]
    missing = [c for c in ["execution_date", "execution_datetime", "securityid", "mid_price"] if c not in usecols]
    if missing:
        raise RuntimeError(f"missing required columns: {missing}")

    p = pd.read_csv(pos_path, usecols=usecols, low_memory=False)

    p["execution_date"] = p["execution_date"].astype(int)
    p["execution_datetime"] = pd.to_datetime(p["execution_datetime"].astype(str), errors="coerce")
    p["securityid"] = pd.to_numeric(p["securityid"], errors="coerce").astype("Int64")

    if "trade_side" not in p.columns:
        p["trade_side"] = "NONE"
    p["trade_side"] = p["trade_side"].astype(str).str.upper()

    for c in [
        "mid_price", "bid_price", "ask_price", "exec_price",
        "trade_notional", "current_qty", "target_qty",
        "effective_target_qty",
        "delta_qty_raw", "delta_qty_executable",
        "executed_delta_weight", "executed_weight",
        "fee", "slippage",
    ]:
        if c in p.columns:
            p[c] = pd.to_numeric(p[c], errors="coerce")

    p = p.dropna(subset=["execution_datetime", "securityid", "mid_price"]).copy()
    p["securityid"] = p["securityid"].astype(int)
    p = p.sort_values(["execution_datetime", "securityid"]).reset_index(drop=True)

    for c in ["trade_notional", "delta_qty_executable"]:
        if c not in p.columns:
            p[c] = 0.0

    reported_trade = (
        p["trade_notional"].fillna(0).gt(0)
        & p["trade_side"].isin(["BUY", "SELL"])
    )

    real_qty_trade = (
        reported_trade
        & p["delta_qty_executable"].fillna(0).ne(0)
    )

    fake_zero_qty_trade = reported_trade & p["delta_qty_executable"].fillna(0).eq(0)

    print("\n===== input trade audit =====")
    print("position rows:", len(p))
    print("reported trade rows:", int(reported_trade.sum()))
    print("real qty trade rows:", int(real_qty_trade.sum()))
    print("fake zero-qty trade rows:", int(fake_zero_qty_trade.sum()))
    print("reported notional:", float(p.loc[reported_trade, "trade_notional"].sum()))
    print("fake reported notional:", float(p.loc[fake_zero_qty_trade, "trade_notional"].sum()))

    audit_rows = []

    fake = p.loc[fake_zero_qty_trade].copy()
    if len(fake):
        fake.to_csv(out_audit, index=False)
        print("saved fake trade audit:", out_audit)

    trade = p.loc[real_qty_trade].copy()
    if len(trade):
        trade["qty_notional"] = trade["delta_qty_executable"].abs() * trade["exec_price"]
        print("qty*price notional:", float(trade["qty_notional"].sum()))
        print("old / qty_notional ratio:")
        print((trade["trade_notional"] / trade["qty_notional"].replace(0, np.nan)).describe())

    cash = capital
    qty = defaultdict(float)
    last_mid = {}

    prev_equity = capital
    peak_equity = capital
    max_drawdown = 0.0

    total_turnover = 0.0
    total_fee = 0.0
    total_slippage = 0.0
    total_spread_cost_est = 0.0
    total_trade_events = 0
    total_buy_trades = 0
    total_sell_trades = 0
    clipped_short_sell_qty = 0.0

    negative_qty_rows = []
    minute_rows = []

    for dt, g in p.groupby("execution_datetime", sort=True):
        execution_date = int(g["execution_date"].iloc[0])

        # update visible mid prices
        for _, r in g.iterrows():
            sid = int(r["securityid"])
            mid = float(r["mid_price"])
            if np.isfinite(mid) and mid > 0:
                last_mid[sid] = mid

        turnover = 0.0
        fee_sum = 0.0
        slippage_sum = 0.0
        spread_cost_est = 0.0
        trade_events = 0
        buy_trades = 0
        sell_trades = 0

        tg = g[
            g["trade_notional"].fillna(0).gt(0)
            & g["trade_side"].isin(["BUY", "SELL"])
            & g["delta_qty_executable"].fillna(0).ne(0)
        ]

        for _, r in tg.iterrows():
            sid = int(r["securityid"])
            side = str(r["trade_side"]).upper()

            dq = float(r["delta_qty_executable"])
            if not np.isfinite(dq) or abs(dq) < 1e-12:
                continue

            if side == "BUY" and dq < 0:
                dq = abs(dq)
            if side == "SELL" and dq > 0:
                dq = -abs(dq)

            if side == "SELL" and args.clip_short_sell:
                available = max(qty[sid], 0.0)
                sell_qty = min(abs(dq), available)
                if sell_qty < abs(dq):
                    clipped_short_sell_qty += abs(dq) - sell_qty
                dq = -sell_qty
                if abs(dq) < 1e-12:
                    continue

            exec_price = r.get("exec_price", np.nan)
            if not np.isfinite(exec_price) or exec_price <= 0:
                if side == "BUY":
                    exec_price = r.get("ask_price", np.nan)
                else:
                    exec_price = r.get("bid_price", np.nan)

            if not np.isfinite(exec_price) or exec_price <= 0:
                continue

            notional = abs(dq) * exec_price
            if notional <= 0:
                continue

            fee = notional * args.fee_bps / 10000.0
            slippage = notional * args.slippage_bps / 10000.0

            mid = r.get("mid_price", np.nan)
            if np.isfinite(mid) and mid > 0:
                if side == "BUY":
                    spread_cost_est += max(exec_price - mid, 0.0) * abs(dq)
                else:
                    spread_cost_est += max(mid - exec_price, 0.0) * abs(dq)

            if side == "BUY":
                cash -= notional
                buy_trades += 1
            else:
                cash += notional
                sell_trades += 1

            cash -= fee + slippage
            qty[sid] += dq

            if qty[sid] < -1e-6:
                negative_qty_rows.append({
                    "execution_datetime": dt,
                    "execution_date": execution_date,
                    "securityid": sid,
                    "side": side,
                    "dq": dq,
                    "qty_after": qty[sid],
                    "exec_price": exec_price,
                    "notional": notional,
                })

            turnover += notional
            fee_sum += fee
            slippage_sum += slippage
            trade_events += 1

        market_value = 0.0
        gross_exposure = 0.0
        net_exposure = 0.0
        n_hold = 0

        for sid, q in list(qty.items()):
            if abs(q) < 1e-12:
                qty[sid] = 0.0
                continue

            mid = last_mid.get(sid, np.nan)
            if not np.isfinite(mid) or mid <= 0:
                continue

            mv = q * mid
            market_value += mv
            gross_exposure += abs(mv)
            net_exposure += mv
            n_hold += 1

        equity = cash + market_value
        step_net_pnl = equity - prev_equity

        peak_equity = max(peak_equity, equity)
        drawdown = equity - peak_equity
        max_drawdown = min(max_drawdown, drawdown)

        total_turnover += turnover
        total_fee += fee_sum
        total_slippage += slippage_sum
        total_spread_cost_est += spread_cost_est
        total_trade_events += trade_events
        total_buy_trades += buy_trades
        total_sell_trades += sell_trades

        minute_rows.append({
            "execution_datetime": dt,
            "execution_date": execution_date,
            "cash": cash,
            "market_value": market_value,
            "equity": equity,
            "step_net_pnl": step_net_pnl,
            "turnover": turnover,
            "fee": fee_sum,
            "slippage": slippage_sum,
            "spread_cost_est": spread_cost_est,
            "trade_events": trade_events,
            "buy_trades": buy_trades,
            "sell_trades": sell_trades,
            "gross_exposure": gross_exposure,
            "net_exposure": net_exposure,
            "gross_weight": gross_exposure / capital,
            "net_weight": net_exposure / capital,
            "n_hold": n_hold,
            "drawdown": drawdown,
        })

        prev_equity = equity

    minute = pd.DataFrame(minute_rows)
    minute["cum_net_pnl"] = minute["equity"] - capital
    minute.to_csv(out_minute, index=False)

    daily = (
        minute.groupby("execution_date", as_index=False)
        .agg(
            num_minutes=("execution_datetime", "count"),
            num_trade_events=("trade_events", "sum"),
            num_buy_trades=("buy_trades", "sum"),
            num_sell_trades=("sell_trades", "sum"),
            turnover=("turnover", "sum"),
            fee=("fee", "sum"),
            slippage=("slippage", "sum"),
            spread_cost_est=("spread_cost_est", "sum"),
            gross_exposure_mean=("gross_exposure", "mean"),
            gross_weight_mean=("gross_weight", "mean"),
            net_weight_mean=("net_weight", "mean"),
            n_hold_mean=("n_hold", "mean"),
            end_equity=("equity", "last"),
            end_cum_net_pnl=("cum_net_pnl", "last"),
            daily_net_pnl=("step_net_pnl", "sum"),
            min_drawdown=("drawdown", "min"),
        )
    )
    daily.to_csv(out_daily, index=False)

    neg = pd.DataFrame(negative_qty_rows)
    if len(neg):
        neg.to_csv(out_negative, index=False)

    summary = {
        "start_date": int(daily["execution_date"].min()),
        "end_date": int(daily["execution_date"].max()),
        "capital": capital,
        "num_minutes": len(minute),
        "num_days": len(daily),
        "reported_trade_rows": int(reported_trade.sum()),
        "real_qty_trade_rows": int(real_qty_trade.sum()),
        "fake_zero_qty_trade_rows": int(fake_zero_qty_trade.sum()),
        "reported_trade_notional": float(p.loc[reported_trade, "trade_notional"].sum()),
        "fake_reported_notional": float(p.loc[fake_zero_qty_trade, "trade_notional"].sum()),
        "num_trade_events": int(total_trade_events),
        "num_buy_trades": int(total_buy_trades),
        "num_sell_trades": int(total_sell_trades),
        "total_turnover_qty_price": total_turnover,
        "turnover_to_capital_qty_price": total_turnover / capital,
        "total_fee_qty_price": total_fee,
        "total_slippage": total_slippage,
        "total_spread_cost_est": total_spread_cost_est,
        "final_equity": float(minute["equity"].iloc[-1]),
        "total_net_pnl": float(minute["equity"].iloc[-1] - capital),
        "return_on_capital": float((minute["equity"].iloc[-1] - capital) / capital),
        "max_drawdown": float(max_drawdown),
        "avg_gross_weight": float(minute["gross_weight"].mean()),
        "avg_net_weight": float(minute["net_weight"].mean()),
        "avg_n_hold": float(minute["n_hold"].mean()),
        "negative_qty_events": int(len(negative_qty_rows)),
        "clipped_short_sell_qty": float(clipped_short_sell_qty),
    }

    summary_df = pd.DataFrame.from_dict(summary, orient="index", columns=["value"])
    summary_df.to_csv(out_summary)

    print("\n===== v4 actual-qty summary =====")
    print(summary_df)

    print("\n===== daily =====")
    print(daily)

    print("\nsaved minute:", out_minute)
    print("saved daily :", out_daily)
    print("saved summary:", out_summary)
    if len(neg):
        print("saved negative qty:", out_negative)


if __name__ == "__main__":
    main()
