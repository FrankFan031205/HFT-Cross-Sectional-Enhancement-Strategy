import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd


def floor_lot(x, lot):
    if not np.isfinite(x) or x <= 0:
        return 0.0
    return np.floor(x / lot) * lot


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--positions", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--tag", required=True)

    ap.add_argument("--capital", type=float, default=200_000_000)
    ap.add_argument("--fee_bps", type=float, default=0.5)
    ap.add_argument("--slippage_bps", type=float, default=0.0)
    ap.add_argument("--lot_size", type=float, default=100)

    ap.add_argument("--gross_cap", type=float, default=0.8)
    ap.add_argument("--entry_rebalance_ratio", type=float, default=0.5)
    ap.add_argument("--reduce_rebalance_ratio", type=float, default=0.25)
    ap.add_argument("--exit_zero_ratio", type=float, default=1.0)

    ap.add_argument("--min_trade_notional", type=float, default=5000.0)
    ap.add_argument("--entry_max_spread_bps", type=float, default=10.0)
    ap.add_argument("--exit_max_spread_bps", type=float, default=999.0)

    args = ap.parse_args()

    capital = float(args.capital)
    lot = float(args.lot_size)

    pos_path = Path(args.positions)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_minute = out_dir / f"v5_actual_qty_minute_{args.tag}.csv"
    out_daily = out_dir / f"v5_actual_qty_daily_{args.tag}.csv"
    out_summary = out_dir / f"v5_actual_qty_summary_{args.tag}.csv"
    out_trades = out_dir / f"v5_actual_qty_trades_{args.tag}.csv"

    print("reading:", pos_path)

    head = pd.read_csv(pos_path, nrows=0).columns.tolist()

    need_cols = [
        "execution_date",
        "execution_datetime",
        "securityid",
        "mid_price",
        "bid_price",
        "ask_price",
        "spread_bps_realized",
        "spread_bps",
        "valid_market_bool",
        "valid_market",
        "effective_target_weight",
        "target_weight",
        "desired_target_weight",
        "buy_edge_bps",
        "sell_edge_bps",
        "selected",
        "optimizer_status",
        "state",
        "blocked_reason",
    ]

    usecols = [c for c in need_cols if c in head]

    required = ["execution_date", "execution_datetime", "securityid", "mid_price"]
    missing = [c for c in required if c not in usecols]
    if missing:
        raise RuntimeError(f"missing required columns: {missing}")

    df = pd.read_csv(pos_path, usecols=usecols, low_memory=False)

    df["execution_date"] = df["execution_date"].astype(int)
    df["execution_datetime"] = pd.to_datetime(df["execution_datetime"].astype(str), errors="coerce")
    df["securityid"] = pd.to_numeric(df["securityid"], errors="coerce").astype("Int64")

    for c in [
        "mid_price",
        "bid_price",
        "ask_price",
        "spread_bps_realized",
        "spread_bps",
        "effective_target_weight",
        "target_weight",
        "desired_target_weight",
        "buy_edge_bps",
        "sell_edge_bps",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["execution_datetime", "securityid", "mid_price"]).copy()
    df["securityid"] = df["securityid"].astype(int)

    if "effective_target_weight" in df.columns:
        df["_target_weight"] = df["effective_target_weight"].fillna(0.0)
    elif "target_weight" in df.columns:
        df["_target_weight"] = df["target_weight"].fillna(0.0)
    elif "desired_target_weight" in df.columns:
        df["_target_weight"] = df["desired_target_weight"].fillna(0.0)
    else:
        raise RuntimeError("no target weight column found")

    df["_target_weight"] = df["_target_weight"].clip(lower=0.0)

    if "spread_bps_realized" in df.columns:
        df["_spread_bps"] = df["spread_bps_realized"]
    elif "spread_bps" in df.columns:
        df["_spread_bps"] = df["spread_bps"]
    else:
        df["_spread_bps"] = np.nan

    df = df.sort_values(["execution_datetime", "securityid"]).reset_index(drop=True)

    print("rows:", len(df))
    print("date:", df["execution_date"].min(), "->", df["execution_date"].max())
    print("minutes:", df["execution_datetime"].nunique())
    print("symbols:", df["securityid"].nunique())
    print("target weight describe:")
    print(df["_target_weight"].describe())

    cash = capital
    qty = defaultdict(float)
    last_mid = {}

    prev_equity = capital
    peak_equity = capital
    max_drawdown = 0.0

    minute_rows = []
    trade_rows = []

    total_turnover = 0.0
    total_fee = 0.0
    total_slippage = 0.0
    total_spread_cost = 0.0

    total_trade_events = 0
    total_buy_trades = 0
    total_sell_trades = 0
    total_blocked_gross_cap = 0
    total_blocked_small = 0
    total_blocked_spread = 0
    total_zero_target_exit = 0

    for dt, g in df.groupby("execution_datetime", sort=True):
        execution_date = int(g["execution_date"].iloc[0])

        for _, r in g.iterrows():
            sid = int(r["securityid"])
            mid = float(r["mid_price"])
            if np.isfinite(mid) and mid > 0:
                last_mid[sid] = mid

        def calc_portfolio():
            mv = 0.0
            gross = 0.0
            net = 0.0
            n = 0

            for sid0, q0 in qty.items():
                if abs(q0) < 1e-12:
                    continue

                mid0 = last_mid.get(sid0, np.nan)
                if not np.isfinite(mid0) or mid0 <= 0:
                    continue

                v = q0 * mid0
                mv += v
                gross += abs(v)
                net += v
                n += 1

            eq = cash + mv
            return mv, gross, net, n, eq

        market_value, gross_exposure, net_exposure, n_hold, equity_before = calc_portfolio()

        turnover = 0.0
        fee_sum = 0.0
        slippage_sum = 0.0
        spread_cost_sum = 0.0
        trade_events = 0
        buy_trades = 0
        sell_trades = 0
        blocked_gross_cap = 0
        blocked_small = 0
        blocked_spread = 0
        zero_target_exit = 0

        g2 = g.copy()
        g2["_current_qty"] = g2["securityid"].map(lambda x: qty[int(x)])
        g2["_exec_buy"] = g2["ask_price"].where(g2["ask_price"].notna(), g2["mid_price"])
        g2["_exec_sell"] = g2["bid_price"].where(g2["bid_price"].notna(), g2["mid_price"])

        target_qty_list = []
        delta_qty_list = []
        action_priority = []

        for _, r in g2.iterrows():
            sid = int(r["securityid"])
            tw = float(r["_target_weight"])
            mid = float(r["mid_price"])

            if not np.isfinite(mid) or mid <= 0:
                target_qty = 0.0
            else:
                target_notional = tw * capital
                target_qty = floor_lot(target_notional / mid, lot)

            cur = qty[sid]
            delta = target_qty - cur

            target_qty_list.append(target_qty)
            delta_qty_list.append(delta)

            if delta < 0:
                action_priority.append(0)
            else:
                action_priority.append(1)

        g2["_target_qty_v5"] = target_qty_list
        g2["_delta_qty_raw_v5"] = delta_qty_list
        g2["_priority"] = action_priority
        g2 = g2.sort_values(["_priority", "securityid"])

        for _, r in g2.iterrows():
            sid = int(r["securityid"])

            cur_qty = qty[sid]
            target_qty = float(r["_target_qty_v5"])
            raw_delta = target_qty - cur_qty

            if abs(raw_delta) < lot:
                continue

            spread_bps = r.get("_spread_bps", np.nan)
            mid = float(r["mid_price"])

            if raw_delta > 0:
                if gross_exposure / capital >= args.gross_cap:
                    blocked_gross_cap += 1
                    continue

                if np.isfinite(spread_bps) and spread_bps > args.entry_max_spread_bps:
                    blocked_spread += 1
                    continue

                exec_price = float(r["_exec_buy"])
                if not np.isfinite(exec_price) or exec_price <= 0:
                    continue

                desired_qty = floor_lot(raw_delta * args.entry_rebalance_ratio, lot)
                if desired_qty < lot:
                    desired_qty = lot

                remaining_gross_notional = max(args.gross_cap * capital - gross_exposure, 0.0)
                cap_qty = floor_lot(remaining_gross_notional / exec_price, lot)

                trade_qty = min(desired_qty, cap_qty)
                trade_qty = floor_lot(trade_qty, lot)

                side = "BUY"

            else:
                if np.isfinite(spread_bps) and spread_bps > args.exit_max_spread_bps:
                    blocked_spread += 1
                    continue

                exec_price = float(r["_exec_sell"])
                if not np.isfinite(exec_price) or exec_price <= 0:
                    continue

                reduce_need = abs(raw_delta)

                if target_qty <= 0:
                    desired_qty = floor_lot(cur_qty * args.exit_zero_ratio, lot)
                    zero_target_exit += 1
                else:
                    desired_qty = floor_lot(reduce_need * args.reduce_rebalance_ratio, lot)
                    if desired_qty < lot:
                        desired_qty = lot

                trade_qty = min(desired_qty, cur_qty)
                trade_qty = floor_lot(trade_qty, lot)

                side = "SELL"

            if trade_qty < lot:
                blocked_small += 1
                continue

            notional = trade_qty * exec_price
            if notional < args.min_trade_notional:
                blocked_small += 1
                continue

            fee = notional * args.fee_bps / 10000.0
            slippage = notional * args.slippage_bps / 10000.0

            if side == "BUY":
                dq = trade_qty
                cash -= notional
                buy_trades += 1
                spread_cost = max(exec_price - mid, 0.0) * trade_qty if np.isfinite(mid) else 0.0
            else:
                dq = -trade_qty
                cash += notional
                sell_trades += 1
                spread_cost = max(mid - exec_price, 0.0) * trade_qty if np.isfinite(mid) else 0.0

            cash -= fee + slippage
            qty[sid] += dq

            turnover += notional
            fee_sum += fee
            slippage_sum += slippage
            spread_cost_sum += spread_cost
            trade_events += 1

            trade_rows.append({
                "execution_datetime": dt,
                "execution_date": execution_date,
                "securityid": sid,
                "side": side,
                "qty": dq,
                "exec_price": exec_price,
                "notional": notional,
                "fee": fee,
                "slippage": slippage,
                "spread_cost_est": spread_cost,
                "target_weight": float(r["_target_weight"]),
                "target_qty": target_qty,
                "qty_after": qty[sid],
                "raw_delta_before_trade": raw_delta,
                "spread_bps": spread_bps,
            })

            market_value, gross_exposure, net_exposure, n_hold, equity_tmp = calc_portfolio()

        market_value, gross_exposure, net_exposure, n_hold, equity = calc_portfolio()

        step_net_pnl = equity - prev_equity

        peak_equity = max(peak_equity, equity)
        drawdown = equity - peak_equity
        max_drawdown = min(max_drawdown, drawdown)

        total_turnover += turnover
        total_fee += fee_sum
        total_slippage += slippage_sum
        total_spread_cost += spread_cost_sum

        total_trade_events += trade_events
        total_buy_trades += buy_trades
        total_sell_trades += sell_trades
        total_blocked_gross_cap += blocked_gross_cap
        total_blocked_small += blocked_small
        total_blocked_spread += blocked_spread
        total_zero_target_exit += zero_target_exit

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
            "spread_cost_est": spread_cost_sum,
            "trade_events": trade_events,
            "buy_trades": buy_trades,
            "sell_trades": sell_trades,
            "blocked_gross_cap": blocked_gross_cap,
            "blocked_small": blocked_small,
            "blocked_spread": blocked_spread,
            "zero_target_exit": zero_target_exit,
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

    trades = pd.DataFrame(trade_rows)
    trades.to_csv(out_trades, index=False)

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
            blocked_gross_cap=("blocked_gross_cap", "sum"),
            blocked_small=("blocked_small", "sum"),
            blocked_spread=("blocked_spread", "sum"),
            zero_target_exit=("zero_target_exit", "sum"),
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

    summary = {
        "start_date": int(daily["execution_date"].min()),
        "end_date": int(daily["execution_date"].max()),
        "capital": capital,
        "num_minutes": len(minute),
        "num_days": len(daily),
        "num_trade_events": int(total_trade_events),
        "num_buy_trades": int(total_buy_trades),
        "num_sell_trades": int(total_sell_trades),
        "total_turnover": total_turnover,
        "turnover_to_capital": total_turnover / capital,
        "total_fee": total_fee,
        "total_slippage": total_slippage,
        "total_spread_cost_est": total_spread_cost,
        "final_equity": float(minute["equity"].iloc[-1]),
        "total_net_pnl": float(minute["equity"].iloc[-1] - capital),
        "return_on_capital": float((minute["equity"].iloc[-1] - capital) / capital),
        "max_drawdown": float(max_drawdown),
        "avg_gross_weight": float(minute["gross_weight"].mean()),
        "avg_net_weight": float(minute["net_weight"].mean()),
        "avg_n_hold": float(minute["n_hold"].mean()),
        "total_blocked_gross_cap": int(total_blocked_gross_cap),
        "total_blocked_small": int(total_blocked_small),
        "total_blocked_spread": int(total_blocked_spread),
        "total_zero_target_exit": int(total_zero_target_exit),
        "gross_cap": args.gross_cap,
        "entry_rebalance_ratio": args.entry_rebalance_ratio,
        "reduce_rebalance_ratio": args.reduce_rebalance_ratio,
        "exit_zero_ratio": args.exit_zero_ratio,
        "min_trade_notional": args.min_trade_notional,
    }

    summary_df = pd.DataFrame.from_dict(summary, orient="index", columns=["value"])
    summary_df.to_csv(out_summary)

    print("\n===== v5 actual-qty-control summary =====")
    print(summary_df)

    print("\n===== daily =====")
    print(daily)

    print("\nsaved minute:", out_minute)
    print("saved daily :", out_daily)
    print("saved summary:", out_summary)
    print("saved trades:", out_trades)


if __name__ == "__main__":
    main()
