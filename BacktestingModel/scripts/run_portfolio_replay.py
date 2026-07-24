import argparse
import os
from collections import defaultdict

import numpy as np
import pandas as pd


def sid(x):
    return str(x).replace(".0", "").zfill(6)


def pick_mark_col(df, preferred="auto"):
    if preferred != "auto":
        if preferred not in df.columns:
            raise RuntimeError(f"mark price column not found: {preferred}")
        return preferred

    for c in ["decision_mid", "mid_price", "future_mid", "fill_price"]:
        if c in df.columns:
            return c

    raise RuntimeError("No mark price column found. Need one of decision_mid/mid_price/future_mid/fill_price.")


def finalize_day(
    daily_rows,
    date,
    model,
    day_start_equity,
    day_end_equity,
    day_num_trades,
    day_buy_trades,
    day_sell_trades,
    day_turnover,
    day_fee,
    day_max_gross,
    day_sum_gross,
    day_gross_obs,
    day_max_net_exposure,
    day_min_equity,
    day_max_equity,
    day_max_drawdown,
    current_cash,
    inventory_value,
    gross_exposure,
    net_exposure,
    capital,
    num_long_symbols,
    num_short_symbols,
):
    if date is None:
        return

    daily_pnl = day_end_equity - day_start_equity
    avg_gross = day_sum_gross / day_gross_obs if day_gross_obs else np.nan

    daily_rows.append({
        "model": model,
        "date": date,
        "num_trades": day_num_trades,
        "buy_trades": day_buy_trades,
        "sell_trades": day_sell_trades,
        "turnover": day_turnover,
        "fee": day_fee,
        "start_equity": day_start_equity,
        "end_equity": day_end_equity,
        "daily_pnl": daily_pnl,
        "daily_return_on_capital": daily_pnl / capital if capital else np.nan,
        "daily_return_on_turnover_bps": daily_pnl / day_turnover * 10000 if day_turnover else np.nan,
        "end_cash": current_cash,
        "end_inventory_value": inventory_value,
        "end_gross_exposure": gross_exposure,
        "end_net_exposure": net_exposure,
        "max_gross_exposure": day_max_gross,
        "avg_gross_exposure": avg_gross,
        "max_abs_net_exposure": day_max_net_exposure,
        "min_equity": day_min_equity,
        "max_equity": day_max_equity,
        "max_drawdown": day_max_drawdown,
        "num_long_symbols": num_long_symbols,
        "num_short_symbols": num_short_symbols,
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trades", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out-prefix", required=True)
    parser.add_argument("--capital", type=float, default=0.0)
    parser.add_argument("--initial-position-per-symbol", type=float, default=0.0)
    parser.add_argument("--mark-col", default="auto")
    parser.add_argument("--record-every", type=int, default=10000)
    parser.add_argument("--allow-short", type=int, default=1)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_prefix), exist_ok=True)
    os.makedirs("outputs/metrics", exist_ok=True)
    os.makedirs("outputs/portfolio", exist_ok=True)

    print("[1] loading trades:", args.trades)
    df = pd.read_csv(args.trades, low_memory=False)

    if len(df) == 0:
        raise RuntimeError("trades file is empty")

    df["fill_time"] = pd.to_datetime(df["fill_time"])
    df["decision_time"] = pd.to_datetime(df["decision_time"])
    df["securityid"] = df["securityid"].map(sid)
    df["side"] = df["side"].astype(str).str.upper()
    df["fill_price"] = pd.to_numeric(df["fill_price"], errors="coerce")
    df["fill_qty"] = pd.to_numeric(df["fill_qty"], errors="coerce")
    df["fee"] = pd.to_numeric(df["fee"], errors="coerce").fillna(0.0)

    mark_col = pick_mark_col(df, args.mark_col)
    df[mark_col] = pd.to_numeric(df[mark_col], errors="coerce")
    df["mark_price"] = df[mark_col].fillna(df["fill_price"])

    df = df.dropna(subset=["fill_time", "securityid", "side", "fill_price", "fill_qty", "mark_price"]).copy()
    df = df.sort_values(["fill_time", "securityid"]).reset_index(drop=True)

    print("trades shape:", df.shape)
    print("date range:", df["fill_time"].min(), "->", df["fill_time"].max())
    print("num securities:", df["securityid"].nunique())
    print("mark_col:", mark_col)
    print("capital:", args.capital)
    print("initial_position_per_symbol:", args.initial_position_per_symbol)
    print("allow_short:", args.allow_short)

    symbols = sorted(df["securityid"].unique())
    first_px = df.groupby("securityid")["mark_price"].first().to_dict()

    pos = defaultdict(float)
    last_px = {}
    pos_value = {}
    gross_value = {}

    cash = float(args.capital)
    inventory_value = 0.0
    gross_exposure = 0.0
    net_exposure = 0.0

    def set_mark(symbol, px):
        nonlocal inventory_value, gross_exposure, net_exposure

        old_val = pos_value.get(symbol, 0.0)
        old_gross = gross_value.get(symbol, 0.0)

        p = pos[symbol]
        new_val = p * px
        new_gross = abs(p) * px

        pos_value[symbol] = new_val
        gross_value[symbol] = new_gross
        last_px[symbol] = px

        inventory_value += new_val - old_val
        gross_exposure += new_gross - old_gross
        net_exposure = inventory_value

    # Initial inventory, marked at first observed price.
    if args.initial_position_per_symbol != 0:
        for s in symbols:
            px = float(first_px[s])
            pos[s] = float(args.initial_position_per_symbol)
            last_px[s] = px
            pos_value[s] = pos[s] * px
            gross_value[s] = abs(pos[s]) * px
            inventory_value += pos_value[s]
            gross_exposure += gross_value[s]

        # Treat initial inventory as bought at first mid, so starting PnL is zero.
        cash -= inventory_value
        net_exposure = inventory_value

    initial_equity = cash + inventory_value

    print("initial_cash:", cash)
    print("initial_inventory_value:", inventory_value)
    print("initial_equity:", initial_equity)
    print("initial_gross_exposure:", gross_exposure)

    daily_rows = []
    curve_rows = []

    current_date = None
    day_start_equity = None
    day_num_trades = 0
    day_buy_trades = 0
    day_sell_trades = 0
    day_turnover = 0.0
    day_fee = 0.0
    day_max_gross = 0.0
    day_sum_gross = 0.0
    day_gross_obs = 0
    day_max_net_exposure = 0.0
    day_min_equity = np.inf
    day_max_equity = -np.inf
    day_running_max_equity = -np.inf
    day_max_drawdown = 0.0

    total_turnover = 0.0
    total_fee = 0.0
    min_equity = np.inf
    max_equity = -np.inf
    running_max_equity = -np.inf
    max_drawdown = 0.0
    max_gross_exposure = gross_exposure
    max_abs_net_exposure = abs(net_exposure)
    num_short_events = 0
    num_short_violations = 0

    def equity():
        return cash + inventory_value

    for i, r in enumerate(df.itertuples(index=False), start=1):
        date = r.fill_time.strftime("%Y%m%d")

        if current_date is None:
            current_date = date
            day_start_equity = equity()
            day_min_equity = day_start_equity
            day_max_equity = day_start_equity
            day_running_max_equity = day_start_equity

        if date != current_date:
            long_symbols = sum(1 for v in pos.values() if v > 0)
            short_symbols = sum(1 for v in pos.values() if v < 0)

            finalize_day(
                daily_rows=daily_rows,
                date=current_date,
                model=args.model,
                day_start_equity=day_start_equity,
                day_end_equity=equity(),
                day_num_trades=day_num_trades,
                day_buy_trades=day_buy_trades,
                day_sell_trades=day_sell_trades,
                day_turnover=day_turnover,
                day_fee=day_fee,
                day_max_gross=day_max_gross,
                day_sum_gross=day_sum_gross,
                day_gross_obs=day_gross_obs,
                day_max_net_exposure=day_max_net_exposure,
                day_min_equity=day_min_equity,
                day_max_equity=day_max_equity,
                day_max_drawdown=day_max_drawdown,
                current_cash=cash,
                inventory_value=inventory_value,
                gross_exposure=gross_exposure,
                net_exposure=net_exposure,
                capital=args.capital,
                num_long_symbols=long_symbols,
                num_short_symbols=short_symbols,
            )

            current_date = date
            day_start_equity = equity()
            day_num_trades = 0
            day_buy_trades = 0
            day_sell_trades = 0
            day_turnover = 0.0
            day_fee = 0.0
            day_max_gross = gross_exposure
            day_sum_gross = 0.0
            day_gross_obs = 0
            day_max_net_exposure = abs(net_exposure)
            day_min_equity = day_start_equity
            day_max_equity = day_start_equity
            day_running_max_equity = day_start_equity
            day_max_drawdown = 0.0

        s = r.securityid
        px = float(r.mark_price)

        # Update existing inventory to current mark first.
        set_mark(s, px)

        qty = float(r.fill_qty)
        fill_price = float(r.fill_price)
        fee = float(r.fee)
        notional = fill_price * qty

        if r.side == "BUY":
            cash -= notional + fee
            pos[s] += qty
            day_buy_trades += 1
        elif r.side == "SELL":
            cash += notional - fee
            pos[s] -= qty
            day_sell_trades += 1
        else:
            raise RuntimeError(f"Unknown side: {r.side}")

        set_mark(s, px)

        if pos[s] < 0:
            num_short_events += 1
            if not args.allow_short:
                num_short_violations += 1

        e = equity()

        day_num_trades += 1
        day_turnover += notional
        day_fee += fee

        total_turnover += notional
        total_fee += fee

        max_gross_exposure = max(max_gross_exposure, gross_exposure)
        max_abs_net_exposure = max(max_abs_net_exposure, abs(net_exposure))
        min_equity = min(min_equity, e)
        max_equity = max(max_equity, e)

        running_max_equity = max(running_max_equity, e)
        max_drawdown = min(max_drawdown, e - running_max_equity)

        day_max_gross = max(day_max_gross, gross_exposure)
        day_sum_gross += gross_exposure
        day_gross_obs += 1
        day_max_net_exposure = max(day_max_net_exposure, abs(net_exposure))
        day_min_equity = min(day_min_equity, e)
        day_max_equity = max(day_max_equity, e)
        day_running_max_equity = max(day_running_max_equity, e)
        day_max_drawdown = min(day_max_drawdown, e - day_running_max_equity)

        if i % args.record_every == 0 or i == len(df):
            curve_rows.append({
                "model": args.model,
                "event_id": i,
                "time": r.fill_time,
                "date": date,
                "cash": cash,
                "inventory_value": inventory_value,
                "equity": e,
                "pnl": e - initial_equity,
                "gross_exposure": gross_exposure,
                "net_exposure": net_exposure,
                "turnover": total_turnover,
                "fee": total_fee,
                "max_drawdown": max_drawdown,
                "num_short_events": num_short_events,
                "num_short_violations": num_short_violations,
            })

    long_symbols = sum(1 for v in pos.values() if v > 0)
    short_symbols = sum(1 for v in pos.values() if v < 0)

    finalize_day(
        daily_rows=daily_rows,
        date=current_date,
        model=args.model,
        day_start_equity=day_start_equity,
        day_end_equity=equity(),
        day_num_trades=day_num_trades,
        day_buy_trades=day_buy_trades,
        day_sell_trades=day_sell_trades,
        day_turnover=day_turnover,
        day_fee=day_fee,
        day_max_gross=day_max_gross,
        day_sum_gross=day_sum_gross,
        day_gross_obs=day_gross_obs,
        day_max_net_exposure=day_max_net_exposure,
        day_min_equity=day_min_equity,
        day_max_equity=day_max_equity,
        day_max_drawdown=day_max_drawdown,
        current_cash=cash,
        inventory_value=inventory_value,
        gross_exposure=gross_exposure,
        net_exposure=net_exposure,
        capital=args.capital,
        num_long_symbols=long_symbols,
        num_short_symbols=short_symbols,
    )

    final_equity = equity()
    total_pnl = final_equity - initial_equity

    pos_rows = []
    for s in sorted(set(list(pos.keys()) + symbols)):
        p = pos[s]
        px = last_px.get(s, first_px.get(s, np.nan))
        pos_rows.append({
            "model": args.model,
            "securityid": s,
            "ending_position": p,
            "last_price": px,
            "ending_value": p * px if pd.notna(px) else np.nan,
            "abs_ending_value": abs(p) * px if pd.notna(px) else np.nan,
        })

    summary = pd.DataFrame([{
        "model": args.model,
        "trades_file": args.trades,
        "num_trades": len(df),
        "num_securities": len(symbols),
        "start_time": df["fill_time"].min(),
        "end_time": df["fill_time"].max(),
        "capital": args.capital,
        "initial_position_per_symbol": args.initial_position_per_symbol,
        "initial_equity": initial_equity,
        "final_cash": cash,
        "final_inventory_value": inventory_value,
        "final_equity": final_equity,
        "total_pnl": total_pnl,
        "total_turnover": total_turnover,
        "total_fee": total_fee,
        "pnl_bps_on_turnover": total_pnl / total_turnover * 10000 if total_turnover else np.nan,
        "return_on_capital": total_pnl / args.capital if args.capital else np.nan,
        "max_gross_exposure": max_gross_exposure,
        "return_on_max_gross_exposure": total_pnl / max_gross_exposure if max_gross_exposure else np.nan,
        "max_abs_net_exposure": max_abs_net_exposure,
        "min_equity": min_equity,
        "max_equity": max_equity,
        "max_drawdown": max_drawdown,
        "num_long_symbols_end": long_symbols,
        "num_short_symbols_end": short_symbols,
        "num_short_events": num_short_events,
        "num_short_violations_if_no_short": num_short_violations,
        "mark_col": mark_col,
    }])

    daily = pd.DataFrame(daily_rows)
    curve = pd.DataFrame(curve_rows)
    positions = pd.DataFrame(pos_rows)

    summary_path = f"{args.out_prefix}_summary.csv"
    daily_path = f"{args.out_prefix}_daily.csv"
    curve_path = f"{args.out_prefix}_curve.csv"
    position_path = f"{args.out_prefix}_positions.csv"

    summary.to_csv(summary_path, index=False)
    daily.to_csv(daily_path, index=False)
    curve.to_csv(curve_path, index=False)
    positions.to_csv(position_path, index=False)

    print("\n===== portfolio summary =====")
    print(summary.T.to_string())

    print("\n===== daily portfolio pnl =====")
    print(daily.to_string(index=False))

    print("\n===== ending positions top abs exposure =====")
    print(positions.sort_values("abs_ending_value", ascending=False).head(30).to_string(index=False))

    print("\nsaved:")
    print(summary_path)
    print(daily_path)
    print(curve_path)
    print(position_path)


if __name__ == "__main__":
    main()
