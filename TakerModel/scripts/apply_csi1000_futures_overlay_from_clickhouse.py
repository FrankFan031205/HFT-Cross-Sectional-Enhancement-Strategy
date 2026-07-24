# -*- coding: utf-8 -*-
import argparse
import json
import subprocess
import urllib.parse
import urllib.request
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def qident(x):
    return "`" + str(x).replace("`", "``") + "`"


def try_http(sql, fmt, host, port, user, password, timeout=120):
    q = sql.strip()
    if fmt:
        q += f" FORMAT {fmt}"

    url = f"http://{host}:{port}/"
    params = []
    if user:
        params.append(("user", user))
    if password:
        params.append(("password", password))
    if params:
        url += "?" + urllib.parse.urlencode(params)

    req = urllib.request.Request(url, data=q.encode("utf-8"), method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8")


def ch_query(sql, fmt="CSVWithNames", host="127.0.0.1", ports=(18123, 8123), user="default", password=""):
    q = sql.strip()
    if fmt:
        q_fmt = q + f" FORMAT {fmt}"
    else:
        q_fmt = q

    # Prefer clickhouse-client because your local HTTP port may be unavailable.
    try:
        return subprocess.check_output(
            ["clickhouse-client", "--query", q_fmt],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=180,
        )
    except Exception as e_client:
        last_err = e_client

    for port in ports:
        try:
            return try_http(q, fmt, host, port, user, password, timeout=180)
        except Exception as e_http:
            last_err = e_http

    raise RuntimeError(f"Cannot query ClickHouse via clickhouse-client or HTTP ports {ports}. Last error: {last_err}")


def ch_json(sql, **kwargs):
    txt = ch_query(sql, fmt="JSONEachRow", **kwargs)
    rows = []
    for line in txt.splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def ch_df(sql, **kwargs):
    txt = ch_query(sql, fmt="CSVWithNames", **kwargs)
    if not txt.strip():
        return pd.DataFrame()
    return pd.read_csv(StringIO(txt))


def pick_col(cols, candidates, required=True):
    lower_map = {c.lower(): c for c in cols}

    for c in candidates:
        if c in cols:
            return c
        if c.lower() in lower_map:
            return lower_map[c.lower()]

    if required:
        raise KeyError(f"cannot find any of {candidates}; columns={cols}")

    return None


def parse_datetime(raw_dt, raw_time, table_date):
    date_str = str(table_date)

    if raw_dt is not None:
        s = raw_dt.astype(str)

        dt = pd.to_datetime(s, errors="coerce")
        ok_ratio = dt.notna().mean()
        if ok_ratio > 0.8:
            return dt

        # Try numeric YYYYMMDDHHMMSS...
        z = s.str.replace(r"\.0$", "", regex=True).str.replace(r"\D", "", regex=True)
        z14 = z.str.slice(0, 14)
        dt = pd.to_datetime(z14, format="%Y%m%d%H%M%S", errors="coerce")
        if dt.notna().mean() > 0.8:
            return dt

        # Try HHMMSS / HHMMSSmmm
        hhmmss = z
        hhmmss_num = pd.to_numeric(hhmmss, errors="coerce")
        hhmmss = np.where(hhmmss_num > 235959, (hhmmss_num // 1000).astype("Int64").astype(str), hhmmss)
        hhmmss = pd.Series(hhmmss).astype(str).str.zfill(6).str.slice(0, 6)
        dt = pd.to_datetime(
            date_str + " " +
            hhmmss.str.slice(0, 2) + ":" +
            hhmmss.str.slice(2, 4) + ":" +
            hhmmss.str.slice(4, 6),
            errors="coerce",
        )
        if dt.notna().mean() > 0.8:
            return dt

    if raw_time is None:
        raise RuntimeError("cannot parse datetime: no datetime or time column")

    s = raw_time.astype(str)

    # Time string like 09:31:00.500
    if s.str.contains(":").mean() > 0.5:
        dt = pd.to_datetime(date_str + " " + s, errors="coerce")
        if dt.notna().mean() > 0.8:
            return dt

    # Time int like 93000000 / 093100
    z = s.str.replace(r"\.0$", "", regex=True).str.replace(r"\D", "", regex=True)
    z_num = pd.to_numeric(z, errors="coerce")
    z = np.where(z_num > 235959, (z_num // 1000).astype("Int64").astype(str), z)
    z = pd.Series(z).astype(str).str.zfill(6).str.slice(0, 6)

    dt = pd.to_datetime(
        date_str + " " +
        z.str.slice(0, 2) + ":" +
        z.str.slice(2, 4) + ":" +
        z.str.slice(4, 6),
        errors="coerce",
    )

    return dt


def get_tables(database, start_date, end_date, ch_kwargs):
    sql = f"""
    SELECT name
    FROM system.tables
    WHERE database = '{database}'
      AND match(name, '^20[0-9]{{6}}$')
      AND name >= '{start_date}'
      AND name <= '{end_date}'
    ORDER BY name
    """
    rows = ch_json(sql, **ch_kwargs)
    return [r["name"] for r in rows]


def get_columns(database, table, ch_kwargs):
    sql = f"""
    SELECT name, type
    FROM system.columns
    WHERE database = '{database}'
      AND table = '{table}'
    ORDER BY position
    """
    rows = ch_json(sql, **ch_kwargs)
    return [r["name"] for r in rows]



def choose_contract(database, table, sec_col, prefix, explicit_contract, cols, ch_kwargs):
    if explicit_contract:
        return explicit_contract

    turnover_col = pick_col(
        cols,
        ["Turnover", "turnover", "Amount", "amount", "TurnoverAmount", "turnover_amount"],
        required=False,
    )
    volume_col = pick_col(
        cols,
        ["Volume", "volume", "TotalVolume", "total_volume", "TradeVolume", "trade_volume"],
        required=False,
    )
    oi_col = pick_col(
        cols,
        ["OpenInterest", "openinterest", "open_interest"],
        required=False,
    )

    if turnover_col:
        score_expr = f"max(toFloat64OrZero(toString({qident(turnover_col)})))"
        score_name = f"max_{turnover_col}"
    elif volume_col:
        score_expr = f"max(toFloat64OrZero(toString({qident(volume_col)})))"
        score_name = f"max_{volume_col}"
    elif oi_col:
        score_expr = f"avg(toFloat64OrZero(toString({qident(oi_col)})))"
        score_name = f"avg_{oi_col}"
    else:
        score_expr = "count()"
        score_name = "cnt"

    sql = f"""
    SELECT
        toString({qident(sec_col)}) AS contract,
        count() AS cnt,
        {score_expr} AS score
    FROM {qident(database)}.{qident(table)}
    WHERE startsWith(toString({qident(sec_col)}), '{prefix}')
    GROUP BY contract
    ORDER BY score DESC, cnt DESC, contract ASC
    LIMIT 1
    """

    rows = ch_json(sql, **ch_kwargs)

    if not rows:
        return None

    chosen = rows[0]["contract"]
    print(f"[choose_contract] {table}: {chosen}, score_by={score_name}, score={rows[0].get('score')}, cnt={rows[0].get('cnt')}")
    return chosen


def export_futures_minute(database, start_date, end_date, prefix, contract, out_csv, ch_kwargs):
    tables = get_tables(database, start_date, end_date, ch_kwargs)
    if not tables:
        raise RuntimeError(f"no date tables found in {database} between {start_date} and {end_date}")

    all_rows = []

    for table in tables:
        cols = get_columns(database, table, ch_kwargs)

        sec_col = pick_col(cols, ["SecurityID", "securityid", "InstrumentID", "instrumentid", "symbol", "Symbol"])

        dt_col = pick_col(cols, ["datetime", "DateTime", "dateTime", "timestamp", "Timestamp", "LocalTime", "localtime"], required=False)
        time_col = pick_col(cols, ["time", "Time", "UpdateTime", "updatetime", "UpdateMillisec", "TradingTime"], required=False)

        price_col = pick_col(cols, ["LastPrice", "last_price", "lastprice", "price", "Price", "ClosePrice", "close"], required=False)
        bid_col = pick_col(cols, ["BidPrice1", "bidprice1", "bid1", "bid_price1", "bid_price", "Bid1"], required=False)
        ask_col = pick_col(cols, ["AskPrice1", "askprice1", "ask1", "ask_price1", "ask_price", "Ask1"], required=False)

        if price_col is None and (bid_col is None or ask_col is None):
            print(f"[skip] {table}: cannot find price or bid/ask columns")
            continue

        chosen = choose_contract(database, table, sec_col, prefix, contract, cols, ch_kwargs)
        if chosen is None:
            print(f"[skip] {table}: no contract starts with {prefix}")
            continue

        exprs = [
            f"toString({qident(sec_col)}) AS SecurityID",
        ]

        if dt_col:
            exprs.append(f"{qident(dt_col)} AS raw_datetime")
        if time_col:
            exprs.append(f"{qident(time_col)} AS raw_time")
        if price_col:
            exprs.append(f"{qident(price_col)} AS raw_price")
        if bid_col:
            exprs.append(f"{qident(bid_col)} AS raw_bid")
        if ask_col:
            exprs.append(f"{qident(ask_col)} AS raw_ask")

        order_col = dt_col or time_col or sec_col

        sql = f"""
        SELECT
            {", ".join(exprs)}
        FROM {qident(database)}.{qident(table)}
        WHERE toString({qident(sec_col)}) = '{chosen}'
        ORDER BY {qident(order_col)}
        """

        df = ch_df(sql, **ch_kwargs)
        if df.empty:
            print(f"[skip] {table}: empty for {chosen}")
            continue

        raw_dt = df["raw_datetime"] if "raw_datetime" in df.columns else None
        raw_time = df["raw_time"] if "raw_time" in df.columns else None

        df["datetime"] = parse_datetime(raw_dt, raw_time, table)
        df = df.dropna(subset=["datetime"]).copy()

        if df.empty:
            print(f"[skip] {table}: datetime parse empty")
            continue

        if "raw_bid" in df.columns and "raw_ask" in df.columns:
            bid = pd.to_numeric(df["raw_bid"], errors="coerce")
            ask = pd.to_numeric(df["raw_ask"], errors="coerce")
            mid = (bid + ask) / 2.0
            valid_mid = (bid > 0) & (ask > 0) & (ask >= bid)
            df["futures_price"] = np.where(valid_mid, mid, np.nan)
        else:
            df["futures_price"] = np.nan

        if "raw_price" in df.columns:
            last = pd.to_numeric(df["raw_price"], errors="coerce")
            df["futures_price"] = pd.Series(df["futures_price"]).fillna(last)

        df["futures_price"] = pd.to_numeric(df["futures_price"], errors="coerce")
        df = df.dropna(subset=["futures_price"])
        df = df[df["futures_price"] > 0].copy()

        if df.empty:
            print(f"[skip] {table}: no valid price")
            continue

        df["date"] = int(table)
        df["contract"] = chosen
        df["datetime"] = df["datetime"].dt.floor("min")

        minute = (
            df.sort_values("datetime")
              .groupby(["date", "datetime", "contract"], as_index=False)
              .agg(futures_price=("futures_price", "last"))
        )

        minute["futures_ret"] = minute.groupby("date")["futures_price"].pct_change()
        minute["futures_ret"] = minute["futures_ret"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

        all_rows.append(minute)

        print(f"[ok] {table} contract={chosen} rows={len(minute)}")

    if not all_rows:
        raise RuntimeError("no futures minute data exported")

    out = pd.concat(all_rows, ignore_index=True)
    out = out.sort_values(["datetime", "contract"]).drop_duplicates("datetime", keep="last")
    out = out[["date", "datetime", "contract", "futures_price", "futures_ret"]]

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print("\n===== futures export summary =====")
    print("rows:", len(out))
    print("date:", out["date"].min(), "->", out["date"].max())
    print("datetime:", out["datetime"].min(), "->", out["datetime"].max())
    print("contracts:")
    print(out["contract"].value_counts().to_string())
    print("[saved]", out_path)

    return out


def compound_curve(x):
    x = pd.Series(x).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return (1.0 + x).cumprod() - 1.0


def compound_return(x):
    x = pd.Series(x).replace([np.inf, -np.inf], np.nan).dropna()
    if x.empty:
        return np.nan
    return float((1.0 + x).prod() - 1.0)


def make_ticks(df):
    first = df.groupby("date", as_index=False)["bar_index"].min()
    n = len(first)
    step = 1 if n <= 12 else 2 if n <= 24 else max(1, n // 12)
    ticks = first.iloc[::step]
    return ticks["bar_index"].tolist(), ticks["date"].astype(str).tolist()


def apply_overlay(curve_csv, futures_csv, out_dir, target_total_gross, leverage, fee_rate, allow_short_futures):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    curve = pd.read_csv(curve_csv, low_memory=False)
    fut = pd.read_csv(futures_csv, low_memory=False)

    curve["datetime"] = pd.to_datetime(curve["datetime"])
    fut["datetime"] = pd.to_datetime(fut["datetime"])

    # Align futures to the stock curve minute grid first, then recompute futures_ret.
    # This avoids using returns between non-stock minutes such as pre-open/lunch/extra futures timestamps.
    fut = fut[["datetime", "futures_price", "contract"]].drop_duplicates("datetime").sort_values("datetime")
    df = curve.merge(fut, on="datetime", how="left")

    df = df.sort_values(["date", "datetime"]).reset_index(drop=True)
    df["futures_price"] = df["futures_price"].ffill()
    df["contract"] = df["contract"].ffill()

    missing_price = int(df["futures_price"].isna().sum())
    if missing_price > 0:
        print(f"[WARN] missing futures_price after merge/ffill: {missing_price}")

    df["futures_ret"] = df.groupby("date")["futures_price"].pct_change()
    df["futures_ret"] = df["futures_ret"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if "held_gross_prev" in df.columns:
        gross_col = "held_gross_prev"
    elif "target_gross" in df.columns:
        gross_col = "target_gross"
    else:
        raise KeyError("curve csv must contain held_gross_prev or target_gross")

    df["stock_gross"] = pd.to_numeric(df[gross_col], errors="coerce").fillna(0.0)

    raw_notional = float(target_total_gross) - df["stock_gross"]
    if int(allow_short_futures) == 1:
        df["futures_notional"] = raw_notional
    else:
        df["futures_notional"] = raw_notional.clip(lower=0.0)

    prev_notional = df["futures_notional"].shift(1)
    prev_notional.iloc[0] = df["futures_notional"].iloc[0]

    df["futures_turnover"] = (df["futures_notional"] - prev_notional).abs()
    df["futures_margin_used"] = df["futures_notional"].abs() / float(leverage)

    df["futures_pnl_ret"] = df["futures_notional"] * df["futures_ret"]
    df["futures_fee_ret"] = df["futures_turnover"] * float(fee_rate)

    df["actual_ret_with_futures"] = (
        df["actual_ret"]
        + df["futures_pnl_ret"]
        - df["futures_fee_ret"]
    )

    df["actualret_with_futures"] = compound_curve(df["actual_ret_with_futures"])
    df["benchmarkret_full"] = compound_curve(df["benchmark_ret"])
    df["alpharet_with_futures"] = df["actualret_with_futures"] - df["benchmarkret_full"]

    df["alpha_ret_with_futures"] = df["actual_ret_with_futures"] - df["benchmark_ret"]
    df["alpharet_with_futures_compound"] = compound_curve(df["alpha_ret_with_futures"])

    if "actualret_clean" in df.columns:
        df["actualret_no_futures"] = df["actualret_clean"]
    elif "actualret" in df.columns:
        df["actualret_no_futures"] = df["actualret"]
    else:
        df["actualret_no_futures"] = compound_curve(df["actual_ret"])

    if "benchmarkret_clean" in df.columns:
        df["benchmarkret_no_futures"] = df["benchmarkret_clean"]
    elif "benchmarkret" in df.columns:
        df["benchmarkret_no_futures"] = df["benchmarkret"]
    else:
        df["benchmarkret_no_futures"] = compound_curve(df["benchmark_ret"])

    df["alpharet_no_futures"] = df["actualret_no_futures"] - df["benchmarkret_no_futures"]

    df["bar_index"] = np.arange(len(df))

    out_csv = out_dir / "curve_with_real_csi1000_futures_overlay.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    daily = (
        df.groupby("date", as_index=False)
        .agg(
            actual_return_with_futures=("actual_ret_with_futures", compound_return),
            benchmark_return=("benchmark_ret", compound_return),
            stock_actual_return=("actual_ret", compound_return),
            avg_stock_gross=("stock_gross", "mean"),
            avg_futures_notional=("futures_notional", "mean"),
            avg_margin_used=("futures_margin_used", "mean"),
            futures_turnover=("futures_turnover", "sum"),
            futures_fee_ret=("futures_fee_ret", "sum"),
            n_minutes=("datetime", "nunique"),
        )
    )

    daily["excess_return_with_futures"] = daily["actual_return_with_futures"] - daily["benchmark_return"]
    daily["actualret_with_futures"] = compound_curve(daily["actual_return_with_futures"])
    daily["benchmarkret"] = compound_curve(daily["benchmark_return"])
    daily["alpharet_with_futures"] = daily["actualret_with_futures"] - daily["benchmarkret"]

    ex = daily["excess_return_with_futures"].dropna()
    if len(ex) >= 2 and ex.std(ddof=1) > 0:
        daily_sharpe = ex.mean() / ex.std(ddof=1) * np.sqrt(252)
        daily_tstat = ex.mean() / ex.std(ddof=1) * np.sqrt(len(ex))
    else:
        daily_sharpe = np.nan
        daily_tstat = np.nan

    daily["daily_excess_sharpe_with_futures"] = daily_sharpe
    daily["daily_excess_tstat_with_futures"] = daily_tstat

    daily_csv = out_dir / "daily_with_real_csi1000_futures_overlay.csv"
    daily.to_csv(daily_csv, index=False, encoding="utf-8-sig")

    tick_pos, tick_lab = make_ticks(df)

    plt.figure(figsize=(14, 6))
    plt.plot(df["bar_index"], df["actualret_with_futures"] * 100.0, label="actualret_with_futures")
    plt.plot(df["bar_index"], df["benchmarkret_full"] * 100.0, label="benchmarkret_full")
    plt.plot(df["bar_index"], df["alpharet_with_futures"] * 100.0, label="alpharet_with_futures")
    plt.axhline(0.0, linewidth=0.8)
    plt.xticks(tick_pos, tick_lab, rotation=45)
    plt.title("Pure-CS NAV with Real CSI1000 Futures Overlay")
    plt.xlabel("trading minute index")
    plt.ylabel("cumulative return (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "curve_with_real_csi1000_futures_overlay.png", dpi=180)
    plt.close()

    plt.figure(figsize=(14, 5))
    plt.plot(df["bar_index"], df["alpharet_no_futures"] * 100.0, label="alpha_no_futures")
    plt.plot(df["bar_index"], df["alpharet_with_futures"] * 100.0, label="alpha_with_futures")
    plt.axhline(0.0, linewidth=0.8)
    plt.xticks(tick_pos, tick_lab, rotation=45)
    plt.title("Alpha Before and After Real CSI1000 Futures Overlay")
    plt.xlabel("trading minute index")
    plt.ylabel("cumulative alpha (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "alpha_before_after_real_csi1000_futures_overlay.png", dpi=180)
    plt.close()

    summary = pd.DataFrame([{
        "target_total_gross": float(target_total_gross),
        "leverage": float(leverage),
        "futures_fee_rate": float(fee_rate),
        "avg_stock_gross": df["stock_gross"].mean(),
        "avg_futures_notional": df["futures_notional"].mean(),
        "avg_margin_used": df["futures_margin_used"].mean(),
        "total_futures_turnover": df["futures_turnover"].sum(),
        "total_futures_fee_return": df["futures_fee_ret"].sum(),
        "final_actualret_no_futures": df["actualret_no_futures"].iloc[-1],
        "final_alpharet_no_futures": df["alpharet_no_futures"].iloc[-1],
        "final_actualret_with_futures": df["actualret_with_futures"].iloc[-1],
        "final_benchmarkret_full": df["benchmarkret_full"].iloc[-1],
        "final_alpharet_with_futures": df["alpharet_with_futures"].iloc[-1],
        "final_alpharet_with_futures_compound": df["alpharet_with_futures_compound"].iloc[-1],
        "daily_excess_sharpe_with_futures": daily_sharpe,
        "daily_excess_tstat_with_futures": daily_tstat,
        "n_minutes": len(df),
        "n_days": daily["date"].nunique(),
    }])

    summary_csv = out_dir / "summary_with_real_csi1000_futures_overlay.csv"
    summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    print("\n===== real CSI1000 futures overlay summary =====")
    print(summary.to_string(index=False))
    print("\n[saved]", summary_csv)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--database", default="stock_index_main_500ms_v2")
    ap.add_argument("--start-date", type=int, required=True)
    ap.add_argument("--end-date", type=int, required=True)
    ap.add_argument("--contract-prefix", default="IM")
    ap.add_argument("--contract", default="")
    ap.add_argument("--futures-out", required=True)

    ap.add_argument("--curve-csv", required=True)
    ap.add_argument("--out-dir", required=True)

    ap.add_argument("--target-total-gross", type=float, default=1.0)
    ap.add_argument("--leverage", type=float, default=10.0)
    ap.add_argument("--futures-fee-rate", type=float, default=0.000023)
    ap.add_argument("--allow-short-futures", type=int, default=0)

    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--ports", default="18123,8123")
    ap.add_argument("--user", default="default")
    ap.add_argument("--password", default="")

    args = ap.parse_args()

    ch_kwargs = {
        "host": args.host,
        "ports": tuple(int(x) for x in args.ports.split(",") if x.strip()),
        "user": args.user,
        "password": args.password,
    }

    export_futures_minute(
        database=args.database,
        start_date=args.start_date,
        end_date=args.end_date,
        prefix=args.contract_prefix,
        contract=args.contract.strip() or None,
        out_csv=args.futures_out,
        ch_kwargs=ch_kwargs,
    )

    apply_overlay(
        curve_csv=args.curve_csv,
        futures_csv=args.futures_out,
        out_dir=args.out_dir,
        target_total_gross=args.target_total_gross,
        leverage=args.leverage,
        fee_rate=args.futures_fee_rate,
        allow_short_futures=args.allow_short_futures,
    )


if __name__ == "__main__":
    main()
