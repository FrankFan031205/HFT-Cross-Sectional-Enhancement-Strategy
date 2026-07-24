import sys
import importlib
from pathlib import Path

import pandas as pd


def _add_project_paths():
    root = Path(__file__).resolve().parents[2]
    candidates = [
        root,
        root.parent,
        root / "FactorGenerate",
        root / "PricingModel",
        root / "MarketMakingModel",
    ]
    for p in candidates:
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)


def get_clickhouse_client(cfg, date=None):
    ch_cfg = cfg.get("clickhouse_connect", {})
    host = ch_cfg.get("host", "")

    if host:
        import clickhouse_connect
        return clickhouse_connect.get_client(
            host=host,
            port=int(ch_cfg.get("port", 8123)),
            username=ch_cfg.get("username") or None,
            password=ch_cfg.get("password") or None,
            database=ch_cfg.get("database") or None,
        )

    _add_project_paths()

    module_names = [
        "data_loader",
        "config.data_loader",
        "src.data_loader",
        "utils.data_loader",
    ]

    last_errors = []

    for mod_name in module_names:
        try:
            mod = importlib.import_module(mod_name)
            if hasattr(mod, "get_clickhouse_client"):
                fn = getattr(mod, "get_clickhouse_client")
                for args in [(date,), (str(date),), ()]:
                    try:
                        return fn(*args)
                    except Exception as e:
                        last_errors.append((mod_name, args, repr(e)))
        except Exception as e:
            last_errors.append((mod_name, "import", repr(e)))

    raise RuntimeError(
        "Cannot create ClickHouse client. "
        f"Last errors: {last_errors[:5]}"
    )


def query_df(client, sql):
    if hasattr(client, "query_df"):
        return client.query_df(sql)

    if hasattr(client, "query_dataframe"):
        return client.query_dataframe(sql)

    if hasattr(client, "execute"):
        try:
            data, columns = client.execute(sql, with_column_types=True)
            col_names = [c[0] for c in columns]
            return pd.DataFrame(data, columns=col_names)
        except TypeError:
            data = client.execute(sql)
            return pd.DataFrame(data)

    raise RuntimeError("Unsupported ClickHouse client type.")


def format_symbols(symbols):
    """
    Trade table SecurityID is Int32, so query with integer ids:
        000056 -> 56
        000001 -> 1
    Python side will zfill back to 6 digits later.
    """
    xs = []
    for s in symbols:
        s = str(s).replace(".0", "").strip()
        if not s:
            continue
        xs.append(str(int(s)))
    return ",".join(xs)


def _get_trade_table_for_date(cfg, date):
    db_cfg = cfg["db"]
    database = db_cfg["trade_database"]
    table_template = db_cfg.get("trade_table_template", "{date}")
    table = table_template.format(date=int(date))
    return database, table


def _to_trade_time_int(ts):
    """
    Convert pandas Timestamp to HHMMSSmmm int.
    Example:
        2024-10-22 09:30:00.000 -> 93000000
        2024-10-22 09:30:00.500 -> 93000500
    """
    ts = pd.Timestamp(ts)
    return int(ts.strftime("%H%M%S") + f"{ts.microsecond // 1000:03d}")


def _make_datetime_from_trade_time(date, time_series):
    """
    date: 20241022
    time: 93000000 -> 20241022_093000000
    """
    date_str = str(int(date))
    t = time_series.astype(str).str.replace(".0", "", regex=False).str.zfill(9)
    s = date_str + "_" + t
    return pd.to_datetime(s, format="%Y%m%d_%H%M%S%f", errors="coerce")


def fetch_trades_from_db(client, quotes, cfg):
    db_cfg = cfg["db"]
    template = db_cfg["trade_sql_template"]

    q = quotes.copy()
    q["datetime"] = pd.to_datetime(q["datetime"])
    q["date_key"] = q["datetime"].dt.strftime("%Y%m%d").astype(int)

    latency_ms = int(cfg["backtest"]["latency_ms"])
    ttl_ms = int(cfg["backtest"]["quote_ttl_ms"])

    dfs = []

    for date, g in q.groupby("date_key"):
        database, table = _get_trade_table_for_date(cfg, date)

        symbols = sorted(
            g["securityid"]
            .astype(str)
            .str.replace(".0", "", regex=False)
            .str.zfill(6)
            .unique()
        )
        symbols_sql = format_symbols(symbols)

        start_time = g["datetime"].min() + pd.Timedelta(milliseconds=latency_ms)
        end_time = g["datetime"].max() + pd.Timedelta(milliseconds=latency_ms + ttl_ms)

        start_time_int = _to_trade_time_int(start_time)
        end_time_int = _to_trade_time_int(end_time)

        sql = template.format(
            database=database,
            table=table,
            date=int(date),
            symbols=symbols_sql,
            start_time=start_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            end_time=end_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            start_time_int=start_time_int,
            end_time_int=end_time_int,
        )

        print(f"[DB] querying trades database={database}, table={table}, symbols={len(symbols)}")
        print(f"[DB] time int range: {start_time_int} -> {end_time_int}")

        df = query_df(client, sql)
        print(f"[DB] date={date}, rows={len(df)}")

        if len(df) > 0:
            if "datetime" not in df.columns:
                if "time" not in df.columns:
                    raise RuntimeError("trade query returned neither datetime nor time")
                df["datetime"] = _make_datetime_from_trade_time(date, df["time"])

            dfs.append(df)

    if not dfs:
        return pd.DataFrame(columns=["datetime", "securityid", "price", "qty", "side", "seqno"])

    return pd.concat(dfs, ignore_index=True)
