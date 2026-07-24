import os
import argparse
import yaml
import numpy as np
import pandas as pd

from utils import data_loader


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def ensure_dir(path):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def normalize_securityid(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s.zfill(6)


def get_clickhouse_client(date):
    if hasattr(data_loader, "get_clickhouse_client"):
        try:
            return data_loader.get_clickhouse_client(date)
        except TypeError:
            return data_loader.get_clickhouse_client()

    if hasattr(data_loader, "get_client"):
        try:
            return data_loader.get_client(date)
        except TypeError:
            return data_loader.get_client()

    candidates = [
        "clickhouse_client",
        "client",
        "ck_client",
        "ch_client",
        "conn",
    ]

    for name in candidates:
        if hasattr(data_loader, name):
            obj = getattr(data_loader, name)
            if obj is not None:
                return obj

    available = [x for x in dir(data_loader) if not x.startswith("_")]
    raise RuntimeError(
        "Cannot find ClickHouse client in data_loader. "
        f"Available attributes: {available}"
    )


def query_df(client, sql):
    if hasattr(client, "query_df"):
        return client.query_df(sql)

    if hasattr(client, "query_dataframe"):
        return client.query_dataframe(sql)

    if hasattr(client, "query"):
        res = client.query(sql)

        if hasattr(res, "result_df"):
            return res.result_df

        if hasattr(res, "result_rows") and hasattr(res, "column_names"):
            return pd.DataFrame(res.result_rows, columns=res.column_names)

    if hasattr(client, "execute"):
        try:
            data, columns = client.execute(sql, with_column_types=True)
            col_names = [c[0] for c in columns]
            return pd.DataFrame(data, columns=col_names)
        except TypeError:
            data = client.execute(sql)
            return pd.DataFrame(data)

    raise RuntimeError("Unsupported ClickHouse client type")


def quote_sql_value(x, securityid_type):
    if securityid_type == "int":
        return str(int(str(x)))
    return "'" + str(x) + "'"


def load_universe(cfg):
    ucfg = cfg["universe"]

    path = ucfg["source_csv"]
    date_col = ucfg.get("date_col", "date")
    sec_col = ucfg.get("securityid_col", "securityid")

    print("loading universe:", path, flush=True)

    df = pd.read_csv(
        path,
        usecols=[date_col, sec_col],
        dtype={sec_col: str},
    )

    df[date_col] = df[date_col].astype(int)
    df[sec_col] = df[sec_col].map(normalize_securityid)

    df = df.dropna()
    df = df.drop_duplicates([date_col, sec_col])

    start = int(cfg["date"]["start"])
    end = int(cfg["date"]["end"])

    df = df[(df[date_col] >= start) & (df[date_col] <= end)].copy()

    print("universe date range:", df[date_col].min(), df[date_col].max(), flush=True)
    print("universe date-security pairs:", len(df), flush=True)
    print("universe stocks:", df[sec_col].nunique(), flush=True)

    universe = {}
    for d, g in df.groupby(date_col):
        universe[int(d)] = sorted(g[sec_col].unique().tolist())

    return universe


def build_market_query(cfg, date, securityids):
    db = cfg["database"]

    table = str(db["table"]).format(date=date)

    sec_col = db["securityid_col"]
    time_col = db["time_col"]

    date_filter_col = db.get("date_filter_col", "")
    securityid_type = db.get("securityid_type", "string")

    ids = ", ".join([quote_sql_value(x, securityid_type) for x in securityids])

    where_parts = [f"{sec_col} IN ({ids})"]

    if date_filter_col:
        where_parts.append(f"{date_filter_col} = {date}")

    where_sql = " AND ".join(where_parts)

    select_cols = [
        f"{sec_col} AS securityid",
        f"{time_col} AS time",
    ]

    for level in range(1, 11):
        select_cols.append(f"bidprice{level} AS bid{level}")
        select_cols.append(f"askprice{level} AS ask{level}")
        select_cols.append(f"bidvolume{level} AS bid{level}_volume")
        select_cols.append(f"askvolume{level} AS ask{level}_volume")

    select_sql = ",\n        ".join(select_cols)

    sql = f"""
    SELECT
        {select_sql}
    FROM {table}
    WHERE {where_sql}
    ORDER BY securityid, time
    """

    return sql


def build_limit_query(cfg, date, securityids):
    lcfg = cfg.get("limit", {})
    if not lcfg.get("enable", False):
        return None

    table = str(lcfg["table"]).format(date=date)
    sec_col = lcfg["securityid_col"]
    up_col = lcfg["limit_up_col"]
    down_col = lcfg["limit_down_col"]

    securityid_type = lcfg.get(
        "securityid_type",
        cfg["database"].get("securityid_type", "string"),
    )

    ids = ", ".join([quote_sql_value(x, securityid_type) for x in securityids])

    sql = f"""
    SELECT
        {sec_col} AS securityid,
        {up_col} AS limit_up_price,
        {down_col} AS limit_down_price
    FROM {table}
    WHERE {sec_col} IN ({ids})
    """

    return sql


def clean_market_df(df, date, price_scale=1.0):
    needed = ["securityid", "time", "bid1", "ask1"]
    missing = [c for c in needed if c not in df.columns]

    if missing:
        raise RuntimeError(
            f"query result missing columns: {missing}; got columns={list(df.columns)}"
        )

    out = df.copy()

    out["date"] = int(date)
    out["securityid"] = out["securityid"].map(normalize_securityid)

    raw_time = out["time"].astype(str).str.strip()
    digits = raw_time.str.replace(r"\D", "", regex=True)

    time_str = digits.str[-9:].str.zfill(9)
    out["time"] = time_str.astype(int)

    price_cols = []
    volume_cols = []

    for level in range(1, 11):
        bid_col = f"bid{level}"
        ask_col = f"ask{level}"
        bid_vol_col = f"bid{level}_volume"
        ask_vol_col = f"ask{level}_volume"

        price_cols.extend([bid_col, ask_col])
        volume_cols.extend([bid_vol_col, ask_vol_col])

        if bid_col not in out.columns:
            out[bid_col] = np.nan

        if ask_col not in out.columns:
            out[ask_col] = np.nan

        if bid_vol_col not in out.columns:
            out[bid_vol_col] = 0

        if ask_vol_col not in out.columns:
            out[ask_vol_col] = 0

    for c in price_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    for c in volume_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=["securityid", "time", "bid1", "ask1"])

    if price_scale and float(price_scale) != 1.0:
        for c in price_cols:
            out[c] = out[c] / float(price_scale)

    out = out[(out["bid1"] > 0) & (out["ask1"] > 0)]
    out = out[out["ask1"] >= out["bid1"]]

    out["mid_price"] = (out["bid1"] + out["ask1"]) / 2.0
    out["spread"] = out["ask1"] - out["bid1"]

    out["datetime"] = (
        out["date"].astype(str)
        + "_"
        + out["time"].astype(str).str.zfill(9)
    )

    keep_cols = [
        "date",
        "securityid",
        "time",
        "datetime",
    ]

    for level in range(1, 11):
        keep_cols.extend([
            f"bid{level}",
            f"ask{level}",
            f"bid{level}_volume",
            f"ask{level}_volume",
        ])

    keep_cols.extend([
        "mid_price",
        "spread",
    ])

    out = out[keep_cols]
    out = out.sort_values(["date", "securityid", "time"]).reset_index(drop=True)

    return out


def clean_limit_df(df, price_scale=1.0):
    needed = ["securityid", "limit_up_price", "limit_down_price"]
    missing = [c for c in needed if c not in df.columns]

    if missing:
        raise RuntimeError(
            f"limit query result missing columns: {missing}; got columns={list(df.columns)}"
        )

    out = df[needed].copy()

    out["securityid"] = out["securityid"].map(normalize_securityid)
    out["limit_up_price"] = pd.to_numeric(out["limit_up_price"], errors="coerce")
    out["limit_down_price"] = pd.to_numeric(out["limit_down_price"], errors="coerce")

    if price_scale and float(price_scale) != 1.0:
        out["limit_up_price"] = out["limit_up_price"] / float(price_scale)
        out["limit_down_price"] = out["limit_down_price"] / float(price_scale)

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.drop_duplicates("securityid")

    return out


def add_limit_prices(market_df, client, cfg, date, securityids):
    lcfg = cfg.get("limit", {})

    if not lcfg.get("enable", False):
        market_df["limit_up_price"] = np.nan
        market_df["limit_down_price"] = np.nan
        return market_df

    sql = build_limit_query(cfg, date, securityids)
    limit_df = query_df(client, sql)

    if limit_df is None or len(limit_df) == 0:
        market_df["limit_up_price"] = np.nan
        market_df["limit_down_price"] = np.nan
        return market_df

    price_scale = float(lcfg.get("price_scale", 1.0))
    limit_df = clean_limit_df(limit_df, price_scale=price_scale)

    market_df = market_df.merge(
        limit_df,
        on="securityid",
        how="left",
    )

    return market_df


def add_forward_returns(df, horizons):
    df = df.sort_values(["date", "securityid", "time"]).copy()

    for h in horizons:
        future_mid = df.groupby(["date", "securityid"])["mid_price"].shift(-h)
        ret = (future_mid - df["mid_price"]) / df["mid_price"]

        df[f"ret_{h}"] = ret
        df[f"label_{h}"] = ret

    return df


def append_csv(df, path):
    ensure_dir(path)
    header = not os.path.exists(path)
    df.to_csv(path, mode="a", header=header, index=False)


def append_error(path, msg):
    ensure_dir(path)
    with open(path, "a") as f:
        f.write(str(msg) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    out_path = cfg["output"]["csv_path"]
    err_path = cfg["output"]["error_log_path"]

    horizons = [int(x) for x in cfg["return"]["horizons"]]
    drop_nan_returns = bool(cfg["return"].get("drop_nan_returns", True))

    chunk_size = int(cfg["database"].get("query_chunk_size", 50))
    price_scale = float(cfg["database"].get("price_scale", 1.0))

    ensure_dir(out_path)
    ensure_dir(err_path)

    if os.path.exists(out_path):
        os.remove(out_path)

    if os.path.exists(err_path):
        os.remove(err_path)

    universe = load_universe(cfg)

    total_rows = 0

    for date in sorted(universe.keys()):
        print("loading date:", date, flush=True)

        data_loader.init_clickhouse_client(date)
        client = get_clickhouse_client(date)

        sec_list = universe[date]
        print("num securities:", len(sec_list), flush=True)

        for start in range(0, len(sec_list), chunk_size):
            chunk = sec_list[start:start + chunk_size]

            try:
                print(
                    date,
                    start + 1,
                    "/",
                    len(sec_list),
                    "chunk_size:",
                    len(chunk),
                    flush=True,
                )

                sql = build_market_query(cfg, date, chunk)
                df = query_df(client, sql)

                if df is None or len(df) == 0:
                    print("  empty market query result", flush=True)
                    continue

                df = clean_market_df(df, date, price_scale=price_scale)

                df = add_limit_prices(
                    market_df=df,
                    client=client,
                    cfg=cfg,
                    date=date,
                    securityids=chunk,
                )

                df = add_forward_returns(df, horizons)

                if drop_nan_returns:
                    ret_cols = [f"ret_{h}" for h in horizons]
                    df = df.dropna(subset=ret_cols)

                keep_cols = [
                    "date",
                    "securityid",
                    "time",
                    "datetime",
                ]

                for level in range(1, 11):
                    keep_cols.extend([
                        f"bid{level}",
                        f"ask{level}",
                        f"bid{level}_volume",
                        f"ask{level}_volume",
                    ])

                keep_cols.extend([
                    "mid_price",
                    "spread",
                    "limit_up_price",
                    "limit_down_price",
                ])

                keep_cols.extend([f"ret_{h}" for h in horizons])
                keep_cols.extend([f"label_{h}" for h in horizons])

                df = df[keep_cols]

                append_csv(df, out_path)

                total_rows += len(df)

                print("  appended rows:", len(df), "total:", total_rows, flush=True)

            except Exception as e:
                msg = {
                    "date": date,
                    "chunk_start": start,
                    "chunk": chunk,
                    "error": repr(e),
                }
                print("chunk error:", msg, flush=True)
                append_error(err_path, msg)

    print("saved:", out_path, flush=True)
    print("error log:", err_path, flush=True)
    print("total rows:", total_rows, flush=True)


if __name__ == "__main__":
    main()