import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import clickhouse_connect


DB_SNAPSHOT = "500ms"
DB_LIMIT = "A_share_Limit"


def load_clickhouse_client(config_path):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    ch = cfg["clickhouse"]

    return clickhouse_connect.get_client(
        host=ch["host"],
        port=ch["port"],
        username=ch["username"],
        password=ch["passward"],
    )


def get_dates(client, start_date, end_date):
    tables = client.query(f"SHOW TABLES FROM `{DB_SNAPSHOT}`").result_rows

    dates = []
    for row in tables:
        name = str(row[0])
        if name.isdigit() and start_date <= name <= end_date:
            dates.append(name)

    return sorted(dates)


def get_table_columns(client, db, table):
    q = f"DESCRIBE TABLE `{db}`.`{table}`"
    desc = client.query_df(q)

    if "name" in desc.columns:
        return set(desc["name"].astype(str).tolist())

    return set(desc.iloc[:, 0].astype(str).tolist())


def load_universe(path):
    df = pd.read_csv(path, usecols=["securityid"])
    ids = (
        df["securityid"]
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(6)
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    return ids


def chunk_list(xs, n):
    for i in range(0, len(xs), n):
        yield xs[i:i + n]


def make_security_filter(stock_ids):
    padded = [str(x).zfill(6) for x in stock_ids]
    unpadded = [str(int(x)) for x in padded]

    padded_sql = ",".join([f"'{x}'" for x in padded])
    unpadded_sql = ",".join([f"'{x}'" for x in unpadded])

    return (
        f"(toString(SecurityID) IN ({padded_sql}) "
        f"OR toString(SecurityID) IN ({unpadded_sql}))"
    )


def build_snapshot_select_cols(client, date):
    cols = get_table_columns(client, DB_SNAPSHOT, date)

    if "SecurityID" not in cols:
        raise KeyError(f"SecurityID not found in {DB_SNAPSHOT}.{date}")

    if "datetime" not in cols:
        raise KeyError(f"datetime not found in {DB_SNAPSHOT}.{date}")

    select_cols = [
        f"{date} AS date",
        "SecurityID AS securityid",
        "datetime AS datetime",
    ]

    for i in range(1, 11):
        bid_price_col = f"bidprice{i}"
        ask_price_col = f"askprice{i}"
        bid_volume_col = f"bidvolume{i}"
        ask_volume_col = f"askvolume{i}"

        required = [
            bid_price_col,
            ask_price_col,
            bid_volume_col,
            ask_volume_col,
        ]

        for c in required:
            if c not in cols:
                raise KeyError(f"{c} not found in {DB_SNAPSHOT}.{date}")

        select_cols.append(f"{bid_price_col} AS bid{i}")
        select_cols.append(f"{ask_price_col} AS ask{i}")
        select_cols.append(f"{bid_volume_col} AS bid{i}_volume")
        select_cols.append(f"{ask_volume_col} AS ask{i}_volume")

    return ",\n        ".join(select_cols)


def make_time_key_from_datetime(x):
    s = x.astype(str).str.strip()
    out = s.copy()

    mask_underscore = s.str.contains("_", regex=False)
    out.loc[mask_underscore] = s.loc[mask_underscore].str.split("_").str[-1]

    mask_space = s.str.contains(" ", regex=False)
    out.loc[mask_space] = s.loc[mask_space].str.split(" ").str[-1]

    out = (
        out.astype(str)
        .str.replace(":", "", regex=False)
        .str.replace(".", "", regex=False)
        .str.replace(r"\D", "", regex=True)
    )

    def fix_token(v):
        v = str(v)
        if len(v) == 6:
            return v + "000"
        return v[-9:].zfill(9)

    return out.map(fix_token)


def normalize_prices(df):
    price_cols = []

    for i in range(1, 11):
        price_cols.append(f"bid{i}")
        price_cols.append(f"ask{i}")

    price_cols.extend(["limit_up_price", "limit_down_price"])

    existing = [c for c in price_cols if c in df.columns]
    sample_cols = [c for c in ["bid1", "ask1"] if c in df.columns]

    if not sample_cols:
        return df

    med = df[sample_cols].replace(0, np.nan).stack().median()

    if pd.notna(med) and med > 100:
        for c in existing:
            df[c] = df[c] / 100.0

    return df


def query_snapshot_chunk(client, date, stock_ids):
    select_cols = build_snapshot_select_cols(client, date)
    security_filter = make_security_filter(stock_ids)

    q = f"""
    SELECT
        {select_cols}
    FROM `{DB_SNAPSHOT}`.`{date}`
    WHERE {security_filter}
    ORDER BY SecurityID, datetime
    """

    df = client.query_df(q)

    if df.empty:
        return df

    df["date"] = df["date"].astype(str)
    df["securityid"] = (
        df["securityid"]
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(6)
    )
    df["datetime"] = df["datetime"].astype(str)
    df["time"] = make_time_key_from_datetime(df["datetime"])

    df = normalize_prices(df)

    return df


def query_limit_one_date(client, date, stock_ids):
    try:
        cols = get_table_columns(client, DB_LIMIT, date)
    except Exception as e:
        print(f"warning: cannot describe {DB_LIMIT}.{date}: {e}")
        return pd.DataFrame(columns=["securityid", "limit_up_price", "limit_down_price"])

    if "SecurityID" not in cols:
        print(f"warning: SecurityID not found in {DB_LIMIT}.{date}")
        return pd.DataFrame(columns=["securityid", "limit_up_price", "limit_down_price"])

    if "limitUpPrice" not in cols or "limitDownPrice" not in cols:
        print(f"warning: limit columns not found in {DB_LIMIT}.{date}")
        return pd.DataFrame(columns=["securityid", "limit_up_price", "limit_down_price"])

    security_filter = make_security_filter(stock_ids)

    q = f"""
    SELECT
        SecurityID AS securityid,
        limitUpPrice AS limit_up_price,
        limitDownPrice AS limit_down_price
    FROM `{DB_LIMIT}`.`{date}`
    WHERE {security_filter}
    """

    try:
        df = client.query_df(q)
    except Exception as e:
        print(f"warning: cannot query {DB_LIMIT}.{date}: {e}")
        return pd.DataFrame(columns=["securityid", "limit_up_price", "limit_down_price"])

    if df.empty:
        return pd.DataFrame(columns=["securityid", "limit_up_price", "limit_down_price"])

    df["securityid"] = (
        df["securityid"]
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(6)
    )

    df = df.drop_duplicates(["securityid"], keep="last")

    return df


def load_daily_market_features(date, stock_ids, daily_dir):
    daily_path = Path(daily_dir) / f"daily_market_{date[:4]}-{date[4:6]}-{date[6:8]}.csv"

    if not daily_path.exists():
        print(f"warning: missing daily market file: {daily_path}")
        return pd.DataFrame(columns=["date", "securityid", "marketValue", "turnoverRate"])

    daily = pd.read_csv(daily_path)

    need_cols = ["ticker", "tradeDate", "marketValue", "turnoverRate"]

    for c in need_cols:
        if c not in daily.columns:
            raise KeyError(f"missing {c} in {daily_path}")

    tmp = daily[need_cols].copy()

    tmp["securityid"] = (
        tmp["ticker"]
        .astype(str)
        .str.replace(r"\..*$", "", regex=True)
        .str.zfill(6)
    )

    tmp["date"] = (
        tmp["tradeDate"]
        .astype(str)
        .str.replace("-", "", regex=False)
    )

    stock_set = set([str(x).zfill(6) for x in stock_ids])

    tmp = tmp[tmp["date"] == date]
    tmp = tmp[tmp["securityid"].isin(stock_set)]

    tmp = tmp[["date", "securityid", "marketValue", "turnoverRate"]]
    tmp = tmp.drop_duplicates(["date", "securityid"], keep="last")

    return tmp


def add_market_features(df, limit_df, daily_df):
    if not limit_df.empty:
        df = df.merge(limit_df, on="securityid", how="left")
    else:
        df["limit_up_price"] = np.nan
        df["limit_down_price"] = np.nan

    if not daily_df.empty:
        df = df.merge(daily_df, on=["date", "securityid"], how="left")
    else:
        df["marketValue"] = np.nan
        df["turnoverRate"] = np.nan

    df = normalize_prices(df)

    df["mid_price"] = (df["bid1"] + df["ask1"]) / 2.0
    df["spread"] = df["ask1"] - df["bid1"]

    return df


def add_returns_and_volatility(df, horizons):
    df = df.sort_values(["securityid", "time"]).copy()

    g = df.groupby("securityid", group_keys=False)

    mid_ret_1 = (
        g["mid_price"]
        .pct_change()
        .replace([np.inf, -np.inf], np.nan)
    )

    df["volatility_60"] = (
        mid_ret_1.groupby(df["securityid"])
        .rolling(60, min_periods=10)
        .std()
        .reset_index(level=0, drop=True)
        .fillna(0.0)
    )

    for h in horizons:
        future_mid = g["mid_price"].shift(-h)
        ret = future_mid / df["mid_price"] - 1.0

        df[f"ret_{h}"] = ret
        df[f"label_{h}"] = ret

    return df


def reorder_columns(df, horizons):
    cols = [
        "date",
        "securityid",
        "time",
        "datetime",
    ]

    for i in range(1, 11):
        cols.extend([
            f"bid{i}",
            f"ask{i}",
            f"bid{i}_volume",
            f"ask{i}_volume",
        ])

    cols.extend([
        "mid_price",
        "spread",
        "limit_up_price",
        "limit_down_price",
        "marketValue",
        "turnoverRate",
        "volatility_60",
    ])

    for h in horizons:
        cols.append(f"ret_{h}")

    for h in horizons:
        cols.append(f"label_{h}")

    cols = [c for c in cols if c in df.columns]

    return df[cols]


def export_one_date(client, date, stock_ids, output_dir, daily_dir, chunk_stocks, horizons, overwrite):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    out_path = output_dir / f"market_return_{date}_{len(stock_ids)}.csv"

    if out_path.exists() and not overwrite:
        print(f"skip existing: {out_path}")
        return

    tmp_path = output_dir / f".tmp_market_return_{date}_{len(stock_ids)}.csv"

    if tmp_path.exists():
        tmp_path.unlink()

    print(f"date {date}: output {out_path}")

    limit_df = query_limit_one_date(client, date, stock_ids)
    daily_df = load_daily_market_features(date, stock_ids, daily_dir)

    print(f"date {date}: limit rows {len(limit_df)}, daily rows {len(daily_df)}")

    wrote_header = False
    total_rows = 0

    for chunk_id, chunk_ids in enumerate(chunk_list(stock_ids, chunk_stocks)):
        print(f"date {date}: chunk {chunk_id}, stocks {len(chunk_ids)}")

        df = query_snapshot_chunk(client, date, chunk_ids)

        if df.empty:
            print(f"date {date}: chunk {chunk_id} empty")
            continue

        df = add_market_features(df, limit_df, daily_df)
        df = add_returns_and_volatility(df, horizons)
        df = reorder_columns(df, horizons)

        df.to_csv(
            tmp_path,
            mode="a",
            index=False,
            header=not wrote_header,
        )

        wrote_header = True
        total_rows += len(df)

        print(f"date {date}: chunk {chunk_id} rows {len(df)}, total {total_rows}")

        del df
        gc.collect()

    if total_rows == 0:
        print(f"warning: date {date} exported zero rows")
        if tmp_path.exists():
            tmp_path.unlink()
        return

    tmp_path.rename(out_path)

    print(f"date {date}: saved {out_path}, rows {total_rows}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", required=True)
    parser.add_argument("--start_date", required=True)
    parser.add_argument("--end_date", required=True)
    parser.add_argument("--universe_from", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--daily_dir", default="/home/fwz/cross_common_data/daily_market")
    parser.add_argument("--chunk_stocks", type=int, default=25)
    parser.add_argument("--horizons", default="30,60,90,120")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max_dates", type=int, default=0)

    args = parser.parse_args()

    horizons = [int(x) for x in args.horizons.split(",")]

    client = load_clickhouse_client(args.config)
    dates = get_dates(client, args.start_date, args.end_date)

    if args.max_dates and args.max_dates > 0:
        dates = dates[:args.max_dates]

    if len(dates) == 0:
        raise RuntimeError(f"no dates found from {args.start_date} to {args.end_date}")

    stock_ids = load_universe(args.universe_from)

    print("num dates:", len(dates))
    print("first date:", dates[0])
    print("last date:", dates[-1])
    print("num stocks:", len(stock_ids))
    print("first stocks:", stock_ids[:20])
    print("output_dir:", args.output_dir)
    print("chunk_stocks:", args.chunk_stocks)
    print("horizons:", horizons)

    for date in dates:
        export_one_date(
            client=client,
            date=date,
            stock_ids=stock_ids,
            output_dir=args.output_dir,
            daily_dir=args.daily_dir,
            chunk_stocks=args.chunk_stocks,
            horizons=horizons,
            overwrite=args.overwrite,
        )

    print("ALL DONE")


if __name__ == "__main__":
    main()
