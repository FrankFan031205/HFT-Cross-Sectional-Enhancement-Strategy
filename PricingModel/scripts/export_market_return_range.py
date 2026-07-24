import argparse
from pathlib import Path

import yaml
import numpy as np
import pandas as pd
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


def get_universe_from_file(path, n_stocks):
    df = pd.read_csv(path, usecols=["securityid"])

    ids = (
        df["securityid"]
        .astype(str)
        .str.zfill(6)
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    if n_stocks is not None:
        ids = ids[:n_stocks]

    return ids


def get_universe_from_db(client, first_date, n_stocks):
    q = f"""
    SELECT DISTINCT SecurityID
    FROM `{DB_SNAPSHOT}`.`{first_date}`
    ORDER BY SecurityID
    LIMIT {n_stocks}
    """

    ids = [str(x[0]).zfill(6) for x in client.query(q).result_rows]
    return ids


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


def make_security_filter(stock_ids):
    padded = [str(x).zfill(6) for x in stock_ids]
    unpadded = [str(int(x)) for x in padded]

    padded_sql = ",".join([f"'{x}'" for x in padded])
    unpadded_sql = ",".join([f"'{x}'" for x in unpadded])

    return f"(toString(SecurityID) IN ({padded_sql}) OR toString(SecurityID) IN ({unpadded_sql}))"

def query_snapshot_one_date(client, date, stock_ids):
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
    df["securityid"] = df["securityid"].astype(str).str.zfill(6)
    df["datetime"] = df["datetime"].astype(str)

    if df["datetime"].str.contains("_").any():
        df["time"] = df["datetime"].str.split("_").str[-1].str.zfill(9)
    else:
        df["time"] = df["datetime"].str[-9:].str.zfill(9)

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

    df["securityid"] = df["securityid"].astype(str).str.zfill(6)
    df = df.drop_duplicates(["securityid"], keep="last")

    return df


def export_one_date(client, date, stock_ids):
    snapshot = query_snapshot_one_date(client, date, stock_ids)

    if snapshot.empty:
        return snapshot

    limit_df = query_limit_one_date(client, date, stock_ids)

    out = snapshot.merge(
        limit_df,
        on="securityid",
        how="left",
    )

    out = normalize_prices(out)

    out["mid_price"] = (out["bid1"] + out["ask1"]) / 2
    out["spread"] = out["ask1"] - out["bid1"]

    return out


def add_forward_returns(df, horizons):
    df = df.sort_values(["date", "securityid", "time"]).copy()

    g = df.groupby(["date", "securityid"], group_keys=False)

    for h in horizons:
        future_mid = g["mid_price"].shift(-h)
        ret = future_mid / df["mid_price"] - 1

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
    ])

    for h in horizons:
        cols.append(f"ret_{h}")

    for h in horizons:
        cols.append(f"label_{h}")

    cols = [c for c in cols if c in df.columns]

    return df[cols]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", required=True)
    parser.add_argument("--start_date", required=True)
    parser.add_argument("--end_date", required=True)
    parser.add_argument("--n_stocks", type=int, default=100)
    parser.add_argument("--universe_from", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--horizons", default="30,60,90,120")

    args = parser.parse_args()

    horizons = [int(x) for x in args.horizons.split(",")]

    client = load_clickhouse_client(args.config)

    dates = get_dates(client, args.start_date, args.end_date)

    print("dates:", dates)

    if len(dates) == 0:
        raise RuntimeError(
            f"No snapshot tables found from {args.start_date} to {args.end_date}"
        )

    if args.universe_from and Path(args.universe_from).exists():
        stock_ids = get_universe_from_file(args.universe_from, args.n_stocks)
        print("universe from file:", args.universe_from)
    else:
        stock_ids = get_universe_from_db(client, dates[0], args.n_stocks)
        print("universe from db:", dates[0])

    print("num stocks:", len(stock_ids))
    print("first stocks:", stock_ids[:10])

    all_parts = []

    for date in dates:
        print("exporting", date)

        part = export_one_date(client, date, stock_ids)

        if part.empty:
            print("empty:", date)
            continue

        print(date, part.shape)
        all_parts.append(part)

    if len(all_parts) == 0:
        raise RuntimeError("No data exported.")

    out = pd.concat(all_parts, ignore_index=True)

    out = add_forward_returns(out, horizons)
    out = reorder_columns(out, horizons)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    out.to_csv(output_path, index=False)

    print("saved:", output_path)
    print("shape:", out.shape)
    print("date range:", out["date"].min(), out["date"].max())
    print("num dates:", out["date"].nunique())
    print("num stocks:", out["securityid"].nunique())
    print(out.head())
    print(out.tail())


if __name__ == "__main__":
    main()
