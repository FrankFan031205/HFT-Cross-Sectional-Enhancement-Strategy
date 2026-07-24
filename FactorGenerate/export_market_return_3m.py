import os
import yaml
import numpy as np
import pandas as pd
import clickhouse_connect


START_DATE = "20241022"
END_DATE = "20250114"
N_STOCKS = 100
PRICE_SCALE = 100.0
HORIZONS = [30, 60, 90, 120]
QUERY_CHUNK_SIZE = 20

CONFIG_PATH = "/home/fwz/projects/HFT_010-dev_fwz/FactorGenerate/default.yaml"

OUT_DIR = "/home/fwz/projects/HFT_010-dev_fwz/PricingModel/data"
OUT_PATH = f"{OUT_DIR}/market_return_{START_DATE}_{END_DATE}_{N_STOCKS}.csv"
ERR_PATH = f"{OUT_DIR}/market_return_{START_DATE}_{END_DATE}_{N_STOCKS}_errors.txt"

EXISTING_FILE = "/home/fwz/projects/HFT_010-dev_fwz/PricingModel/data/market_return_20241022_20241122_100.csv"


def normalize_securityid(x):
    if pd.isna(x):
        return None

    s = str(x).strip()

    if s.endswith(".0"):
        s = s[:-2]

    if s == "":
        return None

    return s.zfill(6)


def ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def append_error(msg):
    ensure_dir(ERR_PATH)
    with open(ERR_PATH, "a") as f:
        f.write(str(msg) + "\n")


def load_clickhouse_client():
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)

    ch = cfg["clickhouse"]

    client = clickhouse_connect.get_client(
        host=ch["host"],
        port=ch["port"],
        username=ch["username"],
        password=ch.get("passward", ch.get("password")),
    )

    return client


def load_stock_pool(client):
    if os.path.exists(EXISTING_FILE):
        stock_ids = []
        seen = set()

        print("loading stock pool from existing file:", EXISTING_FILE, flush=True)

        for chunk in pd.read_csv(
            EXISTING_FILE,
            usecols=["securityid"],
            dtype={"securityid": str},
            chunksize=1000000,
        ):
            vals = (
                chunk["securityid"]
                .map(normalize_securityid)
                .dropna()
                .drop_duplicates()
                .tolist()
            )

            for sid in vals:
                if sid not in seen:
                    seen.add(sid)
                    stock_ids.append(sid)

                if len(stock_ids) >= N_STOCKS:
                    break

            print("collected stocks:", len(stock_ids), flush=True)

            if len(stock_ids) >= N_STOCKS:
                break

        stock_ids = stock_ids[:N_STOCKS]

        print(
            "use stock pool from existing file:",
            len(stock_ids),
            stock_ids[:10],
            flush=True,
        )

        return stock_ids

    stock_sql = f"""
    SELECT DISTINCT SecurityID
    FROM `500ms`.`{START_DATE}`
    ORDER BY SecurityID
    LIMIT {N_STOCKS}
    """

    rows = client.query(stock_sql).result_rows
    stock_ids = [normalize_securityid(x[0]) for x in rows]
    stock_ids = [x for x in stock_ids if x is not None]

    print(
        "use stock pool from ClickHouse:",
        len(stock_ids),
        stock_ids[:10],
        flush=True,
    )

    return stock_ids


def get_available_dates(client):
    snapshot_tables = set(
        x[0] for x in client.query("SHOW TABLES FROM `500ms`").result_rows
    )

    limit_tables = set(
        x[0] for x in client.query("SHOW TABLES FROM `A_share_Limit`").result_rows
    )

    dates = sorted(
        d for d in snapshot_tables
        if START_DATE <= d <= END_DATE and d in limit_tables
    )

    print("num dates:", len(dates), flush=True)
    print("first dates:", dates[:5], flush=True)
    print("last dates:", dates[-5:], flush=True)

    return dates


def build_market_sql(date, stock_ids):
    ids = ",".join("'" + x + "'" for x in stock_ids)

    select_cols = [
        f"'{date}' AS date",
        "s.SecurityID AS securityid",
        "s.datetime AS raw_time",
    ]

    for level in range(1, 11):
        select_cols.extend([
            f"toFloat64(s.bidprice{level}) / {PRICE_SCALE} AS bid{level}",
            f"toFloat64(s.askprice{level}) / {PRICE_SCALE} AS ask{level}",
            f"toFloat64(s.bidvolume{level}) AS bid{level}_volume",
            f"toFloat64(s.askvolume{level}) AS ask{level}_volume",
        ])

    select_cols.extend([
        f"toFloat64(l.limitUpPrice) / {PRICE_SCALE} AS limit_up_price",
        f"toFloat64(l.limitDownPrice) / {PRICE_SCALE} AS limit_down_price",
    ])

    select_sql = ",\n            ".join(select_cols)

    sql = f"""
    SELECT
            {select_sql}
    FROM `500ms`.`{date}` AS s
    LEFT JOIN `A_share_Limit`.`{date}` AS l
        ON toInt32(s.SecurityID) = l.SecurityID
    WHERE s.SecurityID IN ({ids})
    ORDER BY s.SecurityID, s.datetime
    """

    return sql


def clean_market_df(df, date):
    if df is None or len(df) == 0:
        return pd.DataFrame()

    out = df.copy()

    out["date"] = int(date)
    out["securityid"] = out["securityid"].map(normalize_securityid)

    raw_time = out["raw_time"].astype(str).str.strip()
    digits = raw_time.str.replace(r"\D", "", regex=True)
    out["time"] = digits.str[-9:].str.zfill(9).astype(int)

    out["datetime"] = (
        out["date"].astype(str)
        + "_"
        + out["time"].astype(str).str.zfill(9)
    )

    price_cols = []
    volume_cols = []

    for level in range(1, 11):
        price_cols.extend([f"bid{level}", f"ask{level}"])
        volume_cols.extend([f"bid{level}_volume", f"ask{level}_volume"])

    for col in price_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    for col in volume_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)

    out["limit_up_price"] = pd.to_numeric(
        out["limit_up_price"],
        errors="coerce",
    )

    out["limit_down_price"] = pd.to_numeric(
        out["limit_down_price"],
        errors="coerce",
    )

    out = out.replace([np.inf, -np.inf], np.nan)

    out = out.dropna(
        subset=[
            "date",
            "securityid",
            "time",
            "bid1",
            "ask1",
        ]
    )

    out = out[(out["bid1"] > 0) & (out["ask1"] > 0)]
    out = out[out["ask1"] >= out["bid1"]]

    out["mid_price"] = (out["bid1"] + out["ask1"]) / 2.0
    out["spread"] = out["ask1"] - out["bid1"]

    out = out.sort_values(
        ["date", "securityid", "time"]
    ).reset_index(drop=True)

    return out


def add_forward_returns(df):
    if df is None or len(df) == 0:
        return df

    out = df.sort_values(
        ["date", "securityid", "time"]
    ).copy()

    for h in HORIZONS:
        future_mid = (
            out.groupby(["date", "securityid"])["mid_price"]
            .shift(-h)
        )

        ret = (future_mid - out["mid_price"]) / out["mid_price"]

        out[f"ret_{h}"] = ret
        out[f"label_{h}"] = ret

    ret_cols = [f"ret_{h}" for h in HORIZONS]
    out = out.dropna(subset=ret_cols)

    return out


def select_output_columns(df):
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

    keep_cols.extend([f"ret_{h}" for h in HORIZONS])
    keep_cols.extend([f"label_{h}" for h in HORIZONS])

    return df[keep_cols]


def append_csv(df, path, write_header):
    ensure_dir(path)
    df.to_csv(path, mode="a", header=write_header, index=False)


def main():
    ensure_dir(OUT_PATH)
    ensure_dir(ERR_PATH)

    if os.path.exists(OUT_PATH):
        os.remove(OUT_PATH)

    if os.path.exists(ERR_PATH):
        os.remove(ERR_PATH)

    client = load_clickhouse_client()

    stock_ids = load_stock_pool(client)

    if len(stock_ids) == 0:
        raise RuntimeError("empty stock pool")

    dates = get_available_dates(client)

    if len(dates) == 0:
        raise RuntimeError("no matched dates")

    print("output:", OUT_PATH, flush=True)

    total_rows = 0
    first_write = True

    for date in dates:
        print("\nloading date:", date, flush=True)

        for start in range(0, len(stock_ids), QUERY_CHUNK_SIZE):
            chunk = stock_ids[start:start + QUERY_CHUNK_SIZE]

            print(
                "chunk:",
                start + 1,
                "/",
                len(stock_ids),
                "chunk_size:",
                len(chunk),
                flush=True,
            )

            try:
                sql = build_market_sql(date, chunk)
                df = client.query_df(sql)

                if df is None or len(df) == 0:
                    print("  empty query result", flush=True)
                    continue

                df = clean_market_df(df, date)

                if df is None or len(df) == 0:
                    print("  empty after clean", flush=True)
                    continue

                df = add_forward_returns(df)

                if df is None or len(df) == 0:
                    print("  empty after returns", flush=True)
                    continue

                df = select_output_columns(df)

                append_csv(df, OUT_PATH, first_write)
                first_write = False

                total_rows += len(df)

                print(
                    "  appended rows:",
                    len(df),
                    "total:",
                    total_rows,
                    flush=True,
                )

            except Exception as e:
                msg = {
                    "date": date,
                    "chunk_start": start,
                    "chunk": chunk,
                    "error": repr(e),
                }

                print("chunk error:", msg, flush=True)
                append_error(msg)

    print("\nsaved:", OUT_PATH, flush=True)
    print("error log:", ERR_PATH, flush=True)
    print("total rows:", total_rows, flush=True)


if __name__ == "__main__":
    main()