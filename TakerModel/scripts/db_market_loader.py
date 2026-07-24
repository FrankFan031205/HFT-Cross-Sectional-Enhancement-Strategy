import pandas as pd
import numpy as np


def make_clickhouse_client():
    import clickhouse_connect

    client = clickhouse_connect.get_client(
        host="127.0.0.1",
        port=18123,
        username="default",
        password="",
    )
    return client


def split_list(x, n):
    x = list(x)
    for i in range(0, len(x), n):
        yield x[i:i+n]


def load_market_first_snapshot_from_db(
    opt,
    label_horizon=60,
    db_name="500ms",
    symbol_chunk_size=100,
):
    client = make_clickhouse_client()

    opt = opt.copy()
    opt["execution_date"] = opt["execution_date"].astype(str)
    opt["securityid"] = pd.to_numeric(opt["securityid"], errors="coerce").astype("Int64")
    opt["execution_minute"] = pd.to_datetime(opt["execution_minute"])

    all_parts = []

    for date, opt_day in opt.groupby("execution_date"):
        symbols = sorted(opt_day["securityid"].dropna().astype(int).unique().tolist())
        needed_minutes = set(pd.to_datetime(opt_day["execution_minute"].dropna().unique()))

        print("loading date:", date, "symbols:", len(symbols), "needed minutes:", len(needed_minutes))

        day_parts = []

        for chunk_symbols in split_list(symbols, symbol_chunk_size):
            symbol_list = ",".join(str(x) for x in chunk_symbols)

            sql = f"""
            SELECT
                date,
                securityid,
                time,
                datetime,
                bid1,
                ask1,
                mid_price
            FROM `{db_name}`.`{date}`
            WHERE securityid IN ({symbol_list})
            ORDER BY securityid, datetime
            """

            df = client.query_df(sql)

            if df.empty:
                continue

            df.columns = [str(c).strip().lower() for c in df.columns]

            df["datetime"] = pd.to_datetime(df["datetime"])
            df["securityid"] = pd.to_numeric(df["securityid"], errors="coerce").astype("Int64")

            for c in ["bid1", "ask1", "mid_price"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")

            df = df.sort_values(["securityid", "datetime"]).reset_index(drop=True)

            df["future_mid"] = df.groupby("securityid")["mid_price"].shift(-label_horizon)
            df[f"label_{label_horizon}"] = df["future_mid"] / df["mid_price"] - 1.0

            df["execution_minute"] = df["datetime"].dt.floor("min")

            df = df[df["execution_minute"].isin(needed_minutes)].copy()

            if df.empty:
                continue

            df = (
                df.sort_values(["securityid", "execution_minute", "datetime"])
                  .drop_duplicates(["securityid", "execution_minute"], keep="first")
            )

            df = df.rename(
                columns={
                    "datetime": "market_datetime",
                    "bid1": "bid_price",
                    "ask1": "ask_price",
                    "mid_price": "mid_price",
                    f"label_{label_horizon}": "label",
                }
            )

            keep = [
                "securityid",
                "execution_minute",
                "market_datetime",
                "bid_price",
                "ask_price",
                "mid_price",
                "label",
            ]

            day_parts.append(df[keep].copy())

        if day_parts:
            day_df = pd.concat(day_parts, ignore_index=True)
            all_parts.append(day_df)

    if not all_parts:
        return pd.DataFrame(
            columns=[
                "securityid",
                "execution_minute",
                "market_datetime",
                "bid_price",
                "ask_price",
                "mid_price",
                "label",
            ]
        )

    out = pd.concat(all_parts, ignore_index=True)
    out = out.sort_values(["securityid", "execution_minute", "market_datetime"])
    out = out.drop_duplicates(["securityid", "execution_minute"], keep="first")
    out = out.reset_index(drop=True)

    return out
