import os
import yaml
import pandas as pd
import clickhouse_connect

YEAR_MONTH = "202410"
N_STOCKS = 100

OUT_DIR = "/home/fwz/projects/HFT_010-dev_luzw/PricingModel/data"
OUT_PATH = f"{OUT_DIR}/market_{YEAR_MONTH}_{N_STOCKS}.csv"

os.makedirs(OUT_DIR, exist_ok=True)

with open("default.yaml", "r") as f:
    cfg = yaml.safe_load(f)

ch = cfg["clickhouse"]

client = clickhouse_connect.get_client(
    host=ch["host"],
    port=ch["port"],
    username=ch["username"],
    password=ch["passward"],
)

tables = client.query("SHOW TABLES FROM `500ms`").result_rows
dates = sorted([x[0] for x in tables if x[0].startswith(YEAR_MONTH)])

print("dates:", dates)

if not dates:
    raise RuntimeError(f"No tables found for {YEAR_MONTH} in database 500ms")

first_date = dates[0]

stock_sql = f"""
SELECT DISTINCT SecurityID
FROM `500ms`.`{first_date}`
ORDER BY SecurityID
LIMIT {N_STOCKS}
"""

stock_ids = [x[0] for x in client.query(stock_sql).result_rows]
stock_ids_str = ",".join(str(x) for x in stock_ids)

print("num stocks:", len(stock_ids))
print("first stocks:", stock_ids[:10])

all_dfs = []

for_date_count = 0
for date in dates:
    print("exporting", date)

    sql = f"""
    SELECT
        {date} AS date,
        SecurityID AS code,
        timestamp,
        bidprice1 AS bid1,
        askprice1 AS ask1
    FROM `500ms`.`{date}`
    WHERE SecurityID IN ({stock_ids_str})
    ORDER BY SecurityID, timestamp
    """

    df = client.query_df(sql)

    if df.empty:
        print("empty:", date)
        continue

    df["code"] = df["code"].astype(int).astype(str).str.zfill(6)

    all_dfs.append(df)
    for_date_count += 1
    print(date, df.shape)

if not all_dfs:
    raise RuntimeError("No data exported")

out = pd.concat(all_dfs, ignore_index=True)

out = out[["date", "code", "timestamp", "bid1", "ask1"]]
out.to_csv(OUT_PATH, index=False)

print("saved:", OUT_PATH)
print("shape:", out.shape)
print(out.head())
print(out.tail())
