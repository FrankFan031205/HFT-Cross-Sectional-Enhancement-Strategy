import os
import yaml
import pandas as pd
import clickhouse_connect

START_DATE = "20241123"
END_DATE = "20251022"
N_STOCKS = 100

CONFIG_PATH = "/home/fwz/projects/HFT_010-dev_fwz/FactorGenerate/default.yaml"
EXISTING_FILE = "/home/fwz/projects/HFT_010-dev_fwz/PricingModel/data/market_return_20241022_20241122_100.csv"
OUT_PATH = "/home/fwz/projects/HFT_010-dev_fwz/PricingModel/data/universe_20241123_20251022_100.csv"

with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

ch = cfg["clickhouse"]
client = clickhouse_connect.get_client(
    host=ch["host"],
    port=ch["port"],
    username=ch["username"],
    password=ch.get("passward", ch.get("password")),
)

old = pd.read_csv(EXISTING_FILE, usecols=["securityid"])
stock_ids = (
    old["securityid"]
    .astype(str)
    .str.replace(r"\.0$", "", regex=True)
    .str.zfill(6)
    .drop_duplicates()
    .head(N_STOCKS)
    .tolist()
)

snapshot_tables = set(x[0] for x in client.query("SHOW TABLES FROM `500ms`").result_rows)
limit_tables = set(x[0] for x in client.query("SHOW TABLES FROM `A_share_Limit`").result_rows)

dates = sorted(
    d for d in snapshot_tables
    if START_DATE <= d <= END_DATE and d in limit_tables
)

rows = []
for d in dates:
    for sid in stock_ids:
        rows.append((int(d), sid))

out = pd.DataFrame(rows, columns=["date", "securityid"])
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
out.to_csv(OUT_PATH, index=False)

print("saved:", OUT_PATH)
print("num dates:", len(dates))
print("num stocks:", len(stock_ids))
print("shape:", out.shape)
print(out.head())
print(out.tail())
