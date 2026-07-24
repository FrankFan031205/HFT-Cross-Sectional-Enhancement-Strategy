import yaml
import clickhouse_connect
import os

date = "20241022"
out_dir = "/home/fwz/HFT_010/sample_data"
os.makedirs(out_dir, exist_ok=True)

with open("default.yaml") as f:
    cfg = yaml.safe_load(f)

ch = cfg["clickhouse"]
client = clickhouse_connect.get_client(
    host=ch["host"],
    port=ch["port"],
    username=ch["username"],
    password=ch["passward"],
)

tables = {
    "snapshot": "500ms",
    "trade": "A_share_Trade",
    "order": "A_share_Order",
    "cancel": "A_share_Cancel",
}

for name, db in tables.items():
    print(f"Exporting {name} from {db}.`{date}`...")
    sql = f"""
    SELECT *
    FROM {db}.`{date}`
    LIMIT 100
    """
    df = client.query_df(sql)
    out_path = f"{out_dir}/{name}_{date}_sample.csv"
    df.to_csv(out_path, index=False)
    print(out_path, df.shape)

print("Done.")
