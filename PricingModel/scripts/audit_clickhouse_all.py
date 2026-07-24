import os
import re
import subprocess
import pandas as pd
from io import StringIO

OUT_DIR = "PricingModel/data/clickhouse_audit"
os.makedirs(OUT_DIR, exist_ok=True)

EXCLUDE_DBS = {"system", "INFORMATION_SCHEMA", "information_schema"}

# 只对这些日期做深度统计，避免全库 count 太慢
PROBE_DATES = [
    "20241022", "20241122", "20241220", "20250114", "20250228"
]

def ch(query, fmt="CSVWithNames", silent=False):
    q = query.strip()
    if fmt:
        q += f"\nFORMAT {fmt}"
    try:
        res = subprocess.run(
            ["clickhouse-client", "--query", q],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        return res.stdout
    except subprocess.CalledProcessError as e:
        if not silent:
            print("[CH ERROR]", e.stderr[:500])
            print("[QUERY]", q[:500])
        return ""

def read_csv_from_ch(query, silent=False):
    out = ch(query, fmt="CSVWithNames", silent=silent)
    if not out.strip():
        return pd.DataFrame()
    return pd.read_csv(StringIO(out), dtype=str)

def qname(db, tbl):
    return f"`{db}`.`{tbl}`"

def is_date_table(x):
    return bool(re.fullmatch(r"\d{8}", str(x)))

def classify_table(cols, db="", tbl=""):
    c = {str(x).lower() for x in cols}
    name = f"{db}.{tbl}".lower()

    has_lob = any(x in c for x in ["bidprice1", "askprice1", "bidvolume1", "askvolume1"])
    has_trade = any(x in c for x in ["bidseqno", "offerseqno"]) or ("trade" in name)
    has_order = ("order" in name) or ("seqno" in c and "type" in c and "side" in c and "price" in c and "qty" in c)
    has_cancel = ("cancel" in name) or any(x in c for x in ["cancel_qty", "order_seqno", "order_qty"])
    has_limit = ("limit" in name) or any(x in c for x in ["limit_up_price", "limit_down_price"])
    has_index_fut = ("index" in name) or any("ic" in str(x).lower() for x in cols)

    if has_cancel:
        return "cancel"
    if has_order:
        return "order"
    if has_trade:
        return "trade"
    if has_limit:
        return "limit"
    if has_lob and ("index" in name or "stock_index" in name):
        return "index_or_futures_snapshot"
    if has_lob:
        return "snapshot_lob"
    if has_index_fut:
        return "index_related"
    return "unknown"

def infer_freq(rows_per_symbol):
    try:
        x = float(rows_per_symbol)
    except Exception:
        return ""
    if x <= 0:
        return ""
    if 25000 <= x <= 32000:
        return "likely_500ms_snapshot"
    if 4000 <= x <= 6000:
        return "likely_3s_snapshot"
    if 200 <= x <= 400:
        return "likely_1min_or_bar"
    if 1000 <= x < 4000:
        return "subsampled_or_event_data"
    if x > 32000:
        return "event_or_multi_record_snapshot"
    return "low_freq_or_sparse_event"

print("[1] loading all tables...")
tables = read_csv_from_ch("""
SELECT
    database,
    name AS table_name,
    engine,
    total_rows,
    total_bytes,
    formatReadableSize(total_bytes) AS size
FROM system.tables
WHERE database NOT IN ('system', 'INFORMATION_SCHEMA', 'information_schema')
ORDER BY database, table_name
""")

tables.to_csv(f"{OUT_DIR}/all_tables.csv", index=False)
print("[SAVED]", f"{OUT_DIR}/all_tables.csv", "rows=", len(tables))

print("[2] loading all columns...")
columns = read_csv_from_ch("""
SELECT
    database,
    table AS table_name,
    name AS column_name,
    type
FROM system.columns
WHERE database NOT IN ('system', 'INFORMATION_SCHEMA', 'information_schema')
ORDER BY database, table_name, position
""")

columns.to_csv(f"{OUT_DIR}/all_columns.csv", index=False)
print("[SAVED]", f"{OUT_DIR}/all_columns.csv", "rows=", len(columns))

print("[3] building table column summary...")
col_summary = (
    columns.groupby(["database", "table_name"])["column_name"]
    .apply(list)
    .reset_index()
)

col_summary["n_cols"] = col_summary["column_name"].map(len)
col_summary["columns_joined"] = col_summary["column_name"].map(lambda xs: ",".join(map(str, xs)))
col_summary["table_type_guess"] = col_summary.apply(
    lambda r: classify_table(r["column_name"], r["database"], r["table_name"]),
    axis=1
)

table_summary = tables.merge(
    col_summary.drop(columns=["column_name"]),
    on=["database", "table_name"],
    how="left"
)

table_summary["is_date_table"] = table_summary["table_name"].map(is_date_table)
table_summary.to_csv(f"{OUT_DIR}/table_summary.csv", index=False)
print("[SAVED]", f"{OUT_DIR}/table_summary.csv")

print("[4] probing date tables for rows_per_symbol...")
stats_rows = []

probe_tables = table_summary[
    table_summary["is_date_table"].fillna(False)
].copy()

# 只查有 SecurityID + timestamp 的表
has_cols = columns.groupby(["database", "table_name"])["column_name"].apply(lambda x: set(map(str, x))).to_dict()

for _, r in probe_tables.iterrows():
    db = r["database"]
    tbl = r["table_name"]
    cols = has_cols.get((db, tbl), set())
    if "SecurityID" not in cols:
        continue

    # 只对 PROBE_DATES 或者每个库第一张日表做深度统计
    if tbl not in PROBE_DATES:
        continue

    print(f"[PROBE] {db}.{tbl}")

    timestamp_expr = "min(timestamp) AS min_ts, max(timestamp) AS max_ts" if "timestamp" in cols else "'' AS min_ts, '' AS max_ts"

    query = f"""
    SELECT
        '{db}' AS database,
        '{tbl}' AS table_name,
        count() AS rows,
        uniqExact(SecurityID) AS n_symbols,
        round(rows / n_symbols, 2) AS rows_per_symbol,
        {timestamp_expr}
    FROM {qname(db, tbl)}
    """

    df = read_csv_from_ch(query, silent=True)
    if df.empty:
        continue

    df["table_type_guess"] = r.get("table_type_guess", "")
    df["freq_guess"] = df["rows_per_symbol"].map(infer_freq)
    stats_rows.append(df)

if stats_rows:
    stats = pd.concat(stats_rows, ignore_index=True)
else:
    stats = pd.DataFrame()

stats.to_csv(f"{OUT_DIR}/date_table_probe_stats.csv", index=False)
print("[SAVED]", f"{OUT_DIR}/date_table_probe_stats.csv", "rows=", len(stats))

print("[5] security id samples...")
sample_rows = []

for _, r in probe_tables.iterrows():
    db = r["database"]
    tbl = r["table_name"]
    cols = has_cols.get((db, tbl), set())
    if "SecurityID" not in cols:
        continue
    if tbl != "20241022":
        continue

    print(f"[SAMPLE] {db}.{tbl}")
    query = f"""
    SELECT
        '{db}' AS database,
        '{tbl}' AS table_name,
        toString(SecurityID) AS securityid,
        count() AS n
    FROM {qname(db, tbl)}
    GROUP BY securityid
    ORDER BY securityid
    LIMIT 50
    """
    df = read_csv_from_ch(query, silent=True)
    if not df.empty:
        df["table_type_guess"] = r.get("table_type_guess", "")
        sample_rows.append(df)

if sample_rows:
    samples = pd.concat(sample_rows, ignore_index=True)
else:
    samples = pd.DataFrame()

samples.to_csv(f"{OUT_DIR}/securityid_samples_20241022.csv", index=False)
print("[SAVED]", f"{OUT_DIR}/securityid_samples_20241022.csv", "rows=", len(samples))

print("[6] database-level summary...")
if not table_summary.empty:
    db_summary = (
        table_summary.groupby(["database", "table_type_guess"])
        .agg(
            n_tables=("table_name", "count"),
            n_date_tables=("is_date_table", "sum"),
            total_rows=("total_rows", lambda x: pd.to_numeric(x, errors="coerce").sum()),
            total_bytes=("total_bytes", lambda x: pd.to_numeric(x, errors="coerce").sum()),
        )
        .reset_index()
        .sort_values(["database", "table_type_guess"])
    )
else:
    db_summary = pd.DataFrame()

db_summary.to_csv(f"{OUT_DIR}/database_summary.csv", index=False)
print("[SAVED]", f"{OUT_DIR}/database_summary.csv")

print("\n===== TABLE TYPE COUNTS =====")
print(table_summary["table_type_guess"].value_counts(dropna=False).to_string())

print("\n===== DATABASE SUMMARY =====")
print(db_summary.to_string(index=False))

print("\n===== PROBE STATS =====")
if not stats.empty:
    show_cols = ["database", "table_name", "rows", "n_symbols", "rows_per_symbol", "freq_guess", "table_type_guess", "min_ts", "max_ts"]
    print(stats[show_cols].to_string(index=False))
else:
    print("no probe stats")

print("\n===== OUTPUT DIR =====")
print(OUT_DIR)
