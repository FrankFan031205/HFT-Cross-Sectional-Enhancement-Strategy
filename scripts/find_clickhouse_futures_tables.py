# -*- coding: utf-8 -*-
import json
import urllib.parse
import urllib.request

HOST = "127.0.0.1"
PORT = 18123
USER = "default"
PASSWORD = ""

BASE_URL = f"http://{HOST}:{PORT}/"


def ch_query(sql, fmt="JSONEachRow"):
    q = sql.strip()
    if fmt:
        q += f" FORMAT {fmt}"

    url = BASE_URL
    params = []
    if USER:
        params.append(("user", USER))
    if PASSWORD:
        params.append(("password", PASSWORD))
    if params:
        url += "?" + urllib.parse.urlencode(params)

    req = urllib.request.Request(url, data=q.encode("utf-8"), method="POST")

    with urllib.request.urlopen(req, timeout=120) as r:
        txt = r.read().decode("utf-8")

    if fmt == "JSONEachRow":
        out = []
        for line in txt.splitlines():
            line = line.strip()
            if line:
                out.append(json.loads(line))
        return out

    return txt


def print_rows(title, rows, max_rows=200):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    if not rows:
        print("[EMPTY]")
        return

    for i, r in enumerate(rows[:max_rows]):
        print(f"[{i}] {r}")

    if len(rows) > max_rows:
        print(f"... total rows = {len(rows)}, only printed first {max_rows}")


def main():
    print("\n===== ClickHouse databases =====")
    print(ch_query("SHOW DATABASES", fmt="TabSeparated"))

    table_sql = """
    SELECT
        database,
        name AS table,
        engine,
        total_rows
    FROM system.tables
    WHERE
        lower(name) LIKE '%future%'
        OR lower(name) LIKE '%futures%'
        OR lower(name) LIKE '%index%'
        OR lower(name) LIKE '%idx%'
        OR lower(name) LIKE '%ctp%'
        OR lower(name) LIKE '%tick%'
        OR lower(name) LIKE '%quote%'
        OR name LIKE '%IF%'
        OR name LIKE '%IC%'
        OR name LIKE '%IH%'
        OR name LIKE '%IM%'
    ORDER BY database, table
    """

    rows = ch_query(table_sql)
    print_rows("Candidate futures/index/tick tables", rows, max_rows=300)

    col_sql = """
    SELECT
        database,
        table,
        groupArray(name) AS cols
    FROM system.columns
    WHERE
        lower(name) LIKE '%instrument%'
        OR lower(name) LIKE '%symbol%'
        OR lower(name) LIKE '%security%'
        OR lower(name) LIKE '%datetime%'
        OR lower(name) LIKE '%timestamp%'
        OR lower(name) LIKE '%trading%'
        OR lower(name) LIKE '%lastprice%'
        OR lower(name) LIKE '%last_price%'
        OR lower(name) LIKE '%bidprice%'
        OR lower(name) LIKE '%askprice%'
        OR lower(name) LIKE '%bid1%'
        OR lower(name) LIKE '%ask1%'
        OR lower(name) LIKE '%volume%'
        OR lower(name) LIKE '%turnover%'
        OR lower(name) LIKE '%openinterest%'
    GROUP BY database, table
    HAVING length(cols) >= 3
    ORDER BY database, table
    """

    rows = ch_query(col_sql)
    print_rows("Tables with futures-like columns", rows, max_rows=500)

    print("\n===== Next step =====")
    print("把上面最像股指期货的 database/table/cols 贴给我。")
    print("重点找字段类似：InstrumentID / datetime / LastPrice / BidPrice1 / AskPrice1 / Volume / Turnover。")
    print("合约代码重点找：IM, IC, IF, IH。")


if __name__ == "__main__":
    main()
