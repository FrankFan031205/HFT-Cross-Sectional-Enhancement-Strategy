import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.io import load_yaml
from src.db import get_clickhouse_client, query_df


def choose_col(cols, candidates):
    low = {c.lower(): c for c in cols}
    for x in candidates:
        if x.lower() in low:
            return low[x.lower()]
    return None


def main():
    cfg = load_yaml("config/backtest.yaml")
    client = get_clickhouse_client(cfg, date=20241022)

    cols_df = query_df(client, """
    SELECT name, type
    FROM system.columns
    WHERE database = '500ms'
      AND table = '20241022'
    ORDER BY position
    """)

    cols = cols_df["name"].tolist()
    print("500ms columns:")
    print(cols_df.to_string(index=False))

    time_col = choose_col(cols, ["time", "timestamp", "datetime", "Time", "Timestamp"])
    sid_col = choose_col(cols, ["SecurityID", "securityid", "symbol"])
    bid_col = choose_col(cols, ["bidprice1", "bid1", "bidprice1", "bid1_price"])
    ask_col = choose_col(cols, ["askprice1", "ask1", "askprice1", "ask1_price"])

    print("\nchosen:")
    print("time_col:", time_col)
    print("sid_col:", sid_col)
    print("bid_col:", bid_col)
    print("ask_col:", ask_col)

    if sid_col is None or bid_col is None or ask_col is None:
        raise RuntimeError("Cannot identify SecurityID / bid1 / ask1 columns.")

    if time_col is None:
        raise RuntimeError("Cannot identify snapshot time column.")

    # trade price scale = 100, so divide trade price by 100.
    # Snapshot price may be either 717 or 7.17, so normalize by checking magnitude.
    snap_sample = query_df(client, f"""
    SELECT {bid_col} AS bid1, {ask_col} AS ask1
    FROM `500ms`.`20241022`
    LIMIT 100
    """)
    med_bid = snap_sample["bid1"].median()
    snap_scale_expr = "/ 100.0" if med_bid > 100 else ""
    print("snapshot median bid1:", med_bid)
    print("snapshot scale expr:", snap_scale_expr if snap_scale_expr else "no scale")

    # 如果 snapshot 时间字段是 Int32 time，按 intDiv 对齐；
    # 如果是 DateTime/DateTime64，则用 toYYYYMMDD / formatDateTime 不方便，这里先给两套 SQL。
    col_type = cols_df.loc[cols_df["name"] == time_col, "type"].iloc[0]
    print("time col type:", col_type)

    if "Int" in col_type:
        join_cond = f"intDiv(t.time, 500) * 500 = s.{time_col}"
        snap_time_filter = f"s.{time_col} >= 93000000 AND s.{time_col} < 100000000"
    else:
        # snapshot timestamp 按 500ms 对齐可能已经是 DateTime64
        join_cond = f"t.SecurityID = s.{sid_col} AND s.{time_col} >= toDateTime64('2024-10-22 09:30:00.000', 3) AND s.{time_col} < toDateTime64('2024-10-22 10:00:00.000', 3)"
        snap_time_filter = f"s.{time_col} >= toDateTime64('2024-10-22 09:30:00.000', 3) AND s.{time_col} < toDateTime64('2024-10-22 10:00:00.000', 3)"

    if "Int" in col_type:
        sql = f"""
        WITH
        trade_sample AS
        (
            SELECT
                time,
                SecurityID,
                price / 100.0 AS price,
                qty,
                side,
                seqno
            FROM `A_share_Trade`.`20241022`
            WHERE time >= 93000000
              AND time < 100000000
            LIMIT 300000
        ),
        snap AS
        (
            SELECT
                {time_col} AS time,
                {sid_col} AS SecurityID,
                {bid_col} {snap_scale_expr} AS bid1,
                {ask_col} {snap_scale_expr} AS ask1
            FROM `500ms`.`20241022`
            WHERE {time_col} >= 93000000
              AND {time_col} < 100000000
        )
        SELECT
            t.side AS side,
            count() AS n,
            avg(abs(t.price - s.ask1)) AS avg_dist_to_ask1,
            avg(abs(t.price - s.bid1)) AS avg_dist_to_bid1,
            sum(t.price >= s.ask1) AS trade_at_or_above_ask1,
            sum(t.price <= s.bid1) AS trade_at_or_below_bid1
        FROM trade_sample t
        ANY LEFT JOIN snap s
        ON t.SecurityID = s.SecurityID
        AND intDiv(t.time, 500) * 500 = s.time
        GROUP BY t.side
        ORDER BY t.side
        """
    else:
        raise RuntimeError(
            "Snapshot time column is not Int. Paste the 500ms column list to me; I will adjust DateTime join."
        )

    print("\nside diagnostic SQL:")
    print(sql)

    df = query_df(client, sql)
    print("\nside diagnostic result:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
