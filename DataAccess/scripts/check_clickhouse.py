# -*- coding: utf-8 -*-
import argparse

from DataAccess.src.clickhouse_client import ping, query_scalar, load_config
from DataAccess.src.market_loader import count_rows, describe_table, load_table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", type=int, default=20241022)
    ap.add_argument("--sids", nargs="*", default=["000001", "000002"])
    ap.add_argument("--limit", type=int, default=5)
    args = ap.parse_args()

    print("===== ClickHouse connection =====")
    print("ping:", ping())
    print("version:", query_scalar("SELECT version()"))

    cfg = load_config()
    print("\n===== Configured tables =====")
    for key, info in cfg["tables"].items():
        print(f"{key}: {info}")

    print("\n===== Table checks =====")
    for key in cfg["tables"].keys():
        print(f"\n--- {key} ---")
        try:
            desc = describe_table(key, sample_date=args.date)
            print("[schema head]")
            print(desc.head(30))

            n = count_rows(key, dates=[args.date], sids=args.sids)
            print(f"[count] date={args.date}, sids={args.sids}: {n}")

            if n and n > 0:
                df = load_table(
                    key,
                    dates=[args.date],
                    sids=args.sids,
                    columns=None,
                    limit=args.limit,
                )
                print("[sample]")
                print(df)
            else:
                print("[sample] empty under this date/sid filter")
        except Exception as e:
            print("[ERROR]", repr(e))


if __name__ == "__main__":
    main()
