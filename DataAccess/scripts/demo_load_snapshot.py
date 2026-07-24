# -*- coding: utf-8 -*-
from DataAccess.src.market_loader import load_snapshot_500ms


def main():
    df = load_snapshot_500ms(
        dates=[20241022],
        sids=["000001", "000002"],
        limit=10,
    )
    print(df)
    print(df.schema)


if __name__ == "__main__":
    main()
