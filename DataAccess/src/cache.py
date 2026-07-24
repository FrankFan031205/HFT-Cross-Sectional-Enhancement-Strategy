# -*- coding: utf-8 -*-
from pathlib import Path


def write_parquet(df, path, compression="zstd"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if hasattr(df, "write_parquet"):
        df.write_parquet(str(path), compression=compression)
        return str(path)

    df.to_parquet(str(path), index=False)
    return str(path)


def read_parquet(path, as_polars=True):
    path = Path(path)
    if as_polars:
        import polars as pl
        return pl.read_parquet(str(path))

    import pandas as pd
    return pd.read_parquet(str(path))
