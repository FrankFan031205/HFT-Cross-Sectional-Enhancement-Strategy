# -*- coding: utf-8 -*-
"""
Optimizer-side data access helpers.

v1 先保留一个 zzy parquet loader wrapper。
后续如果把 pred/quotes 导入 ClickHouse，可以在这里补 load_optimizer_input_from_clickhouse。
"""
import importlib.util
from pathlib import Path


def load_zzy_master(
    dates=None,
    horizons=None,
    models=("ts", "res"),
    n_workers=48,
    loader_path="/mnt/data1/fwz/HFT_010-dev_fwz/TakerOptimizer/external/zzy_optimizer_data_loader/data_loader.py",
):
    """
    Load other intern's optimizer master table from parquet roots.

    This does not copy data. It imports the external loader and returns the in-memory joined table.
    """
    loader_path = Path(loader_path)
    if not loader_path.exists():
        raise FileNotFoundError(
            f"Cannot find zzy data_loader.py: {loader_path}\n"
            "Put it under TakerOptimizer/external/zzy_optimizer_data_loader/data_loader.py first."
        )

    spec = importlib.util.spec_from_file_location("zzy_optimizer_data_loader", str(loader_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    return mod.load_master(
        dates=dates,
        horizons=horizons,
        models=models,
        n_workers=n_workers,
    )
