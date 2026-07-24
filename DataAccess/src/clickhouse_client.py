# -*- coding: utf-8 -*-
from pathlib import Path

import yaml


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "clickhouse.yaml"


def load_config(config_path=None):
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_client(config_path=None):
    try:
        import clickhouse_connect
    except ImportError as e:
        raise ImportError(
            "Missing dependency: clickhouse-connect. "
            "Install it with: pip install clickhouse-connect"
        ) from e

    cfg = load_config(config_path)
    ch = cfg["clickhouse"]

    return clickhouse_connect.get_client(
        host=ch.get("host", "127.0.0.1"),
        port=int(ch.get("port", 18123)),
        username=ch.get("username", "default"),
        password=ch.get("password", ""),
        database=ch.get("database", "default"),
    )


def get_table_config(table_key, config_path=None):
    cfg = load_config(config_path)
    tables = cfg.get("tables", {})
    if table_key not in tables:
        raise KeyError(f"Unknown table_key={table_key}. Available={list(tables.keys())}")
    return tables[table_key]


def query_df(sql, config_path=None, settings=None, as_polars=True):
    client = get_client(config_path)
    cfg = load_config(config_path)
    final_settings = dict(cfg.get("settings", {}) or {})
    if settings:
        final_settings.update(settings)

    pdf = client.query_df(sql, settings=final_settings)

    if as_polars:
        import polars as pl
        return pl.from_pandas(pdf)
    return pdf


def query_scalar(sql, config_path=None, settings=None):
    client = get_client(config_path)
    cfg = load_config(config_path)
    final_settings = dict(cfg.get("settings", {}) or {})
    if settings:
        final_settings.update(settings)

    res = client.query(sql, settings=final_settings)
    if not res.result_rows:
        return None
    return res.result_rows[0][0]


def ping(config_path=None):
    client = get_client(config_path)
    return client.ping()
