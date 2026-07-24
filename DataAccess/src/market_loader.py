# -*- coding: utf-8 -*-
import polars as pl

from DataAccess.src.clickhouse_client import (
    get_table_config,
    query_df,
    query_scalar,
)
from DataAccess.src.sql_utils import (
    quote_ident,
    date_filter,
    sid_filter,
    build_where,
    select_clause,
    limit_clause,
)


def _table_name_for_date(tcfg, date=None):
    mode = tcfg.get("mode", "single_table")
    if mode == "daily_table":
        if date is None:
            raise ValueError("daily_table mode requires one date")
        db = tcfg["database"]
        tmpl = tcfg.get("table_template", "{date}")
        table = tmpl.format(date=int(date))
        return f"{db}.{table}"
    return tcfg["table"]


def _single_select_sql(
    tcfg,
    date=None,
    dates=None,
    sids=None,
    columns=None,
    where_extra=None,
    order_by=True,
    limit=None,
):
    mode = tcfg.get("mode", "single_table")
    table = _table_name_for_date(tcfg, date=date)
    time_col = tcfg["time_col"]
    sid_col = tcfg["sid_col"]

    filters = [sid_filter(sid_col, sids)]
    if mode != "daily_table":
        filters.append(date_filter(time_col, dates))
    if where_extra:
        filters.append(where_extra)

    col_sql = select_clause(columns)
    where_sql = build_where(filters)

    order_sql = ""
    if order_by:
        order_sql = f"ORDER BY {quote_ident(sid_col)}, {quote_ident(time_col)}"

    return f"""
    SELECT {col_sql}
    FROM {quote_ident(table)}
    WHERE {where_sql}
    {order_sql}
    {limit_clause(limit)}
    """


def load_table(
    table_key,
    dates=None,
    sids=None,
    columns=None,
    where_extra=None,
    order_by=True,
    limit=None,
    config_path=None,
    as_polars=True,
):
    """
    Generic ClickHouse loader.

    Supports:
      single_table: normal db.table + date filter by time_col
      daily_table : database is data type, table is YYYYMMDD
    """
    tcfg = get_table_config(table_key, config_path)
    mode = tcfg.get("mode", "single_table")

    if mode == "daily_table":
        if dates is None:
            raise ValueError("daily_table mode requires dates=[...]")
        dates = list(dates)
        if not dates:
            raise ValueError("empty dates for daily_table")

        parts = []
        for d in dates:
            sql = _single_select_sql(
                tcfg,
                date=d,
                dates=[d],
                sids=sids,
                columns=columns,
                where_extra=where_extra,
                order_by=order_by,
                limit=limit,
            )
            parts.append(query_df(sql, config_path=config_path, as_polars=as_polars))

        if not as_polars:
            import pandas as pd
            return pd.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0]

        return pl.concat(parts, how="vertical") if len(parts) > 1 else parts[0]

    sql = _single_select_sql(
        tcfg,
        date=None,
        dates=dates,
        sids=sids,
        columns=columns,
        where_extra=where_extra,
        order_by=order_by,
        limit=limit,
    )
    return query_df(sql, config_path=config_path, as_polars=as_polars)


def count_rows(table_key, dates=None, sids=None, where_extra=None, config_path=None):
    tcfg = get_table_config(table_key, config_path)
    mode = tcfg.get("mode", "single_table")

    if mode == "daily_table":
        if dates is None:
            raise ValueError("daily_table mode requires dates=[...]")
        total = 0
        for d in dates:
            table = _table_name_for_date(tcfg, date=d)
            sid_col = tcfg["sid_col"]
            filters = [sid_filter(sid_col, sids)]
            if where_extra:
                filters.append(where_extra)
            sql = f"""
            SELECT count()
            FROM {quote_ident(table)}
            WHERE {build_where(filters)}
            """
            total += int(query_scalar(sql, config_path=config_path) or 0)
        return total

    table = _table_name_for_date(tcfg)
    time_col = tcfg["time_col"]
    sid_col = tcfg["sid_col"]
    filters = [date_filter(time_col, dates), sid_filter(sid_col, sids)]
    if where_extra:
        filters.append(where_extra)

    sql = f"""
    SELECT count()
    FROM {quote_ident(table)}
    WHERE {build_where(filters)}
    """
    return query_scalar(sql, config_path=config_path)


def describe_table(table_key, sample_date=None, config_path=None, as_polars=True):
    tcfg = get_table_config(table_key, config_path)
    if tcfg.get("mode", "single_table") == "daily_table":
        if sample_date is None:
            raise ValueError("describe_table for daily_table requires sample_date")
        table = _table_name_for_date(tcfg, date=sample_date)
    else:
        table = _table_name_for_date(tcfg)

    sql = f"DESCRIBE TABLE {quote_ident(table)}"
    return query_df(sql, config_path=config_path, as_polars=as_polars)


def load_snapshot_500ms(dates=None, sids=None, columns=None, limit=None, config_path=None):
    return load_table("snapshot_500ms", dates=dates, sids=sids, columns=columns, limit=limit, config_path=config_path)


def load_trade(dates=None, sids=None, columns=None, limit=None, config_path=None):
    return load_table("trade", dates=dates, sids=sids, columns=columns, limit=limit, config_path=config_path)


def load_order(dates=None, sids=None, columns=None, limit=None, config_path=None):
    return load_table("order", dates=dates, sids=sids, columns=columns, limit=limit, config_path=config_path)


def load_cancel(dates=None, sids=None, columns=None, limit=None, config_path=None):
    return load_table("cancel", dates=dates, sids=sids, columns=columns, limit=limit, config_path=config_path)


def load_index_500ms(dates=None, sids=None, columns=None, limit=None, config_path=None):
    return load_table("index_500ms", dates=dates, sids=sids, columns=columns, limit=limit, config_path=config_path)
