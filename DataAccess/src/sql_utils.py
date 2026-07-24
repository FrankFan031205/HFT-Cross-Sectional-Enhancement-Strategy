# -*- coding: utf-8 -*-
import re

_SAFE_IDENT_RE = re.compile(r"^[A-Za-z0-9_]+$")


def quote_ident(name: str) -> str:
    """Safely quote ClickHouse identifiers, including db.table and numeric table names."""
    parts = str(name).split(".")
    quoted = []
    for p in parts:
        if not _SAFE_IDENT_RE.match(p):
            raise ValueError(f"Unsafe SQL identifier: {name}")
        quoted.append(f"`{p}`")
    return ".".join(quoted)


def sql_str(x) -> str:
    s = str(x).replace("\\", "\\\\").replace("'", "\\'")
    return f"'{s}'"


def sql_int_list(xs) -> str:
    vals = [str(int(x)) for x in xs]
    if not vals:
        raise ValueError("empty int list")
    return ", ".join(vals)


def sql_str_list(xs) -> str:
    vals = [sql_str(x) for x in xs]
    if not vals:
        raise ValueError("empty string list")
    return ", ".join(vals)


def normalize_sid_values(sids):
    out = set()
    for sid in sids:
        raw = str(sid).strip()
        if raw == "":
            continue
        out.add(raw)
        if raw.isdigit():
            out.add(str(int(raw)))
            out.add(raw.zfill(6))
    return sorted(out)


def sid_filter(sid_col: str, sids) -> str:
    if sids is None:
        return "1"
    sids = list(sids)
    if not sids:
        return "1"
    sid_vals = normalize_sid_values(sids)
    return f"toString({quote_ident(sid_col)}) IN ({sql_str_list(sid_vals)})"


def date_filter(time_col: str, dates) -> str:
    if dates is None:
        return "1"
    dates = list(dates)
    if not dates:
        return "1"
    return f"toYYYYMMDD({quote_ident(time_col)}) IN ({sql_int_list(dates)})"


def build_where(filters) -> str:
    valid = [f for f in filters if f and f != "1"]
    if not valid:
        return "1"
    return " AND ".join(f"({f})" for f in valid)


def select_clause(columns) -> str:
    if columns is None or columns == "*" or columns == ["*"]:
        return "*"
    return ", ".join(quote_ident(c) for c in columns)


def limit_clause(limit) -> str:
    if limit is None:
        return ""
    return f"LIMIT {int(limit)}"
