from pathlib import Path
import pandas as pd
import yaml


def get_project_root():
    return Path(__file__).resolve().parents[2]


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(path):
    p = Path(path)
    if p.is_absolute():
        return p
    return get_project_root() / p


def ensure_parent(path):
    p = resolve_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def read_csv_smart(path, **kwargs):
    path = resolve_path(path)
    for enc in ["utf-8-sig", "utf-8", "gbk"]:
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, **kwargs)


def pick_existing_path(primary, fallback=None):
    p = Path(primary)
    if p.exists():
        return str(p)
    if fallback is not None and Path(fallback).exists():
        return str(Path(fallback))
    return str(p)
