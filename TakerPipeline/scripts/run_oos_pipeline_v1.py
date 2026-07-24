import argparse
import os
import sys
import json
import yaml
import glob
import subprocess
from pathlib import Path

import pandas as pd


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def check_file(path, name, required=True):
    exists = os.path.exists(path)
    print(f"[check] {name}: {path}")
    print(f"        exists={exists}")
    if required and not exists:
        raise FileNotFoundError(f"{name} not found: {path}")
    return exists


def read_date_range_csv(path, date_col="date", symbol_col="securityid", nrows=None):
    usecols = None
    header = pd.read_csv(path, nrows=0)
    cols = [str(c).lower() for c in header.columns]

    real_date_col = None
    real_symbol_col = None

    for c in header.columns:
        if str(c).lower() == date_col.lower():
            real_date_col = c
        if str(c).lower() == symbol_col.lower():
            real_symbol_col = c

    if real_date_col is None:
        return None

    usecols = [real_date_col]
    if real_symbol_col is not None:
        usecols.append(real_symbol_col)

    df = pd.read_csv(path, usecols=usecols, low_memory=False, nrows=nrows)
    df.columns = [str(c).lower() for c in df.columns]

    out = {
        "rows": len(df),
        "date_min": str(df["date"].min()),
        "date_max": str(df["date"].max()),
    }

    if "securityid" in df.columns:
        out["n_symbols"] = int(df["securityid"].nunique())

    return out


def inspect_market_files(cfg):
    market_dir = cfg["data"]["market_data_dir"]
    pattern = cfg["data"]["market_file_pattern"]
    test_start = str(cfg["dates"]["test_start_date"])
    test_end = str(cfg["dates"]["test_end_date"])

    files = []
    for fp in sorted(glob.glob(os.path.join(market_dir, pattern.replace("{date}", "*")))):
        base = os.path.basename(fp)
        digits = "".join([x for x in base if x.isdigit()])
        if len(digits) >= 8:
            date = digits[:8]
            if test_start <= date <= test_end:
                files.append((date, fp))

    print(f"[market] dir={market_dir}")
    print(f"[market] test files found={len(files)}")
    if files:
        print("[market] first:", files[0])
        print("[market] last :", files[-1])

    if not files:
        raise FileNotFoundError("No OOS market files found.")

    return files


def inspect_checkpoint(path):
    print("[checkpoint] inspecting:", path)
    try:
        import torch
        ckpt = torch.load(path, map_location="cpu")
    except Exception as e:
        print("[checkpoint] failed to load checkpoint:", repr(e))
        return {"load_ok": False, "error": repr(e)}

    info = {"load_ok": True, "type": str(type(ckpt))}

    if isinstance(ckpt, dict):
        info["keys"] = list(ckpt.keys())
        print("[checkpoint] keys:")
        for k in ckpt.keys():
            print("  ", k)

        shape_info = {}
        for k, v in ckpt.items():
            if hasattr(v, "shape"):
                shape_info[k] = tuple(v.shape)
            elif isinstance(v, dict):
                shape_info[k] = f"dict({len(v)})"
        info["shape_info"] = shape_info
    else:
        print("[checkpoint] object:", ckpt)

    return info


def inspect_train_hidden_factor(cfg):
    path = cfg["model"]["train_hidden_factor_path"]
    check_file(path, "train_hidden_factor", required=True)

    head = pd.read_csv(path, nrows=5)
    print("[train hidden] columns:", head.columns.tolist())
    print(head)

    stat = read_date_range_csv(path)
    print("[train hidden] stats:", stat)

    return {
        "columns": head.columns.tolist(),
        "stats": stat,
    }


def check_oos_hidden_factor(cfg):
    path = cfg["model"]["oos_hidden_factor_path"]
    exists = check_file(path, "oos_hidden_factor", required=False)

    if not exists:
        print()
        print("[STOP] OOS hidden factor does not exist yet.")
        print("       Next step: run frozen-model inference to create:")
        print("       ", path)
        return False

    stat = read_date_range_csv(path)
    print("[oos hidden] stats:", stat)

    test_start = str(cfg["dates"]["test_start_date"])
    test_end = str(cfg["dates"]["test_end_date"])

    if stat is not None:
        if stat["date_min"] < test_start or stat["date_max"] > test_end:
            raise ValueError(
                f"OOS hidden factor date range invalid: {stat}. "
                f"Expected within {test_start}-{test_end}"
            )

    return True


def write_taker_config(cfg):
    taker_cfg_path = cfg["taker_backtest"]["config_path"]
    ensure_dir(os.path.dirname(taker_cfg_path))

    output_dir = cfg["taker_backtest"]["output_dir"]
    ensure_dir(output_dir)
    ensure_dir(os.path.join(output_dir, "positions"))
    ensure_dir(os.path.join(output_dir, "metrics"))

    label_col = cfg["pricing"]["label_col"]
    target_pos_path = cfg["optimizer"]["output_path"]

    taker_cfg = f"""project:
  name: taker_model_feature_transformer_h120_oos
  version: v3b_oos
  seed: 42

data:
  market_data_dir: {cfg["data"]["market_data_dir"]}
  market_file_pattern: {cfg["data"]["market_file_pattern"]}
  optimizer_output_path: {target_pos_path}

  start_date: {cfg["dates"]["test_start_date"]}
  end_date: {cfg["dates"]["test_end_date"]}

output:
  position_output_path: {output_dir}/positions/taker_positions_feature_transformer_h120_oos.csv
  minute_metrics_path: {output_dir}/metrics/taker_minute_metrics_feature_transformer_h120_oos.csv
  daily_metrics_path: {output_dir}/metrics/taker_daily_metrics_feature_transformer_h120_oos.csv
  trade_reason_path: {output_dir}/metrics/taker_trade_reason_feature_transformer_h120_oos.csv
  summary_path: {output_dir}/metrics/taker_summary_feature_transformer_h120_oos.csv

columns:
  datetime_col: datetime
  date_col: date
  symbol_col: securityid

  target_weight_col: effective_target_weight
  label_col: {label_col}

  bid_col: bid1
  ask_col: ask1
  mid_col: mid_price

time_alignment:
  optimizer_datetime_role: decision_time
  execution_lag_minutes: {cfg["taker_backtest"]["execution_lag_minutes"]}

execution:
  capital: {cfg["taker_backtest"]["capital"]}
  taker_fee_bps: {cfg["taker_backtest"]["taker_fee_bps"]}
  slippage_bps: {cfg["taker_backtest"]["slippage_bps"]}

  entry_rebalance_ratio: 0.5
  exit_rebalance_ratio: 1.0

filters:
  require_optimal_for_entry: false
  require_valid_market: true

  entry_min_abs_delta_notional: 50000.0
  entry_max_spread_bps: 10.0
  entry_min_abs_net_alpha_bps: null

  exit_max_spread_bps: 999.0
  hold_min_abs_net_alpha_bps: null

  exit_when_target_zero: true
  exit_when_direction_flip: true
  reduce_when_target_smaller: false

runtime:
  market_chunksize: 2000000
  max_missing_rate: 0.2
"""

    with open(taker_cfg_path, "w") as f:
        f.write(taker_cfg)

    print("[write] taker config:", taker_cfg_path)
    return taker_cfg_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", default="check", choices=[
        "check",
        "write_taker_config",
        "all_check",
    ])
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    project_root = cfg["paths"]["project_root"]
    output_root = cfg["paths"]["output_root"]

    print("===== OOS PIPELINE V1 =====")
    print("project_root:", project_root)
    print("output_root:", output_root)
    print("stage:", args.stage)
    print()

    ensure_dir(output_root)
    ensure_dir(os.path.join(output_root, "hidden_factor"))
    ensure_dir(os.path.join(output_root, "pricing"))
    ensure_dir(os.path.join(output_root, "target_positions"))
    ensure_dir(os.path.join(output_root, "logs"))
    ensure_dir(os.path.join(output_root, "manifest"))

    check_file(cfg["model"]["frozen_model_path"], "frozen_model", required=True)
    inspect_checkpoint(cfg["model"]["frozen_model_path"])

    inspect_train_hidden_factor(cfg)

    inspect_market_files(cfg)

    check_file(cfg["pricing"]["train_priced_dataset_path"], "train_priced_dataset", required=True)
    train_pricing_stat = read_date_range_csv(cfg["pricing"]["train_priced_dataset_path"])
    print("[train pricing] stats:", train_pricing_stat)

    oos_hidden_exists = check_oos_hidden_factor(cfg)

    if args.stage in ["write_taker_config", "all_check"]:
        write_taker_config(cfg)

    manifest = {
        "config": args.config,
        "project_root": project_root,
        "output_root": output_root,
        "train_start_date": cfg["dates"]["train_start_date"],
        "train_end_date": cfg["dates"]["train_end_date"],
        "test_start_date": cfg["dates"]["test_start_date"],
        "test_end_date": cfg["dates"]["test_end_date"],
        "frozen_model_path": cfg["model"]["frozen_model_path"],
        "oos_hidden_factor_path": cfg["model"]["oos_hidden_factor_path"],
        "oos_hidden_exists": oos_hidden_exists,
        "optimizer_output_path": cfg["optimizer"]["output_path"],
    }

    manifest_path = os.path.join(output_root, "manifest", "oos_pipeline_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print("[write] manifest:", manifest_path)
    print()
    print("===== DONE =====")


if __name__ == "__main__":
    main()
