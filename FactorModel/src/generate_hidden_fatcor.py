"""
Generate hidden_factor_xx.csv from a FactorModel config.

Run from FactorModel root:

    cd /home/fwz/projects/HFT_010-dev_fwz/FactorModel
    python src/generate_hidden_factor.py --config configs/factor_model_lgbm.yaml

This script reads the YAML config, identifies model.type, calls the correct
training script, and verifies prediction.output_path was created.
"""

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

import yaml


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def get_model_type(cfg: dict) -> str:
    return str(cfg.get("model", {}).get("type", "")).strip()


def get_project_name(cfg: dict, config_path: str) -> str:
    name = cfg.get("project", {}).get("name")
    return str(name) if name else Path(config_path).stem


def get_prediction_path(cfg: dict) -> str:
    path = cfg.get("prediction", {}).get("output_path")
    if not path:
        raise RuntimeError("Missing prediction.output_path in config.")
    return str(path)


def dispatch_script(model_type: str) -> str:
    if model_type in {"ridge", "lightgbm"}:
        return "src/train_model.py"
    if model_type in {"feature_attention_nn", "attention", "attention_nn"}:
        return "src/train_attention_nn.py"
    if model_type in {"mlp", "residual_mlp"}:
        return "src/train_mlp.py"
    if model_type in {"lookback_attention_nn", "lookback_attention", "temporal_attention"}:
        return "src/train_lookback_attention.py"
    raise RuntimeError(
        f"Unknown model.type: {model_type}. Supported types: ridge, lightgbm, "
        "feature_attention_nn, mlp, residual_mlp, lookback_attention_nn."
    )


def detect_hidden_factor_col(csv_path: str) -> str:
    with open(csv_path, "r") as f:
        header = next(csv.reader(f))
    candidates = [c for c in header if c.startswith("hidden_factor_")]
    if not candidates:
        raise RuntimeError(f"No hidden_factor_* column found in {csv_path}. Columns={header}")
    if len(candidates) > 1:
        print(f"WARNING: multiple hidden_factor_* columns found: {candidates}. Use first one.", flush=True)
    return candidates[0]


def summarize_output(csv_path: str, factor_col: str, chunksize: int = 1_000_000) -> None:
    import pandas as pd

    header = pd.read_csv(csv_path, nrows=0).columns.tolist()
    usecols = [c for c in ["date", "securityid", factor_col, "split"] if c in header]

    total = 0
    non_null = 0
    min_date = None
    max_date = None
    stocks = set()
    split_counts = {}

    for chunk in pd.read_csv(csv_path, usecols=usecols, dtype={"securityid": str}, chunksize=chunksize):
        total += len(chunk)
        non_null += int(chunk[factor_col].notna().sum())
        if "date" in chunk.columns:
            cmin = chunk["date"].min()
            cmax = chunk["date"].max()
            min_date = cmin if min_date is None else min(min_date, cmin)
            max_date = cmax if max_date is None else max(max_date, cmax)
        if "securityid" in chunk.columns:
            stocks.update(chunk["securityid"].dropna().unique().tolist())
        if "split" in chunk.columns:
            vc = chunk["split"].value_counts()
            for k, v in vc.items():
                split_counts[k] = split_counts.get(k, 0) + int(v)

    print("\n=== Output Check ===", flush=True)
    print(f"file: {csv_path}", flush=True)
    print(f"factor_col: {factor_col}", flush=True)
    print(f"rows: {total}", flush=True)
    print(f"non_null_prediction_rows: {non_null}", flush=True)
    print(f"non_null_ratio: {non_null / total if total else 0:.6f}", flush=True)
    print(f"date_range: {min_date} ~ {max_date}", flush=True)
    print(f"num_stocks: {len(stocks)}", flush=True)
    if split_counts:
        print(f"split_counts: {split_counts}", flush=True)


def run_one_config(config_path: str, python_bin: str, log_dir: str, dry_run: bool, skip_if_exists: bool) -> None:
    cfg = load_config(config_path)
    model_type = get_model_type(cfg)
    project_name = get_project_name(cfg, config_path)
    output_path = get_prediction_path(cfg)
    train_script = dispatch_script(model_type)

    if skip_if_exists and os.path.exists(output_path):
        print(f"[SKIP] {output_path} already exists.", flush=True)
        factor_col = detect_hidden_factor_col(output_path)
        summarize_output(output_path, factor_col)
        return

    if not os.path.exists(train_script):
        raise FileNotFoundError(
            f"Cannot find training script: {train_script}. Run this command from FactorModel root."
        )

    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"generate_hidden_factor_{project_name}.log")
    cmd = [python_bin, "-u", train_script, "--config", config_path]

    print("\n=== Generate Hidden Factor ===", flush=True)
    print(f"config: {config_path}", flush=True)
    print(f"model.type: {model_type}", flush=True)
    print(f"train_script: {train_script}", flush=True)
    print(f"output_path: {output_path}", flush=True)
    print(f"log_path: {log_path}", flush=True)
    print("command:", " ".join(cmd), flush=True)

    if dry_run:
        print("dry_run=True, command not executed.", flush=True)
        return

    ensure_dir(log_path)
    with open(log_path, "w") as log_f:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in proc.stdout:
            print(line, end="")
            log_f.write(line)
            log_f.flush()
        ret = proc.wait()

    if ret != 0:
        raise RuntimeError(f"Training failed with return code {ret}. Check log: {log_path}")
    if not os.path.exists(output_path):
        raise RuntimeError(f"Training finished but output file was not found: {output_path}")

    factor_col = detect_hidden_factor_col(output_path)
    summarize_output(output_path, factor_col)
    print("\nDONE.", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", action="append", required=True, help="YAML config. Can be passed multiple times.")
    parser.add_argument("--python", default=sys.executable, help="Python executable. Default: current Python.")
    parser.add_argument("--log_dir", default="logs", help="Log directory.")
    parser.add_argument("--dry_run", action="store_true", help="Print command only.")
    parser.add_argument("--skip_if_exists", action="store_true", help="Skip if prediction.output_path already exists.")
    args = parser.parse_args()

    for config_path in args.config:
        run_one_config(
            config_path=config_path,
            python_bin=args.python,
            log_dir=args.log_dir,
            dry_run=args.dry_run,
            skip_if_exists=args.skip_if_exists,
        )


if __name__ == "__main__":
    main()
