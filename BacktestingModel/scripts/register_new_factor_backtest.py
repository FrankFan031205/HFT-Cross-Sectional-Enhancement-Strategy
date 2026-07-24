import argparse
import os
import subprocess
from pathlib import Path

import pandas as pd
import yaml


DEFAULT_REGISTRY = "config/experiments/factor_registry.yaml"


DEFAULT_REGISTRY_CONTENT = {
    "defaults": {
        "base_config_path": "config/backtest.yaml",
        "fill_model": {
            "mode": "queue_aware_trade",
            "queue_ahead_multiplier": 0.05,
        },
        "policy": {
            "name": "abs_top40",
            "type": "abs_signal_quantile",
            "quantile": 0.60,
        },
        "inventory": {
            "initial_position_per_symbol": 5000,
            "max_position_per_symbol": 15000,
            "sell_floor_position": 3000,
            "buy_block_position": 9000,
            "tplus1": True,
            "allow_short": False,
        },
        "fee": {
            "mode": "single_fill_side_specific",
            "commission_rate": 0.00005,
            "transfer_fee_rate": 0.00001,
            "handling_fee_rate": 0.0000341,
            "regulatory_fee_rate": 0.00002,
            "stamp_duty_rate": 0.0005,
        },
    },
    "experiments": [],
}


def run(cmd):
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def load_registry(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        with open(path, "w") as f:
            yaml.safe_dump(DEFAULT_REGISTRY_CONTENT, f, sort_keys=False, allow_unicode=True)

    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}

    data.setdefault("defaults", DEFAULT_REGISTRY_CONTENT["defaults"])
    data.setdefault("experiments", [])

    return data


def save_registry(path, data):
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def bool_count(s):
    return s.astype(str).str.lower().isin(["1", "true", "yes"]).sum()


def validate_quote_file(path, signal_col, nrows):
    print("===== validate quote decision file =====")
    print("path:", path)
    print("sample rows:", nrows)

    if not Path(path).exists():
        raise FileNotFoundError(f"quote file not found: {path}")

    df = pd.read_csv(path, nrows=nrows, low_memory=False)
    cols = df.columns.tolist()

    print("\n===== columns =====")
    print(cols)

    required = ["datetime", "securityid", "quote_bid", "quote_ask"]
    missing = [c for c in required if c not in cols]

    if missing:
        raise RuntimeError(f"missing required columns: {missing}")

    bid_price_ok = ("bid_price" in cols) or ("bid_quote_price" in cols)
    ask_price_ok = ("ask_price" in cols) or ("ask_quote_price" in cols)

    if not bid_price_ok:
        raise RuntimeError("missing bid price column: need bid_price or bid_quote_price")

    if not ask_price_ok:
        raise RuntimeError("missing ask price column: need ask_price or ask_quote_price")

    signal_candidates = [
        c for c in cols
        if "hidden" in c.lower()
        or "factor" in c.lower()
        or "pred" in c.lower()
        or "alpha" in c.lower()
    ]

    if signal_col:
        if signal_col not in cols:
            raise RuntimeError(
                f"signal_col not found: {signal_col}\n"
                f"candidate signal columns: {signal_candidates}"
            )
        final_signal_col = signal_col
    else:
        if len(signal_candidates) == 1:
            final_signal_col = signal_candidates[0]
            print("\nauto detected signal_col:", final_signal_col)
        else:
            raise RuntimeError(
                "signal_col not provided and cannot auto-detect uniquely.\n"
                f"candidate signal columns: {signal_candidates}"
            )

    print("\n===== basic info =====")
    print("sample shape:", df.shape)
    print("datetime range sample:", df["datetime"].min(), "->", df["datetime"].max())
    print("num securities sample:", df["securityid"].nunique())
    print("quote_bid true count sample:", bool_count(df["quote_bid"]))
    print("quote_ask true count sample:", bool_count(df["quote_ask"]))
    print("signal_col:", final_signal_col)

    if bool_count(df["quote_bid"]) == 0 and bool_count(df["quote_ask"]) == 0:
        raise RuntimeError("quote_bid and quote_ask are both empty in sample")

    print("\nquote file validation passed")

    return final_signal_col


def update_registry(registry_path, tag, quote_path, signal_col, overwrite=False):
    data = load_registry(registry_path)
    experiments = data["experiments"]

    existing_idx = None
    for i, item in enumerate(experiments):
        if item.get("tag") == tag:
            existing_idx = i
            break

    new_item = {
        "tag": tag,
        "quote_decision_path": quote_path,
        "signal_col": signal_col,
    }

    if existing_idx is not None:
        if not overwrite:
            raise RuntimeError(
                f"tag already exists in registry: {tag}\n"
                f"use --overwrite to update it"
            )
        experiments[existing_idx] = new_item
        print(f"updated existing registry entry: {tag}")
    else:
        experiments.append(new_item)
        print(f"added new registry entry: {tag}")

    save_registry(registry_path, data)

    print("\nregistry saved:", registry_path)


def generated_config_path(tag):
    return f"config/experiments/generated/{tag}.yaml"


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--tag", required=True, help="experiment tag, e.g. attention_h60_202410_100")
    parser.add_argument("--quote-path", required=True, help="path to quote decision csv")
    parser.add_argument("--signal-col", default="", help="signal column name; optional if can be auto-detected")
    parser.add_argument("--registry", default=DEFAULT_REGISTRY)
    parser.add_argument("--nrows", type=int, default=10000)

    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--build", action="store_true", help="run build_experiment_configs.py after updating registry")
    parser.add_argument("--sample", action="store_true", help="run small sample backtest")
    parser.add_argument("--max-quotes", type=int, default=5000)
    parser.add_argument("--full", action="store_true", help="start full backtest in background with nohup")

    args = parser.parse_args()

    Path("logs").mkdir(exist_ok=True)
    Path("config/experiments").mkdir(parents=True, exist_ok=True)

    final_signal_col = validate_quote_file(
        path=args.quote_path,
        signal_col=args.signal_col,
        nrows=args.nrows,
    )

    update_registry(
        registry_path=args.registry,
        tag=args.tag,
        quote_path=args.quote_path,
        signal_col=final_signal_col,
        overwrite=args.overwrite,
    )

    if args.build or args.sample or args.full:
        run(["python", "scripts/build_experiment_configs.py", "--registry", args.registry])

    exp_path = generated_config_path(args.tag)

    if args.sample:
        run([
            "python", "scripts/run_backtest_experiment.py",
            "--exp", exp_path,
            "--max-quotes", str(args.max_quotes),
        ])

    if args.full:
        log_path = f"logs/{args.tag}_backtest_experiment.log"
        cmd = (
            f"nohup python scripts/run_backtest_experiment.py "
            f"--exp {exp_path} "
            f"> {log_path} 2>&1 &"
        )
        print("\n$ " + cmd)
        subprocess.run(cmd, shell=True, check=True)
        print("full backtest started in background")
        print("log:", log_path)
        print(f"tail -f {log_path}")

    print("\n===== done =====")
    print("tag:", args.tag)
    print("quote_path:", args.quote_path)
    print("signal_col:", final_signal_col)
    print("registry:", args.registry)
    print("generated_config:", exp_path)


if __name__ == "__main__":
    main()
