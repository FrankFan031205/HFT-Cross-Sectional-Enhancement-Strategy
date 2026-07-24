import argparse
import os
import subprocess
from pathlib import Path

import pandas as pd
import yaml


def qtag(x):
    return str(x).replace(".", "p")


def ensure_dirs():
    for d in [
        "outputs/config_runs",
        "outputs/fills",
        "outputs/trades",
        "outputs/metrics",
        "outputs/portfolio",
        "outputs/cache",
        "logs",
    ]:
        os.makedirs(d, exist_ok=True)


def run(cmd):
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def make_runtime_config(exp_cfg):
    tag = exp_cfg["experiment"]["tag"]
    base_config_path = exp_cfg["input"].get("base_config_path", "config/backtest.yaml")

    with open(base_config_path, "r") as f:
        cfg = yaml.safe_load(f)

    quote_path = exp_cfg["input"]["quote_decision_path"]
    signal_col = exp_cfg["signal"]["col"]

    fill_model = exp_cfg.get("fill_model", {})
    fill_mode = fill_model.get("mode", "queue_aware_trade")
    queue_mult = float(fill_model.get("queue_ahead_multiplier", 0.05))

    fill_prefix = exp_cfg.get("output", {}).get("fill_prefix", f"queue_mult_{qtag(queue_mult)}")

    cfg.setdefault("input", {})
    cfg["input"]["quote_decision_path"] = quote_path

    cfg["signal"] = {
        "col": signal_col,
        "n_bins": int(exp_cfg.get("signal", {}).get("n_bins", 5)),
    }

    cfg.setdefault("fill_model", {})
    cfg["fill_model"]["mode"] = fill_mode
    cfg["fill_model"]["queue_ahead_multiplier"] = queue_mult

    if "fee" in exp_cfg:
        cfg["fee"] = exp_cfg["fee"]

    cfg.setdefault("output", {})

    cfg["output"]["fill_path"] = f"outputs/fills/fills_{fill_prefix}_{tag}.csv"
    cfg["output"]["enriched_fill_path"] = f"outputs/fills/fills_{fill_prefix}_enriched_{tag}.csv"
    cfg["output"]["pnl_path"] = f"outputs/trades/trades_pnl_{fill_prefix}_{tag}.csv"
    cfg["output"]["daily_pnl_path"] = f"outputs/metrics/daily_pnl_{fill_prefix}_{tag}.csv"
    cfg["output"]["summary_path"] = f"outputs/metrics/summary_{fill_prefix}_{tag}.csv"
    cfg["output"]["factor_pnl_path"] = f"outputs/metrics/factor_pnl_{fill_prefix}_{tag}.csv"
    cfg["output"]["trade_cache_path"] = f"outputs/cache/raw_trades_{fill_prefix}_{tag}.csv"
    cfg["output"]["snapshot_cache_path"] = f"outputs/cache/snapshot_state_for_quotes_{fill_prefix}_{tag}.csv"

    out_path = f"outputs/config_runs/backtest_{fill_prefix}_{tag}.yaml"

    with open(out_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    return out_path, cfg, fill_prefix


def apply_policy(exp_cfg, pnl_path):
    tag = exp_cfg["experiment"]["tag"]
    signal_col = exp_cfg["signal"]["col"]

    policy = exp_cfg.get("policy", {})
    policy_name = policy.get("name", "all")
    policy_type = policy.get("type", "all")

    out_path = f"outputs/trades/trades_pnl_{policy_name}_{tag}.csv"

    df = pd.read_csv(pnl_path, low_memory=False)

    if policy_type == "all":
        sub = df.copy()
        threshold = None

    elif policy_type == "abs_signal_quantile":
        q = float(policy.get("quantile", 0.60))
        abs_s = df[signal_col].abs()
        threshold = abs_s.quantile(q)
        sub = df[abs_s >= threshold].copy()

    elif policy_type == "buy_only":
        sub = df[df["side"].astype(str).str.upper() == "BUY"].copy()
        threshold = None

    elif policy_type == "sell_only":
        sub = df[df["side"].astype(str).str.upper() == "SELL"].copy()
        threshold = None

    else:
        raise RuntimeError(f"Unsupported policy type: {policy_type}")

    sub.to_csv(out_path, index=False)

    print("\n===== policy filter =====")
    print("policy:", policy_name)
    print("type:", policy_type)
    print("threshold:", threshold)
    print("input trades:", len(df))
    print("policy trades:", len(sub))
    print("saved:", out_path)

    return out_path


def run_inventory_overlay(exp_cfg, policy_trades_path):
    tag = exp_cfg["experiment"]["tag"]
    policy_name = exp_cfg.get("policy", {}).get("name", "all")
    inv = exp_cfg.get("inventory", {})

    inv0 = int(inv.get("initial_position_per_symbol", 5000))
    cap = int(inv.get("max_position_per_symbol", 15000))
    has_sell_buy_overlay = ("sell_floor_position" in inv) or ("buy_block_position" in inv)

    if has_sell_buy_overlay:
        sell_floor = int(inv.get("sell_floor_position", 3000))
        buy_block = int(inv.get("buy_block_position", 9000))
        inventory_suffix = f"_sellfloor{sell_floor}_buyblock{buy_block}_tplus1"
    else:
        sell_floor = 0
        buy_block = cap
        inventory_suffix = "_tight_tplus1"

    tplus1 = bool(inv.get("tplus1", True))

    out_path = (
        f"outputs/trades/trades_pnl_{policy_name}_inventory{inv0}_cap{cap}"
        f"{inventory_suffix}_{tag}.csv"
    )

    skipped_path = (
        f"outputs/trades/skipped_{policy_name}_inventory{inv0}_cap{cap}"
        f"{inventory_suffix}_{tag}.csv"
    )

    run([
        "python", "scripts/filter_inventory_aware_overlay.py",
        "--trades", policy_trades_path,
        "--output", out_path,
        "--skipped-output", skipped_path,
        "--initial-position-per-symbol", str(inv0),
        "--max-position-per-symbol", str(cap),
        "--sell-floor-position", str(sell_floor),
        "--buy-block-position", str(buy_block),
        "--tplus1", "1" if tplus1 else "0",
    ])

    return out_path, skipped_path


def run_portfolio(exp_cfg, inventory_trades_path):
    tag = exp_cfg["experiment"]["tag"]
    policy_name = exp_cfg.get("policy", {}).get("name", "all")
    inv = exp_cfg.get("inventory", {})

    inv0 = int(inv.get("initial_position_per_symbol", 5000))
    cap = int(inv.get("max_position_per_symbol", 15000))
    has_sell_buy_overlay = ("sell_floor_position" in inv) or ("buy_block_position" in inv)

    if has_sell_buy_overlay:
        sell_floor = int(inv.get("sell_floor_position", 3000))
        buy_block = int(inv.get("buy_block_position", 9000))
        inventory_suffix = f"_sellfloor{sell_floor}_buyblock{buy_block}_tplus1"
    else:
        sell_floor = 0
        buy_block = cap
        inventory_suffix = "_tight_tplus1"

    allow_short = bool(inv.get("allow_short", False))

    model_name = (
        f"{tag}_{policy_name}_inventory{inv0}_cap{cap}"
        f"{inventory_suffix}"
    )

    prefix = f"outputs/portfolio/portfolio_replay_{model_name}"

    run([
        "python", "scripts/run_portfolio_replay.py",
        "--trades", inventory_trades_path,
        "--model", model_name,
        "--out-prefix", prefix,
        "--capital", "0",
        "--initial-position-per-symbol", str(inv0),
        "--allow-short", "1" if allow_short else "0",
        "--record-every", "10000",
    ])

    return prefix


def print_summary(prefix):
    summary_path = prefix + "_summary.csv"
    daily_path = prefix + "_daily.csv"

    if not os.path.exists(summary_path):
        print("summary not found:", summary_path)
        return

    summary = pd.read_csv(summary_path)

    pd.set_option("display.max_columns", 120)
    pd.set_option("display.width", 260)

    cols = [
        "model",
        "num_trades",
        "num_securities",
        "total_pnl",
        "total_turnover",
        "pnl_bps_on_turnover",
        "max_gross_exposure",
        "return_on_max_gross_exposure",
        "max_abs_net_exposure",
        "max_drawdown",
        "num_long_symbols_end",
        "num_short_symbols_end",
        "num_short_events",
        "num_short_violations_if_no_short",
    ]
    cols = [c for c in cols if c in summary.columns]

    print("\n===== final experiment summary =====")
    print(summary[cols].to_string(index=False))

    if os.path.exists(daily_path):
        daily = pd.read_csv(daily_path)
        daily_cols = [
            "model",
            "date",
            "num_trades",
            "turnover",
            "daily_pnl",
            "daily_return_on_turnover_bps",
            "max_gross_exposure",
            "max_drawdown",
            "num_long_symbols",
            "num_short_symbols",
        ]
        daily_cols = [c for c in daily_cols if c in daily.columns]

        print("\n===== daily =====")
        print(daily[daily_cols].to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", required=True)
    parser.add_argument("--skip-fill", action="store_true")
    parser.add_argument("--skip-enrich", action="store_true")
    parser.add_argument("--skip-pnl", action="store_true")
    parser.add_argument("--skip-final", action="store_true")
    parser.add_argument("--max-quotes", default="")
    args = parser.parse_args()

    ensure_dirs()

    with open(args.exp, "r") as f:
        exp_cfg = yaml.safe_load(f)

    runtime_config, cfg, fill_prefix = make_runtime_config(exp_cfg)

    tag = exp_cfg["experiment"]["tag"]
    fill_path = cfg["output"]["fill_path"]
    enriched_path = cfg["output"]["enriched_fill_path"]
    pnl_path = cfg["output"]["pnl_path"]

    print("===== Backtesting experiment =====")
    print("exp:", args.exp)
    print("tag:", tag)
    print("runtime config:", runtime_config)
    print("quote:", cfg["input"]["quote_decision_path"])
    print("signal:", cfg["signal"]["col"])
    print("fill_path:", fill_path)
    print("enriched_path:", enriched_path)
    print("pnl_path:", pnl_path)

    if not args.skip_fill:
        cmd = ["python", "scripts/run_fill_simulation.py", "--config", runtime_config]
        if args.max_quotes:
            cmd += ["--max-quotes", str(args.max_quotes)]
        run(cmd)
    else:
        print("[skip] fill simulation")

    if not args.skip_enrich:
        run([
            "python", "scripts/enrich_fills_with_factor_chunked.py",
            "--config", runtime_config,
            "--fills", fill_path,
            "--output", enriched_path,
            "--chunksize", "1000000",
        ])
    else:
        print("[skip] enrich")

    if not args.skip_pnl:
        run([
            "python", "scripts/run_attention_pnl_backtest.py",
            "--config", runtime_config,
            "--fills", enriched_path,
            "--output", pnl_path,
        ])
    else:
        print("[skip] pnl")

    if not args.skip_final:
        policy_trades = apply_policy(exp_cfg, pnl_path)
        inventory_trades, skipped = run_inventory_overlay(exp_cfg, policy_trades)
        prefix = run_portfolio(exp_cfg, inventory_trades)
        print_summary(prefix)
        print("\nskipped file:", skipped)
    else:
        print("[skip] final inventory-aware replay")


if __name__ == "__main__":
    main()
