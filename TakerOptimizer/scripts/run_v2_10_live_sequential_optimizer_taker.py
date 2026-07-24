import argparse
import subprocess
from pathlib import Path

import pandas as pd
import yaml


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)


def run_cmd(cmd, log_path):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    print("[RUN]", " ".join(cmd))
    print("[LOG]", log_path)

    with open(log_path, "w", encoding="utf-8") as f:
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)

    if p.returncode != 0:
        raise RuntimeError(f"command failed, see log: {log_path}")


def init_files(args):
    Path(args.daily_pnl_path).parent.mkdir(parents=True, exist_ok=True)
    if not Path(args.daily_pnl_path).exists():
        pd.DataFrame(columns=["date", "net_pnl"]).to_csv(args.daily_pnl_path, index=False)

    Path(args.position_state_path).parent.mkdir(parents=True, exist_ok=True)
    if not Path(args.position_state_path).exists():
        pd.DataFrame(columns=[
            "date", "securityid", "actual_qty", "last_price", "market_value"
        ]).to_csv(args.position_state_path, index=False)


def make_optimizer_config(base_cfg_path, date, args):
    cfg = load_yaml(base_cfg_path)
    date = int(date)

    cfg.setdefault("data", {})
    cfg["data"]["start_date"] = date
    cfg["data"]["end_date"] = date
    cfg["data"]["daily_pnl_path"] = args.daily_pnl_path
    cfg["data"]["initial_position_path"] = args.position_state_path

    # Use daily-sliced alpha / pricing if available.
    # This avoids scanning 30-day full CSVs for every sequential day.
    daily_alpha = Path(f"TakerOptimizer/cache/by_day/{args.model_name}/alpha/alpha_{args.model_name}_{date}.csv")
    daily_pricing = Path(f"TakerOptimizer/cache/by_day/{args.model_name}/pricing/pricing_{args.model_name}_{date}.csv")

    if daily_alpha.exists():
        cfg["data"]["global_alpha_path"] = str(daily_alpha)
        cfg["data"]["local_alpha_path"] = str(daily_alpha)
        print("[DAILY ALPHA]", daily_alpha)
    else:
        print("[WARN] daily alpha not found, using base config alpha:", daily_alpha)

    if daily_pricing.exists():
        cfg["data"]["pricing_path"] = str(daily_pricing)
        print("[DAILY PRICING]", daily_pricing)
    else:
        print("[WARN] daily pricing not found, using base config pricing:", daily_pricing)

    cfg["data"]["output_path"] = (
        f"TakerOptimizer/outputs/target_positions/by_day/"
        f"target_positions_v2_10_{args.model_name}_{date}.csv"
    )
    cfg["data"]["log_path"] = (
        f"TakerOptimizer/outputs/logs/by_day/"
        f"optimizer_v2_10_{args.model_name}_{date}.log"
    )

    if "project" in cfg:
        cfg["project"]["version"] = f"v2_10_live_sequential_{args.model_name}_{date}"

    out_cfg = f"TakerOptimizer/config/by_day/optimizer_v2_10_{args.model_name}_{date}.yaml"
    save_yaml(cfg, out_cfg)

    return out_cfg, cfg["data"]["output_path"]


def make_taker_config(base_cfg_path, date, target_position_path, args):
    cfg = load_yaml(base_cfg_path)
    date = int(date)

    cfg.setdefault("data", {})
    cfg["data"]["start_date"] = date
    cfg["data"]["end_date"] = date

    # 当前 TakerModel config 里确认读的是这个 key
    cfg["data"]["optimizer_output_path"] = target_position_path

    # Give TakerModel a one-day market_data_dir to avoid scanning the whole 90-day directory.
    daily_market_dir = Path(f"TakerModel/cache/market_by_day/{date}")
    daily_market_dir.mkdir(parents=True, exist_ok=True)

    src_market = Path(f"PricingModel/data/market_return_20241022_20250114_742_by_date/market_return_{date}_742.csv")
    dst_market = daily_market_dir / f"market_return_{date}_742.csv"

    if src_market.exists() and not dst_market.exists():
        try:
            dst_market.symlink_to(src_market.resolve())
        except FileExistsError:
            pass

    if "market_data_dir" in cfg["data"]:
        cfg["data"]["market_data_dir"] = str(daily_market_dir)
    else:
        cfg["data"]["market_data_dir"] = str(daily_market_dir)

    # 给 TakerModel v5 用；如果脚本不读，也不会影响
    cfg["data"]["initial_position_path"] = args.position_state_path
    cfg["data"]["current_position_path"] = args.position_state_path

    out_root = Path(f"TakerModel/outputs/sequential/{args.model_name}_{date}")
    pos_dir = out_root / "positions"
    metric_dir = out_root / "metrics"

    cfg["output"] = {
        "position_output_path": str(pos_dir / f"taker_positions_{args.model_name}_{date}.csv"),
        "minute_metrics_path": str(metric_dir / f"taker_minute_metrics_{args.model_name}_{date}.csv"),
        "daily_metrics_path": str(metric_dir / f"taker_daily_metrics_{args.model_name}_{date}.csv"),
        "trade_reason_path": str(metric_dir / f"taker_trade_reason_{args.model_name}_{date}.csv"),
        "summary_path": str(metric_dir / f"taker_summary_{args.model_name}_{date}.csv"),
    }

    cfg.setdefault("filters", {})
    cfg["filters"]["exit_when_target_zero"] = True
    cfg["filters"]["exit_when_direction_flip"] = True
    cfg["filters"]["reduce_when_target_smaller"] = False

    cfg.setdefault("sequential", {})
    cfg["sequential"]["enabled"] = True
    cfg["sequential"]["date"] = date
    cfg["sequential"]["target_position_path"] = target_position_path
    cfg["sequential"]["initial_position_path"] = args.position_state_path
    cfg["sequential"]["daily_pnl_output_path"] = str(metric_dir / f"daily_pnl_{date}.csv")
    cfg["sequential"]["eod_position_output_path"] = str(pos_dir / f"eod_positions_{date}.csv")

    out_cfg = f"TakerModel/config/by_day/taker_model_{args.model_name}_{date}.yaml"
    save_yaml(cfg, out_cfg)

    return out_cfg, str(out_root), cfg["output"]["daily_metrics_path"], cfg["output"]["position_output_path"]


def extract_daily_pnl(daily_metrics_path, summary_path, date):
    date = int(date)

    candidates = [
        Path(daily_metrics_path),
        Path(summary_path),
    ]

    for f in candidates:
        if not f.exists():
            continue

        df = pd.read_csv(f)

        # daily metrics table: date, net_pnl / total_net_pnl / daily_net_pnl
        if "date" in df.columns:
            for col in ["net_pnl", "total_net_pnl", "daily_net_pnl", "pnl"]:
                if col in df.columns:
                    tmp = df[df["date"].astype(int) == date]
                    if len(tmp) > 0:
                        val = pd.to_numeric(tmp[col], errors="coerce").dropna()
                        if len(val) > 0:
                            return float(val.iloc[-1])

        # summary table: metric,value
        if len(df.columns) >= 2:
            key_col, val_col = df.columns[0], df.columns[1]
            tmp = df.copy()
            tmp[key_col] = tmp[key_col].astype(str)
            hit = tmp[tmp[key_col].isin(["total_net_pnl", "net_pnl", "daily_net_pnl"])]
            if len(hit) > 0:
                val = pd.to_numeric(hit[val_col], errors="coerce").dropna()
                if len(val) > 0:
                    return float(val.iloc[-1])

        # one-row table
        for col in ["net_pnl", "total_net_pnl", "daily_net_pnl", "pnl"]:
            if col in df.columns and len(df) == 1:
                val = pd.to_numeric(df[col], errors="coerce").iloc[0]
                if pd.notna(val):
                    return float(val)

    raise RuntimeError(
        f"cannot extract daily pnl for {date}; checked {daily_metrics_path} and {summary_path}"
    )


def append_daily_pnl(path, date, net_pnl):
    path = Path(path)
    if path.exists() and path.stat().st_size > 0:
        df = pd.read_csv(path)
    else:
        df = pd.DataFrame(columns=["date", "net_pnl"])

    date = int(date)
    df = df[df["date"].astype(str) != str(date)]
    df = pd.concat([
        df,
        pd.DataFrame([{"date": date, "net_pnl": float(net_pnl)}])
    ], ignore_index=True)

    df["date"] = df["date"].astype(int)
    df = df.sort_values("date")
    df.to_csv(path, index=False)

    print("[APPEND DAILY PNL]", date, net_pnl, "->", path)


def update_position_state(position_output_path, state_path, date):
    position_output_path = Path(position_output_path)
    state_path = Path(state_path)

    if not position_output_path.exists():
        print("[WARN] no position output found:", position_output_path)
        return

    df = pd.read_csv(position_output_path)
    if len(df) == 0:
        print("[WARN] empty position output:", position_output_path)
        return

    symbol_col = "securityid" if "securityid" in df.columns else "SecurityID"

    qty_col = None
    for c in ["actual_qty", "current_qty", "position_qty", "qty", "effective_current_qty"]:
        if c in df.columns:
            qty_col = c
            break

    if qty_col is None:
        print("[WARN] cannot find qty column in position output. columns=", list(df.columns))
        return

    price_col = None
    for c in ["mid_price", "last_price", "price", "close", "fair_price"]:
        if c in df.columns:
            price_col = c
            break

    sort_cols = []
    for c in ["datetime", "minute", "date"]:
        if c in df.columns:
            sort_cols.append(c)

    if sort_cols:
        df = df.sort_values(sort_cols)

    eod = df.groupby(symbol_col, as_index=False).tail(1).copy()
    eod["date"] = int(date)
    eod["securityid"] = eod[symbol_col]
    eod["actual_qty"] = pd.to_numeric(eod[qty_col], errors="coerce").fillna(0)

    if price_col:
        eod["last_price"] = pd.to_numeric(eod[price_col], errors="coerce")
    else:
        eod["last_price"] = pd.NA

    eod["market_value"] = eod["actual_qty"] * pd.to_numeric(eod["last_price"], errors="coerce")

    out = eod[["date", "securityid", "actual_qty", "last_price", "market_value"]]
    out = out[out["actual_qty"].abs() > 0]

    state_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(state_path, index=False)

    print("[UPDATE POSITION STATE]", state_path, "rows=", len(out))


def prepare_positions_for_taker(target_path, date, model_name):
    """
    Convert TakerOptimizer target position output to the column format required by
    run_taker_model_v5_actual_qty_control_from_positions.py.

    Main adapters:
      date/datetime       -> execution_date/execution_datetime
      bid1/ask1           -> bid_price/ask_price
      bid1_volume/...     -> bid_volume/ask_volume
      effective_target_qty -> target_qty
    """
    src = Path(target_path)
    out_dir = Path("TakerOptimizer/outputs/target_positions/by_day/for_taker")
    out_dir.mkdir(parents=True, exist_ok=True)

    dst = out_dir / f"target_positions_v2_10_{model_name}_{int(date)}_for_taker.csv"

    df = pd.read_csv(src)

    def copy_if_missing(dst_col, candidates, default=None):
        if dst_col in df.columns:
            return
        for c in candidates:
            if c in df.columns:
                df[dst_col] = df[c]
                return
        if default is not None:
            df[dst_col] = default

    # ------------------------------------------------------------
    # required execution time columns
    # ------------------------------------------------------------
    if "execution_date" not in df.columns:
        if "date" not in df.columns:
            raise ValueError(f"{src} missing date, cannot create execution_date")
        df["execution_date"] = df["date"].astype(int)

    if "execution_datetime" not in df.columns:
        if "datetime" in df.columns:
            df["execution_datetime"] = df["datetime"]
        elif "minute" in df.columns:
            df["execution_datetime"] = df["minute"]
        else:
            raise ValueError(f"{src} missing datetime/minute, cannot create execution_datetime")

    # ------------------------------------------------------------
    # market price / volume columns required by TakerModel v5
    # ------------------------------------------------------------
    copy_if_missing("bid_price", ["bid1", "bid", "BidPrice1", "bid_price_1"])
    copy_if_missing("ask_price", ["ask1", "ask", "AskPrice1", "ask_price_1"])

    copy_if_missing("bid_volume", ["bid1_volume", "bid_volume1", "BidVolume1", "bidvolume1", "bid_size"])
    copy_if_missing("ask_volume", ["ask1_volume", "ask_volume1", "AskVolume1", "askvolume1", "ask_size"])

    copy_if_missing("mid_price", ["mid_price", "price", "last_price", "mid", "fair_price"])

    if "mid_price" not in df.columns and "bid_price" in df.columns and "ask_price" in df.columns:
        bid = pd.to_numeric(df["bid_price"], errors="coerce")
        ask = pd.to_numeric(df["ask_price"], errors="coerce")
        df["mid_price"] = (bid + ask) / 2.0

    if "spread_bps" not in df.columns and "bid_price" in df.columns and "ask_price" in df.columns and "mid_price" in df.columns:
        bid = pd.to_numeric(df["bid_price"], errors="coerce")
        ask = pd.to_numeric(df["ask_price"], errors="coerce")
        mid = pd.to_numeric(df["mid_price"], errors="coerce").replace(0, pd.NA)
        df["spread_bps"] = (ask - bid) / mid * 10000.0

    # ------------------------------------------------------------
    # target / current position columns
    # ------------------------------------------------------------
    copy_if_missing("target_qty", [
        "effective_target_qty",
        "target_qty",
        "desired_qty",
        "target_position",
        "target_position_qty",
    ])

    copy_if_missing("target_weight", [
        "effective_target_weight",
        "target_weight",
        "desired_weight",
    ])

    copy_if_missing("current_qty", [
        "current_qty",
        "actual_qty",
        "position_qty",
        "effective_current_qty",
    ], default=0)

    copy_if_missing("current_weight", [
        "current_weight",
        "actual_weight",
    ], default=0.0)

    # Some scripts use executable delta.
    copy_if_missing("delta_qty", [
        "executable_delta_qty",
        "delta_qty",
        "delta_qty_raw",
    ], default=0)

    # ------------------------------------------------------------
    # sanity check
    # ------------------------------------------------------------
    required_now = [
        "execution_date",
        "execution_datetime",
        "securityid",
        "bid_price",
        "ask_price",
        "mid_price",
        "target_qty",
    ]

    missing_now = [c for c in required_now if c not in df.columns]
    if missing_now:
        raise ValueError(
            f"prepared positions still missing required columns: {missing_now}; "
            f"available={list(df.columns)}"
        )

    df.to_csv(dst, index=False)

    print("[PREPARE POSITIONS FOR TAKER]", src, "->", dst)
    print("[PREPARE POSITIONS COLUMNS]")
    for c in [
        "execution_date",
        "execution_datetime",
        "securityid",
        "bid_price",
        "ask_price",
        "mid_price",
        "target_qty",
        "target_weight",
        "current_qty",
        "current_weight",
        "bid_volume",
        "ask_volume",
        "spread_bps",
        "delta_qty",
    ]:
        if c in df.columns:
            print("  ", c)

    return str(dst)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--dates", nargs="+", required=True)
    ap.add_argument("--model-name", default="feature_transformer_h120")

    ap.add_argument(
        "--optimizer-script",
        default="TakerOptimizer/scripts/apply_taker_position_optimizer_v2_9_local_global_risk_overlay.py",
    )
    ap.add_argument(
        "--optimizer-base-config",
        required=True,
    )

    ap.add_argument("--taker-script", required=True)
    ap.add_argument("--taker-base-config", required=True)

    ap.add_argument(
        "--daily-pnl-path",
        default="TakerOptimizer/inputs/daily_strategy_pnl.csv",
    )
    ap.add_argument(
        "--position-state-path",
        default="TakerOptimizer/state/current_positions.csv",
    )

    args = ap.parse_args()
    init_files(args)

    for date in args.dates:
        print("\n" + "=" * 100)
        print("[DATE]", date)

        opt_cfg, target_path = make_optimizer_config(
            args.optimizer_base_config,
            date,
            args,
        )

        opt_log = f"TakerOptimizer/outputs/logs/by_day/run_optimizer_v2_10_{args.model_name}_{date}.nohup.log"
        run_cmd(["python", args.optimizer_script, "--config", opt_cfg], opt_log)

        if not Path(target_path).exists():
            raise FileNotFoundError(f"optimizer output not found: {target_path}")

        taker_cfg, taker_out_root, daily_metrics_path, position_output_path = make_taker_config(
            args.taker_base_config,
            date,
            target_path,
            args,
        )

        taker_log = f"TakerModel/outputs/logs/by_day/run_taker_model_{args.model_name}_{date}.nohup.log"
        run_cmd(["python", args.taker_script, "--config", taker_cfg], taker_log)

        cfg = load_yaml(taker_cfg)
        summary_path = cfg["output"]["summary_path"]

        net_pnl = extract_daily_pnl(daily_metrics_path, summary_path, date)
        append_daily_pnl(args.daily_pnl_path, date, net_pnl)

        update_position_state(position_output_path, args.position_state_path, date)

    print("\n[DONE] sequential optimizer + taker model")


if __name__ == "__main__":
    main()
