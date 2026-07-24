import argparse
import os
import sys
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from run_taker_model_v2_turnover_control import (
    load_yaml,
    ensure_dir,
    prepare_optimizer,
    load_market_first_snapshot,
    make_minute_metrics,
    make_daily_metrics,
    make_reason_metrics,
    make_summary,
)


def sign(x, eps=1e-12):
    if x > eps:
        return 1
    if x < -eps:
        return -1
    return 0


def run_exit_control(df, cfg):
    exe = cfg["execution"]
    filt = cfg["filters"]

    capital = float(exe.get("capital", 200000000))
    taker_fee_bps = float(exe.get("taker_fee_bps", 0.5))
    slippage_bps = float(exe.get("slippage_bps", 0.0))

    entry_rebalance_ratio = float(exe.get("entry_rebalance_ratio", 0.5))
    exit_rebalance_ratio = float(exe.get("exit_rebalance_ratio", 1.0))

    require_optimal_for_entry = bool(filt.get("require_optimal_for_entry", True))
    require_valid_market = bool(filt.get("require_valid_market", True))

    entry_min_abs_delta_notional = float(filt.get("entry_min_abs_delta_notional", 0.0))
    entry_max_spread_bps = filt.get("entry_max_spread_bps", None)
    entry_min_abs_net_alpha_bps = filt.get("entry_min_abs_net_alpha_bps", None)

    exit_max_spread_bps = filt.get("exit_max_spread_bps", None)
    hold_min_abs_net_alpha_bps = filt.get("hold_min_abs_net_alpha_bps", None)

    exit_when_target_zero = bool(filt.get("exit_when_target_zero", True))
    exit_when_direction_flip = bool(filt.get("exit_when_direction_flip", True))
    reduce_when_target_smaller = bool(filt.get("reduce_when_target_smaller", True))

    df = df.copy()
    df = df.sort_values(["securityid", "execution_datetime"]).reset_index(drop=True)

    for c in ["bid_price", "ask_price", "mid_price", "label", "desired_target_weight"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["spread_bps_realized"] = (df["ask_price"] - df["bid_price"]) / df["mid_price"] * 10000.0

    rows = []
    current_weight = {}

    for row in df.itertuples(index=False):
        sid = int(row.securityid)
        prev_w = current_weight.get(sid, 0.0)

        desired_w = float(row.desired_target_weight) if pd.notna(row.desired_target_weight) else 0.0

        prev_s = sign(prev_w)
        desired_s = sign(desired_w)

        raw_delta_w = desired_w - prev_w
        raw_delta_notional = abs(raw_delta_w) * capital

        status = str(getattr(row, "optimizer_status", "unknown")).lower()
        valid_market_bool = bool(getattr(row, "valid_market_bool", True))

        spread_bps = getattr(row, "spread_bps_realized", np.nan)
        net_alpha = getattr(row, "net_alpha_bps", np.nan)

        invalid_market = require_valid_market and (not valid_market_bool)

        entry_reasons = []

        if require_optimal_for_entry and status != "optimal":
            entry_reasons.append("not_optimal")

        if invalid_market:
            entry_reasons.append("invalid_market")

        if raw_delta_notional < entry_min_abs_delta_notional:
            entry_reasons.append("small_delta")

        if entry_max_spread_bps is not None:
            if pd.isna(spread_bps) or spread_bps > float(entry_max_spread_bps):
                entry_reasons.append("wide_spread")

        if entry_min_abs_net_alpha_bps is not None:
            if pd.isna(net_alpha) or abs(float(net_alpha)) < float(entry_min_abs_net_alpha_bps):
                entry_reasons.append("weak_edge")

        entry_ok = len(entry_reasons) == 0

        exit_reasons = []

        if invalid_market:
            exit_reasons.append("invalid_market")

        if exit_max_spread_bps is not None:
            if pd.isna(spread_bps) or spread_bps > float(exit_max_spread_bps):
                exit_reasons.append("exit_wide_spread")

        exit_market_ok = len(exit_reasons) == 0

        target_zero = desired_s == 0
        direction_flip = prev_s != 0 and desired_s != 0 and prev_s != desired_s

        same_dir_reduce = (
            prev_s != 0
            and desired_s == prev_s
            and abs(desired_w) < abs(prev_w) - 1e-12
        )

        same_dir_increase = (
            desired_s != 0
            and (prev_s == 0 or desired_s == prev_s)
            and abs(desired_w) > abs(prev_w) + 1e-12
        )

        weak_hold_edge = False
        if hold_min_abs_net_alpha_bps is not None:
            weak_hold_edge = pd.isna(net_alpha) or abs(float(net_alpha)) < float(hold_min_abs_net_alpha_bps)

        executed_delta_w = 0.0
        decision = "hold"
        skip_reason = "hold"

        if prev_s == 0:
            if desired_s == 0:
                decision = "flat"
                skip_reason = "flat"
            else:
                if entry_ok:
                    executed_delta_w = entry_rebalance_ratio * raw_delta_w
                    decision = "enter"
                    skip_reason = "entry_pass"
                else:
                    decision = "skip_entry"
                    skip_reason = "|".join(entry_reasons)

        else:
            if direction_flip and exit_when_direction_flip:
                if exit_market_ok:
                    executed_delta_w = exit_rebalance_ratio * (-prev_w)
                    decision = "exit_direction_flip"
                    skip_reason = "exit_direction_flip"
                else:
                    decision = "hold_exit_blocked"
                    skip_reason = "|".join(exit_reasons)

            elif target_zero and exit_when_target_zero:
                if exit_market_ok:
                    executed_delta_w = exit_rebalance_ratio * (-prev_w)
                    decision = "exit_target_zero"
                    skip_reason = "exit_target_zero"
                else:
                    decision = "hold_exit_blocked"
                    skip_reason = "|".join(exit_reasons)

            elif weak_hold_edge:
                if exit_market_ok:
                    executed_delta_w = exit_rebalance_ratio * (-prev_w)
                    decision = "exit_weak_edge"
                    skip_reason = "exit_weak_edge"
                else:
                    decision = "hold_weak_edge_exit_blocked"
                    skip_reason = "|".join(exit_reasons)

            elif same_dir_reduce and reduce_when_target_smaller:
                if exit_market_ok:
                    executed_delta_w = exit_rebalance_ratio * raw_delta_w
                    decision = "reduce_to_target"
                    skip_reason = "reduce_to_target"
                else:
                    decision = "hold_reduce_blocked"
                    skip_reason = "|".join(exit_reasons)

            elif same_dir_increase:
                if entry_ok:
                    executed_delta_w = entry_rebalance_ratio * raw_delta_w
                    decision = "increase"
                    skip_reason = "entry_pass"
                else:
                    decision = "skip_increase"
                    skip_reason = "|".join(entry_reasons)

            else:
                decision = "hold_position"
                skip_reason = "hold_position"

        new_w = prev_w + executed_delta_w

        if abs(new_w) < 1e-12:
            new_w = 0.0

        trade_side = "NONE"
        if executed_delta_w > 0:
            trade_side = "BUY"
        elif executed_delta_w < 0:
            trade_side = "SELL"

        trade_notional = abs(executed_delta_w) * capital
        gross_exposure = abs(new_w) * capital
        net_exposure = new_w * capital

        effective_bid = row.bid_price if pd.notna(row.bid_price) and row.bid_price > 0 else row.mid_price
        effective_ask = row.ask_price if pd.notna(row.ask_price) and row.ask_price > 0 else row.mid_price

        if trade_side == "BUY":
            exec_price = effective_ask
            spread_cost_ret = max(effective_ask / row.mid_price - 1.0, 0.0) if row.mid_price > 0 else 0.0
        elif trade_side == "SELL":
            exec_price = effective_bid
            spread_cost_ret = max((row.mid_price - effective_bid) / row.mid_price, 0.0) if row.mid_price > 0 else 0.0
        else:
            exec_price = np.nan
            spread_cost_ret = 0.0

        gross_pnl = capital * new_w * row.label if pd.notna(row.label) else np.nan
        spread_cost = trade_notional * spread_cost_ret
        fee = trade_notional * taker_fee_bps / 10000.0
        slippage = trade_notional * slippage_bps / 10000.0
        cost = spread_cost + fee + slippage
        net_pnl = gross_pnl - cost if pd.notna(gross_pnl) else np.nan

        current_weight[sid] = new_w

        base = row._asdict()
        base.update({
            "prev_executed_weight": prev_w,
            "desired_target_weight": desired_w,
            "raw_delta_weight": raw_delta_w,
            "raw_delta_notional": raw_delta_notional,
            "executed_delta_weight": executed_delta_w,
            "executed_weight": new_w,
            "decision": decision,
            "skip_reason": skip_reason,
            "trade_side": trade_side,
            "trade_notional": trade_notional,
            "gross_exposure": gross_exposure,
            "net_exposure": net_exposure,
            "exec_price": exec_price,
            "spread_cost_ret": spread_cost_ret,
            "gross_pnl": gross_pnl,
            "spread_cost": spread_cost,
            "fee": fee,
            "slippage": slippage,
            "cost": cost,
            "net_pnl": net_pnl,
        })

        rows.append(base)

    out = pd.DataFrame(rows)

    out["gross_pnl_bps_on_turnover"] = out["gross_pnl"] / out["trade_notional"].replace(0, np.nan) * 10000.0
    out["net_pnl_bps_on_turnover"] = out["net_pnl"] / out["trade_notional"].replace(0, np.nan) * 10000.0
    out["net_pnl_bps_on_gross_exposure"] = out["net_pnl"] / out["gross_exposure"].replace(0, np.nan) * 10000.0

    keep = (
        (out["executed_weight"].abs() > 0)
        | (out["trade_notional"] > 0)
        | (out["net_pnl"].abs() > 0)
    )

    return out[keep].copy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    optimizer_path = cfg["data"]["optimizer_output_path"]
    market_path = cfg["data"]["market_data_path"]

    print("loading optimizer:", optimizer_path)
    opt = prepare_optimizer(optimizer_path, cfg)

    print("optimizer rows:", len(opt))
    print("optimizer symbols:", opt["securityid"].nunique())
    print("execution minute range:", opt["execution_minute"].min(), opt["execution_minute"].max())

    from db_market_loader import load_market_first_snapshot_from_db

    label_col = cfg["columns"]["label_col"]
    label_horizon = int(str(label_col).replace("label_", ""))

    print("loading market first snapshots from ClickHouse")
    mkt = load_market_first_snapshot_from_db(
        opt,
        label_horizon=label_horizon,
        db_name="500ms",
        symbol_chunk_size=100,
    )

    print("market rows:", len(mkt))
    print("market symbols:", mkt["securityid"].nunique() if not mkt.empty else 0)

    print("normalizing merge key dtypes")

    opt["securityid"] = pd.to_numeric(opt["securityid"], errors="coerce").astype("Int64")
    mkt["securityid"] = pd.to_numeric(mkt["securityid"], errors="coerce").astype("Int64")

    opt["execution_minute"] = pd.to_datetime(opt["execution_minute"], errors="coerce")
    mkt["execution_minute"] = pd.to_datetime(mkt["execution_minute"], errors="coerce")

    print("opt execution_minute dtype:", opt["execution_minute"].dtype)
    print("mkt execution_minute dtype:", mkt["execution_minute"].dtype)
    print("opt securityid dtype:", opt["securityid"].dtype)
    print("mkt securityid dtype:", mkt["securityid"].dtype)

    if mkt.empty:
        raise ValueError("market data loaded from DB is empty. Check DB table/date/securityid query.")

    print("merging")
    merged = opt.merge(mkt, on=["securityid", "execution_minute"], how="left")

    missing_mid = merged["mid_price"].isna().mean()
    missing_label = merged["label"].isna().mean()

    print("merged shape:", merged.shape)
    print("missing mid rate:", missing_mid)
    print("missing label rate:", missing_label)

    max_missing = float(cfg.get("runtime", {}).get("max_missing_rate", 0.2))
    if missing_mid > max_missing or missing_label > max_missing:
        raise ValueError(f"merge failed: missing_mid={missing_mid}, missing_label={missing_label}")

    print("running exit-control taker model")
    result = run_exit_control(merged, cfg)

    minute_metrics = make_minute_metrics(result)
    daily_metrics = make_daily_metrics(result)
    reason_metrics = make_reason_metrics(result)
    summary = make_summary(result, minute_metrics, cfg)

    paths = cfg["output"]
    for p in paths.values():
        ensure_dir(p)

    result.to_csv(paths["position_output_path"], index=False)
    minute_metrics.to_csv(paths["minute_metrics_path"], index=False)
    daily_metrics.to_csv(paths["daily_metrics_path"], index=False)
    reason_metrics.to_csv(paths["trade_reason_path"], index=False)
    summary.to_csv(paths["summary_path"], index=False)

    print()
    print("saved position output:", paths["position_output_path"])
    print("saved minute metrics:", paths["minute_metrics_path"])
    print("saved daily metrics:", paths["daily_metrics_path"])
    print("saved trade reason:", paths["trade_reason_path"])
    print("saved summary:", paths["summary_path"])

    print()
    print("===== summary =====")
    print(summary.T)

    print()
    print("===== trade reason =====")
    print(reason_metrics.to_string(index=False))


if __name__ == "__main__":
    main()
