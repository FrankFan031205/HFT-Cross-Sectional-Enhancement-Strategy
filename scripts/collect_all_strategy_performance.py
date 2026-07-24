# -*- coding: utf-8 -*-
from pathlib import Path
import re
import pandas as pd
import numpy as np

ROOT = Path("/mnt/data1/fwz/HFT_010-dev_fwz_outputs")
OUT_DIR = ROOT / "final_baselines"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TAKER_ROOT = ROOT / "TakerModel"
ALPHA_ROOT = ROOT / "AlphaValidation"

BENCH_DAILY = ROOT / "TakerModel/zzy_zz2000_h5_res_benchmark_compare/tradable_ew_benchmark_daily.csv"
CAPITAL = 200_000_000.0


def parse_summary_csv(path: Path) -> dict:
    df = pd.read_csv(path, low_memory=False)
    if "value" not in df.columns:
        return {}

    key_col = [c for c in df.columns if c != "value"][0]
    s = df.set_index(key_col)["value"]

    def get(k, default=np.nan):
        try:
            return float(s.get(k, default))
        except Exception:
            return default

    tag = path.name
    tag = tag.replace("v5_actual_qty_summary_", "").replace(".csv", "")

    out = {
        "case": tag,
        "summary_path": str(path),
        "start_date": get("start_date"),
        "end_date": get("end_date"),
        "num_days": get("num_days"),
        "num_minutes": get("num_minutes"),
        "return_on_capital": get("return_on_capital"),
        "total_net_pnl": get("total_net_pnl"),
        "max_drawdown": get("max_drawdown"),
        "total_turnover": get("total_turnover"),
        "turnover_to_capital": get("turnover_to_capital"),
        "total_fee": get("total_fee"),
        "total_spread_cost_est": get("total_spread_cost_est"),
        "avg_gross_weight": get("avg_gross_weight"),
        "avg_net_weight": get("avg_net_weight"),
        "avg_n_hold": get("avg_n_hold"),
        "num_trade_events": get("num_trade_events"),
        "num_buy_trades": get("num_buy_trades"),
        "num_sell_trades": get("num_sell_trades"),
        "total_blocked_small": get("total_blocked_small"),
        "total_blocked_spread": get("total_blocked_spread"),
        "total_blocked_market_regime": get("total_blocked_market_regime"),
        "total_market_risk_off_minutes": get("total_market_risk_off_minutes"),
        "gross_cap": get("gross_cap"),
        "min_trade_notional": get("min_trade_notional"),
        "market_regime_target_scale": get("market_regime_target_scale"),
        "market_regime_ret_stop": get("market_regime_ret_stop"),
        "market_regime_up_ratio_stop": get("market_regime_up_ratio_stop"),
    }

    out["total_cost"] = out["total_fee"] + out["total_spread_cost_est"]
    out["gross_alpha_pnl_before_cost"] = out["total_net_pnl"] + out["total_cost"]

    daily_name = path.name.replace("summary", "daily")
    daily_path = path.parent / daily_name
    if daily_path.exists():
        out["daily_path"] = str(daily_path)
    else:
        out["daily_path"] = ""

    return out


def add_benchmark_excess(perf: pd.DataFrame) -> pd.DataFrame:
    if perf.empty or not BENCH_DAILY.exists():
        return perf

    bench = pd.read_csv(BENCH_DAILY, low_memory=False)
    if "execution_date" not in bench.columns or "ew_universe_return" not in bench.columns:
        return perf

    rows = []
    for _, r in perf.iterrows():
        row = r.to_dict()
        daily_path = row.get("daily_path", "")
        if not daily_path or not Path(daily_path).exists():
            rows.append(row)
            continue

        d = pd.read_csv(daily_path, low_memory=False)
        if "execution_date" not in d.columns or "daily_net_pnl" not in d.columns:
            rows.append(row)
            continue

        x = d.merge(bench[["execution_date", "ew_universe_return"]], on="execution_date", how="left")
        x["strategy_return"] = x["daily_net_pnl"] / CAPITAL

        if "net_weight_mean" in x.columns:
            x["scaled_ew_return"] = x["ew_universe_return"] * x["net_weight_mean"]
        else:
            x["scaled_ew_return"] = x["ew_universe_return"] * row.get("avg_net_weight", np.nan)

        strat_ret = (1.0 + x["strategy_return"]).prod() - 1.0
        full_ew_ret = (1.0 + x["ew_universe_return"]).prod() - 1.0
        scaled_ew_ret = (1.0 + x["scaled_ew_return"]).prod() - 1.0

        row["full_ew_return"] = full_ew_ret
        row["scaled_ew_return"] = scaled_ew_ret
        row["excess_vs_full_ew"] = strat_ret - full_ew_ret
        row["excess_vs_scaled_ew"] = strat_ret - scaled_ew_ret
        row["excess_pnl_vs_full_ew"] = (strat_ret - full_ew_ret) * CAPITAL
        row["excess_pnl_vs_scaled_ew"] = (strat_ret - scaled_ew_ret) * CAPITAL

        rows.append(row)

    return pd.DataFrame(rows)


def collect_taker_performance():
    files = sorted(TAKER_ROOT.rglob("v5_actual_qty_summary_*.csv"))
    rows = []
    for p in files:
        try:
            row = parse_summary_csv(p)
            if row:
                rows.append(row)
        except Exception as e:
            print("[WARN] failed summary:", p, e)

    perf = pd.DataFrame(rows)
    if perf.empty:
        print("[WARN] no TakerModel summary files found")
        return perf

    perf = add_benchmark_excess(perf)

    # 排序：收益高优先，其次回撤小
    perf = perf.sort_values(["return_on_capital", "max_drawdown"], ascending=[False, False])

    cols_first = [
        "case",
        "return_on_capital",
        "total_net_pnl",
        "excess_vs_full_ew",
        "excess_pnl_vs_full_ew",
        "excess_vs_scaled_ew",
        "turnover_to_capital",
        "total_cost",
        "total_fee",
        "total_spread_cost_est",
        "max_drawdown",
        "avg_gross_weight",
        "avg_net_weight",
        "avg_n_hold",
        "num_trade_events",
        "total_blocked_market_regime",
        "total_market_risk_off_minutes",
        "summary_path",
    ]
    cols = [c for c in cols_first if c in perf.columns] + [c for c in perf.columns if c not in cols_first]
    perf = perf[cols]

    out = OUT_DIR / "all_taker_strategy_performance.csv"
    perf.to_csv(out, index=False)

    print("\n===== TakerModel Strategy Performance =====")
    display_cols = [
        "case", "return_on_capital", "total_net_pnl",
        "excess_vs_full_ew", "excess_pnl_vs_full_ew",
        "turnover_to_capital", "total_cost", "max_drawdown",
        "avg_gross_weight", "avg_n_hold", "num_trade_events"
    ]
    display_cols = [c for c in display_cols if c in perf.columns]
    print(perf[display_cols].head(30).to_string(index=False))

    print("\n[saved]", out)
    return perf


def parse_alpha_summary(path: Path) -> dict:
    df = pd.read_csv(path, low_memory=False)
    if df.empty:
        return {}

    row = df.iloc[0].to_dict()
    case = path.parent.name

    out = {
        "case": case,
        "alpha_summary_path": str(path),
    }

    for k, v in row.items():
        try:
            out[k] = float(v)
        except Exception:
            out[k] = v

    return out


def collect_alpha_validation():
    files = sorted(ALPHA_ROOT.rglob("alpha_validation_summary.csv"))
    rows = []
    for p in files:
        try:
            row = parse_alpha_summary(p)
            if row:
                rows.append(row)
        except Exception as e:
            print("[WARN] failed alpha:", p, e)

    alpha = pd.DataFrame(rows)
    if alpha.empty:
        print("[WARN] no AlphaValidation summary files found")
        return alpha

    sort_cols = []
    ascending = []
    for c in ["selected_excess_per_gross_bps", "selected_excess_per_gross_tstat", "signal_rank_ic_mean"]:
        if c in alpha.columns:
            sort_cols.append(c)
            ascending.append(False)

    if sort_cols:
        alpha = alpha.sort_values(sort_cols, ascending=ascending)

    out = OUT_DIR / "all_alpha_validation_performance.csv"
    alpha.to_csv(out, index=False)

    print("\n===== Alpha Validation Performance =====")
    display_cols = [
        "case",
        "signal_rank_ic_mean",
        "top_minus_bottom_bps",
        "selected_excess_per_gross_bps",
        "selected_excess_per_gross_tstat",
        "avg_selected_net_weight",
        "avg_n_hold",
        "selected_weighted_target_bps",
        "selected_ew_target_bps",
    ]
    display_cols = [c for c in display_cols if c in alpha.columns]
    print(alpha[display_cols].head(30).to_string(index=False))

    print("\n[saved]", out)
    return alpha


def main():
    print("[scan]", TAKER_ROOT)
    perf = collect_taker_performance()

    print("\n[scan]", ALPHA_ROOT)
    alpha = collect_alpha_validation()

    # 保存一个最简 markdown，方便贴给 mentor
    md_path = OUT_DIR / "all_strategy_performance_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Strategy Performance Summary\n\n")

        if not perf.empty:
            f.write("## TakerModel Backtest\n\n")
            cols = [
                "case", "return_on_capital", "total_net_pnl",
                "excess_vs_full_ew", "turnover_to_capital",
                "max_drawdown", "avg_gross_weight", "avg_n_hold"
            ]
            cols = [c for c in cols if c in perf.columns]
            f.write(perf[cols].head(30).to_markdown(index=False))
            f.write("\n\n")

        if not alpha.empty:
            f.write("## Alpha Validation\n\n")
            cols = [
                "case", "signal_rank_ic_mean", "top_minus_bottom_bps",
                "selected_excess_per_gross_bps",
                "selected_excess_per_gross_tstat",
                "avg_selected_net_weight", "avg_n_hold"
            ]
            cols = [c for c in cols if c in alpha.columns]
            f.write(alpha[cols].head(30).to_markdown(index=False))
            f.write("\n")

    print("\n[saved markdown]", md_path)


if __name__ == "__main__":
    main()
