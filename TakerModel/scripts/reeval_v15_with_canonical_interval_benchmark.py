# -*- coding: utf-8 -*-
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/mnt/data1/fwz/HFT_010-dev_fwz_outputs/final_report")
V15_ROOT = ROOT / "v15_corrected_two_clock_csi2000_warmstart_noovernight"

CANON_TAG = "pure_cs_v15a_reproduce_4060_old_engine_csi2000_warmstart_noovernight"
CANON_NAV = (
    ROOT
    / "v15a_reproduce_4060_old_engine_csi2000_warmstart_noovernight"
    / CANON_TAG
    / f"{CANON_TAG}_nav_curve_benchmark_warmstart_noovernight.csv"
)

OUT_RANK = V15_ROOT / "v15_corrected_sweep_summary_canonical_interval_benchmark_ranked.csv"


def compound(x):
    x = pd.Series(x).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return float((1.0 + x).prod() - 1.0)


def sharpe(x):
    x = pd.Series(x).replace([np.inf, -np.inf], np.nan).dropna()
    if len(x) <= 1 or x.std(ddof=1) == 0:
        return np.nan
    return float(x.mean() / x.std(ddof=1) * np.sqrt(252))


canon = pd.read_csv(CANON_NAV)
canon["datetime"] = pd.to_datetime(canon["datetime"])
canon = canon.sort_values("datetime").drop_duplicates("datetime", keep="last")
canon["date"] = canon["datetime"].dt.strftime("%Y%m%d").astype(int)

canon["canonical_benchmark_nav"] = (1.0 + canon["benchmark_ret"].fillna(0.0)).cumprod()

print("[canonical]", CANON_NAV)
print("canonical rows:", len(canon))
print("canonical full benchmark:", compound(canon["benchmark_ret"]))

rows = []

for run_dir in sorted(V15_ROOT.glob("pure_cs_v15_*_csi2000")):
    tag = run_dir.name
    nav_path = run_dir / f"{tag}_nav_curve.csv"
    sum_path = run_dir / f"{tag}_summary.csv"

    if not nav_path.exists():
        print("[skip missing nav]", nav_path)
        continue

    nav = pd.read_csv(nav_path)
    nav["datetime"] = pd.to_datetime(nav["datetime"])
    nav = nav.sort_values("datetime").drop_duplicates("datetime", keep="last")
    nav["date"] = nav["datetime"].dt.strftime("%Y%m%d").astype(int)

    if "actual_ret" not in nav.columns:
        print("[skip no actual_ret]", nav_path)
        continue

    m = nav.merge(
        canon[["datetime", "canonical_benchmark_nav"]],
        on="datetime",
        how="left",
    )

    hit = m["canonical_benchmark_nav"].notna().mean()
    if hit < 0.99:
        print(f"[WARN] low canonical nav merge hit {tag}: {hit:.2%}")

    m["canonical_benchmark_nav"] = m["canonical_benchmark_nav"].ffill()

    # 关键修正：benchmark 用 execution timestamp 之间的区间收益
    m["canonical_benchmark_step"] = (
        m.groupby("date")["canonical_benchmark_nav"]
         .transform(lambda s: s / s.shift(1) - 1.0)
    )
    m["canonical_benchmark_step"] = m["canonical_benchmark_step"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    actual_return = compound(m["actual_ret"])
    benchmark_return = compound(m["canonical_benchmark_step"])
    alpha_return = actual_return - benchmark_return

    daily = (
        m.groupby("date", as_index=False)
         .agg(
             actual_day=("actual_ret", compound),
             benchmark_day=("canonical_benchmark_step", compound),
             turnover=("turnover_weight", "sum") if "turnover_weight" in m.columns else ("actual_ret", "size"),
             cost=("total_cost", "sum") if "total_cost" in m.columns else ("actual_ret", "size"),
         )
    )
    daily["alpha_day"] = daily["actual_day"] - daily["benchmark_day"]

    daily_excess_sharpe = sharpe(daily["alpha_day"])

    turnover_weight = float(m["turnover_weight"].sum()) if "turnover_weight" in m.columns else np.nan
    total_cost = float(m["total_cost"].sum()) if "total_cost" in m.columns else np.nan
    avg_gross = float(m["gross_prev_to_equity"].mean()) if "gross_prev_to_equity" in m.columns else np.nan

    fallback = np.nan
    param_name = tag.replace("pure_cs_v15_", "").replace("_csi2000", "")
    if sum_path.exists():
        s = pd.read_csv(sum_path).iloc[0]
        fallback = s.get("opt_fallback_count", np.nan)

    m["strategy_nav"] = (1.0 + m["actual_ret"].fillna(0.0)).cumprod()
    m["benchmark_nav_interval"] = (1.0 + m["canonical_benchmark_step"].fillna(0.0)).cumprod()
    m["strategy_cumret"] = m["strategy_nav"] - 1.0
    m["benchmark_cumret_interval"] = m["benchmark_nav_interval"] - 1.0
    m["alpha_cumret_interval"] = m["strategy_cumret"] - m["benchmark_cumret_interval"]

    out_nav = run_dir / f"{tag}_nav_curve_canonical_interval_benchmark.csv"
    out_daily = run_dir / f"{tag}_daily_canonical_interval_benchmark.csv"
    m.to_csv(out_nav, index=False)
    daily.to_csv(out_daily, index=False)

    rows.append({
        "version": tag,
        "param_name": param_name,
        "actual_return": actual_return,
        "benchmark_return": benchmark_return,
        "alpha_return": alpha_return,
        "daily_excess_sharpe": daily_excess_sharpe,
        "avg_gross_prev_to_equity": avg_gross,
        "turnover_weight": turnover_weight,
        "total_cost": total_cost,
        "opt_fallback_count": fallback,
        "benchmark_merge_hit": hit,
        "n_rows": len(m),
        "n_days": m["date"].nunique(),
        "start": str(m["datetime"].min()),
        "end": str(m["datetime"].max()),
        "nav_path": str(out_nav),
        "daily_path": str(out_daily),
    })

    print(
        f"[reeval interval] {tag} "
        f"rows={len(m)} actual={actual_return:.6f} "
        f"bench={benchmark_return:.6f} alpha={alpha_return:.6f} "
        f"sharpe={daily_excess_sharpe:.6f} turn={turnover_weight:.6f} "
        f"hit={hit:.2%}"
    )

if not rows:
    raise RuntimeError("no v15 nav curves found")

df = pd.DataFrame(rows)

for c in ["alpha_return", "daily_excess_sharpe", "turnover_weight", "opt_fallback_count"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df["score_tradeable"] = (
    100.0 * df["alpha_return"].fillna(-999)
    + 0.25 * df["daily_excess_sharpe"].fillna(-999)
    - 0.12 * df["turnover_weight"].fillna(999)
    - 0.05 * df["opt_fallback_count"].fillna(999)
)

df = df.sort_values("score_tradeable", ascending=False).reset_index(drop=True)
df.to_csv(OUT_RANK, index=False)

cols = [
    "param_name", "score_tradeable",
    "actual_return", "benchmark_return", "alpha_return",
    "daily_excess_sharpe", "turnover_weight", "total_cost",
    "opt_fallback_count", "benchmark_merge_hit", "n_rows", "n_days", "start", "end",
]

print("\n===== v15 canonical interval benchmark ranking =====")
print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.6f}"))

print("\n[saved]", OUT_RANK)
