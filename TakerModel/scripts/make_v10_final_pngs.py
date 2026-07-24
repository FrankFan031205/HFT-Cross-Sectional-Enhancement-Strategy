# -*- coding: utf-8 -*-
from pathlib import Path
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE = Path("/mnt/data1/fwz/HFT_010-dev_fwz_outputs/final_report")
V10_ROOT = BASE / "v10_many_horizon_mix_csi2000_warmstart_noovernight"
RANK_PATH = BASE / "v10_all_horizon_mix_csi2000_warmstart_noovernight/all_mix_summary_ranked.csv"

OUT_DIR = BASE / "v10_final_png_csi2000_warmstart_noovernight"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CANDS = {
    "final_40h10_60h20": {
        "mix": "mix_406000",
        "label": "Final: 40% h10 + 60% h20",
    },
    "stable_70h20_30h30": {
        "mix": "mix_007030",
        "label": "Stable: 70% h20 + 30% h30",
    },
    "light_10h10_90h20": {
        "mix": "mix_109000",
        "label": "Light h10: 10% h10 + 90% h20",
    },
    "h20_only": {
        "mix": "h20",
        "label": "h20 only",
    },
    "h10_only": {
        "mix": "h10",
        "label": "h10 only",
    },
}


def nav_path(mix):
    strategy = f"pure_cs_v10_{mix}_csi2000_warmstart_noovernight"
    return V10_ROOT / strategy / f"{strategy}_nav_curve_benchmark_warmstart_noovernight.csv"


def load_nav(mix):
    p = nav_path(mix)
    if not p.exists():
        raise FileNotFoundError(p)

    df = pd.read_csv(p)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    df["actual_nav"] = (1.0 + df["actual_ret"].fillna(0.0)).cumprod()
    df["benchmark_nav"] = (1.0 + df["benchmark_ret"].fillna(0.0)).cumprod()
    df["excess_nav"] = df["actual_nav"] / df["benchmark_nav"] - 1.0

    wealth = 1.0 + df["excess_nav"]
    df["excess_drawdown"] = wealth / wealth.cummax() - 1.0

    return df


def compound(x):
    x = pd.Series(x).fillna(0.0)
    return float((1.0 + x).prod() - 1.0)


navs = {}
for key, meta in CANDS.items():
    try:
        navs[key] = load_nav(meta["mix"])
        print("[loaded]", key, nav_path(meta["mix"]))
    except Exception as e:
        print("[skip]", key, e)


# ============================================================
# 01. Candidate excess NAV
# ============================================================

plt.figure(figsize=(13, 7))
for key, df in navs.items():
    plt.plot(df["datetime"], df["excess_nav"] * 100.0, label=CANDS[key]["label"], linewidth=1.8)

plt.axhline(0.0, linewidth=1.0)
plt.title("Benchmark-relative Excess Return Curve")
plt.xlabel("Time")
plt.ylabel("Excess return (%)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
out = OUT_DIR / "01_excess_nav_candidates.png"
plt.savefig(out, dpi=180)
plt.close()
print("[saved]", out)


# ============================================================
# 02. Final actual vs benchmark NAV
# ============================================================

final_key = "final_40h10_60h20"
df = navs[final_key]

plt.figure(figsize=(13, 7))
plt.plot(df["datetime"], df["actual_nav"], label="Strategy NAV", linewidth=1.8)
plt.plot(df["datetime"], df["benchmark_nav"], label="Benchmark NAV", linewidth=1.8)
plt.title("Final Strategy vs Benchmark NAV")
plt.xlabel("Time")
plt.ylabel("NAV")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
out = OUT_DIR / "02_actual_vs_benchmark_final.png"
plt.savefig(out, dpi=180)
plt.close()
print("[saved]", out)


# ============================================================
# 03. Daily alpha
# ============================================================

daily_rows = []

for key, df in navs.items():
    daily = (
        df.groupby("date", as_index=False)
          .agg(
              actual_day=("actual_ret", compound),
              benchmark_day=("benchmark_ret", compound),
          )
    )
    daily["alpha_day"] = daily["actual_day"] - daily["benchmark_day"]
    daily["version"] = CANDS[key]["label"]
    daily_rows.append(daily)

daily_all = pd.concat(daily_rows, ignore_index=True)
pivot = daily_all.pivot(index="date", columns="version", values="alpha_day").sort_index()

plt.figure(figsize=(14, 7))
for col in pivot.columns:
    plt.plot(pivot.index.astype(str), pivot[col] * 100.0, marker="o", linewidth=1.5, label=col)

plt.axhline(0.0, linewidth=1.0)
plt.title("Daily Benchmark-relative Alpha")
plt.xlabel("Date")
plt.ylabel("Daily alpha (%)")
plt.xticks(rotation=45, ha="right")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
out = OUT_DIR / "03_daily_alpha_candidates.png"
plt.savefig(out, dpi=180)
plt.close()
print("[saved]", out)


# ============================================================
# 04. Excess drawdown
# ============================================================

plt.figure(figsize=(13, 7))
for key, df in navs.items():
    plt.plot(df["datetime"], df["excess_drawdown"] * 100.0, label=CANDS[key]["label"], linewidth=1.6)

plt.axhline(0.0, linewidth=1.0)
plt.title("Benchmark-relative Drawdown")
plt.xlabel("Time")
plt.ylabel("Excess drawdown (%)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
out = OUT_DIR / "04_excess_drawdown_candidates.png"
plt.savefig(out, dpi=180)
plt.close()
print("[saved]", out)


# ============================================================
# 05. Top mix ranking table
# ============================================================

rank = pd.read_csv(RANK_PATH)

show_cols = [
    "mix_name",
    "source_group",
    "alpha_return",
    "daily_excess_sharpe",
    "turnover_weight",
    "total_cost",
    "opt_fallback_count",
]

rank = rank[show_cols].head(15).copy()
rank["alpha_return"] = rank["alpha_return"].map(lambda x: f"{100*x:.3f}%")
rank["daily_excess_sharpe"] = rank["daily_excess_sharpe"].map(lambda x: f"{x:.2f}")
rank["turnover_weight"] = rank["turnover_weight"].map(lambda x: f"{x:.2f}x")
rank["total_cost"] = rank["total_cost"].map(lambda x: f"{x/10000:.1f}w")
rank["opt_fallback_count"] = rank["opt_fallback_count"].astype(int).astype(str)

rank = rank.rename(columns={
    "mix_name": "Mix",
    "source_group": "Group",
    "alpha_return": "Alpha",
    "daily_excess_sharpe": "Sharpe",
    "turnover_weight": "Turnover",
    "total_cost": "Cost",
    "opt_fallback_count": "Fallback",
})

fig = plt.figure(figsize=(14, 7))
ax = fig.add_subplot(111)
ax.axis("off")
ax.set_title("Top Horizon Mix Ranking", fontsize=16, pad=16)

tbl = ax.table(
    cellText=rank.values,
    colLabels=rank.columns,
    loc="center",
    cellLoc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1.0, 1.35)

plt.tight_layout()
out = OUT_DIR / "05_top_mix_ranking_table.png"
plt.savefig(out, dpi=180)
plt.close()
print("[saved]", out)


# ============================================================
# Export daily alpha table
# ============================================================

daily_out = OUT_DIR / "daily_alpha_candidates.csv"
pivot.to_csv(daily_out)
print("[saved]", daily_out)

print("\n===== PNG DONE =====")
print(OUT_DIR)
