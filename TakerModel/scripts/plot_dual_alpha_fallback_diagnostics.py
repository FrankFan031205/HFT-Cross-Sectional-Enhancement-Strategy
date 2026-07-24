# -*- coding: utf-8 -*-
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path("/mnt/data1/fwz/HFT_010-dev_fwz_outputs/final_report/v16_dual_alpha_csi2000_warmstart_noovernight")
RUN = "pure_cs_v16_dual_raw_step10_a10_02_t02_l18_csi2000"

SUMMARY_PATH = ROOT / RUN / "summary_by_rebalance.csv"
CURVE_PATH = ROOT / f"{RUN}_canonical_eval" / f"{RUN}_nav_curve_benchmark_warmstart_noovernight.csv"
OUT_DIR = ROOT / f"{RUN}_canonical_eval" / "fallback_diagnostics"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def compound(x):
    x = pd.to_numeric(x, errors="coerce").fillna(0.0)
    return float((1.0 + x).prod() - 1.0)


def pick_col(df, cands, name):
    for c in cands:
        if c in df.columns:
            return c
    raise KeyError(f"cannot find {name}; candidates={cands}; columns={df.columns.tolist()}")


rb = pd.read_csv(SUMMARY_PATH)
curve = pd.read_csv(CURVE_PATH)

rb["datetime"] = pd.to_datetime(rb["datetime"])
rb["date"] = rb["date"].astype(int)
rb["is_fallback"] = rb["status"].astype(str).str.contains(
    "fallback|error|infeasible", case=False, na=False
)

curve["datetime"] = pd.to_datetime(curve["datetime"])
curve["date"] = curve["datetime"].dt.strftime("%Y%m%d").astype(int)

actual_ret_col = pick_col(curve, ["actual_ret", "strategy_ret", "stock_actual_ret"], "actual ret")
bench_ret_col = pick_col(curve, ["benchmark_ret", "bench_ret"], "benchmark ret")

daily_fb = rb.groupby("date").agg(
    n_rebalance=("status", "size"),
    fallback=("is_fallback", "sum"),
    fallback_rate=("is_fallback", "mean"),
    turnover=("turnover_weight", "sum"),
)

daily_ret = curve.groupby("date").agg(
    actual_day=(actual_ret_col, compound),
    benchmark_day=(bench_ret_col, compound),
)
daily_ret["alpha_day"] = daily_ret["actual_day"] - daily_ret["benchmark_day"]

daily = daily_fb.join(daily_ret, how="left").reset_index()
daily["date_str"] = daily["date"].astype(str)
daily["alpha_bps"] = daily["alpha_day"] * 10000.0

corr_fb = daily["fallback_rate"].corr(daily["alpha_day"])
corr_turn = daily["turnover"].corr(daily["alpha_day"])

# 1) Daily alpha + fallback rate
fig, ax1 = plt.subplots(figsize=(15, 7))
x = np.arange(len(daily))
ax1.bar(x, daily["alpha_bps"], label="daily alpha")
ax1.axhline(0, linewidth=1)
ax1.set_ylabel("daily alpha (bps)")
ax1.set_xlabel("date")
ax1.set_xticks(x)
ax1.set_xticklabels(daily["date_str"], rotation=45, ha="right")

ax2 = ax1.twinx()
ax2.plot(x, daily["fallback_rate"] * 100.0, marker="o", label="fallback rate")
ax2.set_ylabel("fallback rate (%)")

fig.suptitle(f"{RUN}: Daily alpha vs fallback rate\ncorr(fallback_rate, alpha_day) = {corr_fb:.3f}")
fig.tight_layout()
p1 = OUT_DIR / f"{RUN}_daily_alpha_vs_fallback_rate.png"
fig.savefig(p1, dpi=160)
plt.close(fig)

# 2) Scatter: fallback rate vs alpha
fig, ax = plt.subplots(figsize=(9, 7))
ax.scatter(daily["fallback_rate"] * 100.0, daily["alpha_bps"])
for _, r in daily.iterrows():
    ax.annotate(str(int(r["date"])), (r["fallback_rate"] * 100.0, r["alpha_bps"]), fontsize=8)
ax.axhline(0, linewidth=1)
ax.set_xlabel("fallback rate (%)")
ax.set_ylabel("daily alpha (bps)")
ax.set_title(f"Fallback rate vs daily alpha, corr = {corr_fb:.3f}")
ax.grid(True, alpha=0.3)
fig.tight_layout()
p2 = OUT_DIR / f"{RUN}_scatter_fallback_rate_vs_alpha.png"
fig.savefig(p2, dpi=160)
plt.close(fig)

# 3) Scatter: turnover vs alpha
fig, ax = plt.subplots(figsize=(9, 7))
ax.scatter(daily["turnover"], daily["alpha_bps"])
for _, r in daily.iterrows():
    ax.annotate(str(int(r["date"])), (r["turnover"], r["alpha_bps"]), fontsize=8)
ax.axhline(0, linewidth=1)
ax.set_xlabel("daily turnover weight")
ax.set_ylabel("daily alpha (bps)")
ax.set_title(f"Turnover vs daily alpha, corr = {corr_turn:.3f}")
ax.grid(True, alpha=0.3)
fig.tight_layout()
p3 = OUT_DIR / f"{RUN}_scatter_turnover_vs_alpha.png"
fig.savefig(p3, dpi=160)
plt.close(fig)

# 4) Cumulative alpha curve with high-fallback days highlighted
curve = curve.sort_values("datetime").reset_index(drop=True)
curve["alpha_step"] = pd.to_numeric(curve[actual_ret_col], errors="coerce").fillna(0.0) - pd.to_numeric(curve[bench_ret_col], errors="coerce").fillna(0.0)
curve["alpha_cum"] = (1.0 + curve["alpha_step"]).cumprod() - 1.0

high_fb_dates = set(daily.loc[daily["fallback_rate"] >= 0.5, "date"].astype(int))

fig, ax = plt.subplots(figsize=(15, 7))
ax.plot(curve["datetime"], curve["alpha_cum"] * 100.0, label="cumulative alpha")
ax.axhline(0, linewidth=1)
for d in high_fb_dates:
    sub = curve[curve["date"] == d]
    if len(sub):
        ax.axvspan(sub["datetime"].iloc[0], sub["datetime"].iloc[-1], alpha=0.15)
ax.set_title(f"{RUN}: Cumulative alpha with high-fallback days shaded")
ax.set_xlabel("datetime")
ax.set_ylabel("cumulative alpha (%)")
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()
p4 = OUT_DIR / f"{RUN}_cum_alpha_high_fallback_shaded.png"
fig.savefig(p4, dpi=160)
plt.close(fig)

# Save daily table too
daily_path = OUT_DIR / f"{RUN}_daily_fallback_alpha_table.csv"
daily.to_csv(daily_path, index=False)

print("saved:")
for p in [p1, p2, p3, p4, daily_path]:
    print(p)

print("\nsummary:")
print("corr fallback_rate vs alpha_day:", corr_fb)
print("corr turnover vs alpha_day:", corr_turn)
print("high fallback dates:", sorted(high_fb_dates))
