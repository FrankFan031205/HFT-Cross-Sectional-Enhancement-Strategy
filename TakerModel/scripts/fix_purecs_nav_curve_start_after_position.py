# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compound_curve(x):
    x = pd.Series(x).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return (1.0 + x).cumprod() - 1.0


def make_ticks(df):
    first = df.groupby("date", as_index=False)["bar_index"].min()
    n = len(first)
    if n <= 12:
        step = 1
    elif n <= 24:
        step = 2
    else:
        step = max(1, n // 12)
    ticks = first.iloc[::step]
    return ticks["bar_index"].tolist(), ticks["date"].astype(str).tolist()


def filter_after_first_position(df, min_gross):
    out = []
    removed = []

    for d, g in df.groupby("date", sort=True):
        g = g.sort_values("datetime").copy()

        # NAV return uses previous-minute holding, not current target.
        # Therefore the fair start point is the first minute where held_gross_prev is already meaningful.
        gross_col = "held_gross_prev" if "held_gross_prev" in g.columns else "target_gross"

        valid = g[g[gross_col] > min_gross]
        if valid.empty:
            removed.append({
                "date": d,
                "removed_minutes": len(g),
                "first_kept_datetime": None,
                "reason": f"no {gross_col} above threshold",
            })
            continue

        first_dt = valid["datetime"].iloc[0]
        kept = g[g["datetime"] >= first_dt].copy()

        removed.append({
            "date": d,
            "removed_minutes": int((g["datetime"] < first_dt).sum()),
            "first_kept_datetime": first_dt,
            "reason": f"drop pre-position window by {gross_col}",
        })

        out.append(kept)

    if not out:
        raise RuntimeError("no data left after filtering first-position window")

    return pd.concat(out, ignore_index=True), pd.DataFrame(removed)


def recompute_curves(df):
    df = df.sort_values(["date", "datetime"]).reset_index(drop=True)
    df["bar_index"] = np.arange(len(df))

    df["actualret_clean"] = compound_curve(df["actual_ret"])
    df["benchmarkret_clean"] = compound_curve(df["benchmark_ret"])
    df["alpharet_clean"] = df["actualret_clean"] - df["benchmarkret_clean"]

    df["alpha_ret"] = df["actual_ret"] - df["benchmark_ret"]
    df["alpharet_compound_clean"] = compound_curve(df["alpha_ret"])

    return df


def plot_curves(df, out_dir):
    tick_pos, tick_lab = make_ticks(df)

    plt.figure(figsize=(14, 6))
    plt.plot(df["bar_index"], df["actualret_clean"] * 100.0, label="actualret")
    plt.plot(df["bar_index"], df["benchmarkret_clean"] * 100.0, label="benchmarkret")
    plt.plot(df["bar_index"], df["alpharet_clean"] * 100.0, label="alpharet")
    plt.axhline(0.0, linewidth=0.8)
    plt.xticks(tick_pos, tick_lab, rotation=45)
    plt.title("Pure-CS NAV Curve, Start After First Position")
    plt.xlabel("trading minute index, pre-position open window removed")
    plt.ylabel("cumulative return (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = out_dir / "nav_curve_start_after_position_actualret_benchmarkret_alpharet.png"
    plt.savefig(path, dpi=180)
    plt.close()
    print("[saved]", path)

    plt.figure(figsize=(14, 5))
    plt.plot(df["bar_index"], df["alpharet_clean"] * 100.0, label="alpharet")
    plt.axhline(0.0, linewidth=0.8)
    plt.xticks(tick_pos, tick_lab, rotation=45)
    plt.title("Pure-CS Alpha Curve, Start After First Position")
    plt.xlabel("trading minute index, pre-position open window removed")
    plt.ylabel("cumulative alpha (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = out_dir / "nav_curve_start_after_position_alpharet.png"
    plt.savefig(path, dpi=180)
    plt.close()
    print("[saved]", path)


def spike_check(df, out_dir, top_n):
    x = df.copy()
    x["abs_alpha_ret"] = (x["actual_ret"] - x["benchmark_ret"]).abs()
    x["alpha_ret"] = x["actual_ret"] - x["benchmark_ret"]

    cols = [
        "date", "datetime", "bar_index",
        "actual_ret", "benchmark_ret", "alpha_ret",
        "actualret_clean", "benchmarkret_clean", "alpharet_clean",
        "execution_cost", "execution_cost_return",
        "target_gross", "held_gross_prev",
        "benchmark_gross", "n_target_names",
    ]
    cols = [c for c in cols if c in x.columns]

    top = x.sort_values("abs_alpha_ret", ascending=False).head(top_n)[cols]
    path = out_dir / "top_alpha_spikes_after_position_filter.csv"
    top.to_csv(path, index=False, encoding="utf-8-sig")
    print("[saved]", path)

    print("\n===== top alpha spikes after position filter =====")
    print(top.head(20).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curve-csv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--min-gross", type=float, default=0.10)
    ap.add_argument("--top-n", type=int, default=50)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.curve_csv)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values(["date", "datetime"]).reset_index(drop=True)

    required = ["date", "datetime", "actual_ret", "benchmark_ret", "target_gross"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"missing required columns: {missing}")

    clean, removed = filter_after_first_position(df, args.min_gross)
    clean = recompute_curves(clean)

    clean_path = out_dir / "nav_curve_start_after_position.csv"
    removed_path = out_dir / "removed_pre_position_minutes_by_day.csv"

    clean.to_csv(clean_path, index=False, encoding="utf-8-sig")
    removed.to_csv(removed_path, index=False, encoding="utf-8-sig")

    print("[saved]", clean_path)
    print("[saved]", removed_path)

    plot_curves(clean, out_dir)
    spike_check(clean, out_dir, args.top_n)

    print("\n===== final after position filter =====")
    print("n minute bars:", len(clean))
    print("actualret:", clean["actualret_clean"].iloc[-1])
    print("benchmarkret:", clean["benchmarkret_clean"].iloc[-1])
    print("alpharet:", clean["alpharet_clean"].iloc[-1])
    print("alpharet_compound:", clean["alpharet_compound_clean"].iloc[-1])

    print("\n===== removed minutes by day =====")
    print(removed.to_string(index=False))


if __name__ == "__main__":
    main()
