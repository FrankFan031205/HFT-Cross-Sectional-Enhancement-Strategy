# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def make_date_ticks(df):
    first = df.groupby("date", as_index=False)["bar_index"].min()
    n = len(first)
    if n <= 12:
        step = 1
    elif n <= 24:
        step = 2
    else:
        step = max(1, n // 12)
    ticks = first.iloc[::step].copy()
    return ticks["bar_index"].tolist(), ticks["date"].astype(str).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curve-csv", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.curve_csv)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values(["date", "datetime"]).reset_index(drop=True)
    df["bar_index"] = np.arange(len(df))

    tick_pos, tick_lab = make_date_ticks(df)

    plt.figure(figsize=(14, 6))
    plt.plot(df["bar_index"], df["actualret"] * 100, label="actualret")
    plt.plot(df["bar_index"], df["benchmarkret"] * 100, label="benchmarkret")
    plt.plot(df["bar_index"], df["alpharet"] * 100, label="alpharet = actualret - benchmarkret")
    plt.axhline(0, linewidth=0.8)
    plt.xticks(tick_pos, tick_lab, rotation=45)
    plt.title("Pure-CS Minute Curve, Compressed Trading Time")
    plt.xlabel("trading minute, overnight/weekend gaps removed")
    plt.ylabel("cumulative return (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out = out_dir / "curve_minute_compressed_actualret_benchmarkret_alpharet.png"
    plt.savefig(out, dpi=180)
    plt.close()
    print("[saved]", out)

    plt.figure(figsize=(14, 6))
    plt.plot(df["bar_index"], df["actualret"] * 100, label="actualret")
    plt.plot(df["bar_index"], df["benchmarkret"] * 100, label="benchmarkret")
    plt.axhline(0, linewidth=0.8)
    plt.xticks(tick_pos, tick_lab, rotation=45)
    plt.title("Pure-CS Minute Curve: Actual vs Full Benchmark")
    plt.xlabel("trading minute, overnight/weekend gaps removed")
    plt.ylabel("cumulative return (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out = out_dir / "curve_minute_compressed_actual_vs_benchmark.png"
    plt.savefig(out, dpi=180)
    plt.close()
    print("[saved]", out)

    plt.figure(figsize=(14, 5))
    plt.plot(df["bar_index"], df["alpharet"] * 100, label="alpharet")
    plt.axhline(0, linewidth=0.8)
    plt.xticks(tick_pos, tick_lab, rotation=45)
    plt.title("Pure-CS Minute Alpha Curve: actualret - benchmarkret")
    plt.xlabel("trading minute, overnight/weekend gaps removed")
    plt.ylabel("cumulative alpha (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out = out_dir / "curve_minute_compressed_alpharet.png"
    plt.savefig(out, dpi=180)
    plt.close()
    print("[saved]", out)

    out_csv = out_dir / "curve_minute_compressed_actualret_benchmarkret_alpharet.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print("[saved]", out_csv)

    print("\n===== final =====")
    print("n minute bars:", len(df))
    print("actualret:", df["actualret"].iloc[-1])
    print("benchmarkret:", df["benchmarkret"].iloc[-1])
    print("alpharet:", df["alpharet"].iloc[-1])


if __name__ == "__main__":
    main()
