# -*- coding: utf-8 -*-
"""
Prepare v16-minimal input for the old T+1 optimizer.

It keeps the old optimizer input schema, but adds two explicit signal columns:
  alpha20_signal: position alpha (h20)
  alpha10_signal: short-horizon execution alpha (h10)

The old optimizer should use alpha20_signal as --signal-col, and the patched
old-engine script will use alpha10_signal only to modify buy/sell turnover cost.
"""
from __future__ import annotations
from pathlib import Path
import argparse
import pandas as pd
import numpy as np

KEYS = ["date", "ts", "sid"]


def pick_signal_col(df: pd.DataFrame, preferred: str, fallback: list[str]) -> str:
    if preferred and preferred in df.columns:
        return preferred
    for c in fallback:
        if c in df.columns:
            return c
    raise KeyError(f"cannot find signal column. preferred={preferred}, fallback={fallback}, cols={df.columns.tolist()}")


def norm_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = out["date"].astype(int)
    out["ts"] = out["ts"].astype(int)
    out["sid"] = out["sid"].astype(str)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h20-dir", type=str, default="/mnt/data1/fwz/HFT_010-dev_fwz_outputs/TakerPipeline/zzy_pure_cs_h20_res_minute_input_csi2000_weight_v7")
    ap.add_argument("--h10-dir", type=str, default="/mnt/data1/fwz/HFT_010-dev_fwz_outputs/TakerPipeline/zzy_pure_cs_h10_res_minute_input_csi2000_weight_v7")
    ap.add_argument("--fallback-h20-dir", type=str, default="/mnt/data1/fwz/HFT_010-dev_fwz_outputs/TakerPipeline/zzy_pure_cs_h20_res_minute_input")
    ap.add_argument("--fallback-h10-dir", type=str, default="/mnt/data1/fwz/HFT_010-dev_fwz_outputs/TakerPipeline/zzy_pure_cs_h10_res_minute_input")
    ap.add_argument("--out-dir", type=str, default="/mnt/data1/fwz/HFT_010-dev_fwz_outputs/TakerPipeline/zzy_pure_cs_v16_old_engine_dual_alpha_input")
    ap.add_argument("--h20-signal-col", type=str, default="pred_ret_h20")
    ap.add_argument("--h10-signal-col", type=str, default="pred_ret_h10")
    ap.add_argument("--clip", type=float, default=5.0)
    args = ap.parse_args()

    h20_dir = Path(args.h20_dir)
    h10_dir = Path(args.h10_dir)
    if not h20_dir.exists():
        h20_dir = Path(args.fallback_h20_dir)
    if not h10_dir.exists():
        h10_dir = Path(args.fallback_h10_dir)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(h20_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(h20_dir)

    stats = []
    for f20 in files:
        f10 = h10_dir / f20.name
        if not f10.exists():
            raise FileNotFoundError(f"missing h10 file for {f20.name}: {f10}")

        base = norm_keys(pd.read_parquet(f20))
        h10 = norm_keys(pd.read_parquet(f10))

        col20 = pick_signal_col(base, args.h20_signal_col, ["pred_ret_h20", "pred_ret_h10", "pred_z", "pred"])
        col10 = pick_signal_col(h10, args.h10_signal_col, ["pred_ret_h10", "pred_ret_h20", "pred_z", "pred"])

        sig20 = base[KEYS + [col20]].rename(columns={col20: "alpha20_signal"})
        sig10 = h10[KEYS + [col10]].rename(columns={col10: "alpha10_signal"})

        merged = base.drop(columns=["alpha20_signal", "alpha10_signal"], errors="ignore")
        merged = merged.merge(sig20, on=KEYS, how="left").merge(sig10, on=KEYS, how="left")

        for c in ["alpha20_signal", "alpha10_signal"]:
            merged[c] = pd.to_numeric(merged[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-args.clip, args.clip)

        # Old optimizer will read --signal-col alpha20_signal.
        # Keep pred_ret_h20 also aligned to alpha20_signal for diagnostics/backward compatibility.
        merged["pred_ret_h20"] = merged["alpha20_signal"]

        out_path = out_dir / f20.name
        merged.to_parquet(out_path, index=False)

        stats.append({
            "file": f20.name,
            "date": int(merged["date"].iloc[0]),
            "rows": len(merged),
            "n_sid": merged["sid"].nunique(),
            "alpha20_nonzero": float((merged["alpha20_signal"] != 0).mean()),
            "alpha10_nonzero": float((merged["alpha10_signal"] != 0).mean()),
            "alpha20_mean": float(merged["alpha20_signal"].mean()),
            "alpha20_std": float(merged["alpha20_signal"].std()),
            "alpha10_mean": float(merged["alpha10_signal"].mean()),
            "alpha10_std": float(merged["alpha10_signal"].std()),
        })
        print(f"[write] {out_path} rows={len(merged)} h20={col20} h10={col10}")

    st = pd.DataFrame(stats)
    st_path = out_dir / "_prepare_v16_old_engine_input_summary.csv"
    st.to_csv(st_path, index=False)
    print("\n===== summary =====")
    print(st.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("[saved]", st_path)
    print("[out_dir]", out_dir)


if __name__ == "__main__":
    main()
