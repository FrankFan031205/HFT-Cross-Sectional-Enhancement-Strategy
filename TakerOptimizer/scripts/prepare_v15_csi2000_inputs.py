# -*- coding: utf-8 -*-
"""
Prepare CSI2000-weighted h10 input for v15.

Why this exists:
- The previous v15 run accidentally used the old h20/h10 input dirs without CSI2000 weights.
- That changed the benchmark return from about -3.97% to about -7.66%.
- v15 must use the same CSI2000-weighted market/benchmark universe as v10.

Output:
  /mnt/data1/fwz/HFT_010-dev_fwz_outputs/TakerPipeline/zzy_pure_cs_h10_res_minute_input_csi2000_weight_v7
"""
from pathlib import Path
import pandas as pd
import numpy as np

H20_CSI_DIR = Path("/mnt/data1/fwz/HFT_010-dev_fwz_outputs/TakerPipeline/zzy_pure_cs_h20_res_minute_input_csi2000_weight_v7")
H10_ORIG_DIR = Path("/mnt/data1/fwz/HFT_010-dev_fwz_outputs/TakerPipeline/zzy_pure_cs_h10_res_minute_input")
OUT_H10_CSI_DIR = Path("/mnt/data1/fwz/HFT_010-dev_fwz_outputs/TakerPipeline/zzy_pure_cs_h10_res_minute_input_csi2000_weight_v7")
OUT_H10_CSI_DIR.mkdir(parents=True, exist_ok=True)

KEYS = ["date", "ts", "sid"]


def pick_signal_col(cols):
    for c in ["pred_ret_h10", "pred_z", "pred_ret_h20", "pred", "signal", "alpha"]:
        if c in cols:
            return c
    raise KeyError(f"cannot find h10 signal column in {list(cols)}")


def norm_keys(df):
    df = df.copy()
    df["date"] = df["date"].astype(int)
    df["ts"] = df["ts"].astype(int)
    df["sid"] = df["sid"].astype(str)
    return df


def main():
    files = sorted(H20_CSI_DIR.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(H20_CSI_DIR)

    rows = []
    print("[h20 csi dir]", H20_CSI_DIR)
    print("[h10 orig dir]", H10_ORIG_DIR)
    print("[out h10 csi dir]", OUT_H10_CSI_DIR)

    for f in files:
        h10_path = H10_ORIG_DIR / f.name
        if not h10_path.exists():
            raise FileNotFoundError(h10_path)

        base = norm_keys(pd.read_parquet(f))
        h10 = norm_keys(pd.read_parquet(h10_path))
        sig = pick_signal_col(h10.columns)

        h10_small = h10[KEYS + [sig]].rename(columns={sig: "pred_ret_h10"})
        h10_small["pred_ret_h10"] = pd.to_numeric(h10_small["pred_ret_h10"], errors="coerce")
        h10_small = h10_small.drop_duplicates(KEYS, keep="last")

        out = base[KEYS + ["datetime", "securityid", "benchmark_weight"]].merge(
            h10_small, on=KEYS, how="left"
        )
        hit = out["pred_ret_h10"].notna().mean()
        missing = int(out["pred_ret_h10"].isna().sum())
        out["pred_ret_h10"] = out["pred_ret_h10"].fillna(0.0)

        # v15 only needs KEYS + pred_ret_h10 from the h10 dir, but keep datetime/securityid/benchmark_weight
        # for diagnostics and easy manual inspection.
        out_path = OUT_H10_CSI_DIR / f.name
        out.to_parquet(out_path, index=False)

        rows.append({
            "file": f.name,
            "date": int(base["date"].iloc[0]),
            "rows": len(out),
            "hit_ratio": float(hit),
            "missing_rows": missing,
            "h10_signal_col": sig,
            "benchmark_weight_sum_base": float(pd.to_numeric(base["benchmark_weight"], errors="coerce").fillna(0).sum()),
            "n_sid": int(out["sid"].nunique()),
            "n_securityid": int(out["securityid"].nunique()),
        })
        print(f"[write] {out_path.name} rows={len(out)} hit={hit:.4%} missing={missing} sig={sig}")

    summary = pd.DataFrame(rows)
    summary_path = OUT_H10_CSI_DIR / "_prepare_h10_csi2000_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\n===== summary =====")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\n[saved]", summary_path)

    if summary["hit_ratio"].mean() < 0.95:
        raise RuntimeError(f"h10 hit ratio too low: {summary['hit_ratio'].mean():.4%}")


if __name__ == "__main__":
    main()
