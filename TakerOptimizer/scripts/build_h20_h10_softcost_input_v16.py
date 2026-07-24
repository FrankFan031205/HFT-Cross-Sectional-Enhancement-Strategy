# -*- coding: utf-8 -*-
from pathlib import Path
import pandas as pd
import numpy as np

H20_DIR = Path("/mnt/data1/fwz/HFT_010-dev_fwz_outputs/TakerPipeline/zzy_pure_cs_h20_res_minute_input")
H10_DIR = Path("/mnt/data1/fwz/HFT_010-dev_fwz_outputs/TakerPipeline/zzy_pure_cs_h10_res_minute_input")
OUT_DIR = Path("/mnt/data1/fwz/HFT_010-dev_fwz_outputs/TakerPipeline/zzy_pure_cs_h20_h10_softcost_v16_input")

OUT_DIR.mkdir(parents=True, exist_ok=True)

h20_files = sorted(H20_DIR.glob("optimizer_input_*.parquet"))
if not h20_files:
    raise FileNotFoundError(f"no h20 files in {H20_DIR}")

print("h20_files:", len(h20_files))
print("h20_dir:", H20_DIR)
print("h10_dir:", H10_DIR)
print("out_dir:", OUT_DIR)

def norm_key(df):
    df = df.copy()
    df["date"] = df["date"].astype(int)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["securityid"] = df["securityid"].astype(str).str.zfill(6)
    return df

for h20_path in h20_files:
    h10_path = H10_DIR / h20_path.name
    if not h10_path.exists():
        raise FileNotFoundError(f"missing h10 file: {h10_path}")

    h20 = pd.read_parquet(h20_path)
    h10 = pd.read_parquet(h10_path)

    h20 = norm_key(h20)
    h10 = norm_key(h10)

    if "pred_ret_h20" not in h20.columns:
        raise KeyError(f"{h20_path} has no pred_ret_h20; cols={h20.columns.tolist()}")

    if "pred_ret_h10" in h10.columns:
        h10_sig_col = "pred_ret_h10"
    elif "pred_ret" in h10.columns:
        h10_sig_col = "pred_ret"
    else:
        cands = [c for c in h10.columns if "h10" in c.lower() or "pred" in c.lower()]
        if not cands:
            raise KeyError(f"{h10_path} has no h10/pred column; cols={h10.columns.tolist()}")
        h10_sig_col = cands[0]

    key = ["date", "datetime", "securityid"]

    h10_small = h10[key + [h10_sig_col]].copy()
    h10_small = h10_small.rename(columns={h10_sig_col: "pred_ret_h10"})
    h10_small = h10_small.drop_duplicates(key, keep="last")

    # Avoid stale columns if rerun.
    h20 = h20.drop(columns=["pred_ret_h10", "soft_cost_signal"], errors="ignore")

    out = h20.merge(h10_small, on=key, how="left", indicator=True)

    match_rate = float((out["_merge"] == "both").mean())
    nan_rate = float(out["pred_ret_h10"].isna().mean())

    out = out.drop(columns=["_merge"])
    out["pred_ret_h10"] = out["pred_ret_h10"].fillna(0.0).astype("float32")
    out["soft_cost_signal"] = out["pred_ret_h10"]

    out_path = OUT_DIR / h20_path.name
    out.to_parquet(out_path, index=False)

    print(
        h20_path.name,
        "rows=", len(out),
        "match_rate=", round(match_rate, 6),
        "nan_rate=", round(nan_rate, 6),
        "cols=", [c for c in out.columns if "pred" in c or "soft" in c],
    )

print("\nDONE")
print("input_glob:", str(OUT_DIR / "*.parquet"))
