# -*- coding: utf-8 -*-
from pathlib import Path
import pandas as pd

SRC_H10 = Path("/mnt/data1/fwz/HFT_010-dev_fwz_outputs/TakerPipeline/zzy_lasso0707_h10_res_input_csi2000_weight_v7")
SRC_H20 = Path("/mnt/data1/fwz/HFT_010-dev_fwz_outputs/TakerPipeline/zzy_lasso0707_h20_res_input_csi2000_weight_v7")

OUT_H10 = Path("/mnt/data1/fwz/HFT_010-dev_fwz_outputs/TakerPipeline/zzy_lasso0707_h10_res_input_csi2000_weight_v7_compat")
OUT_H20 = Path("/mnt/data1/fwz/HFT_010-dev_fwz_outputs/TakerPipeline/zzy_lasso0707_h20_res_input_csi2000_weight_v7_compat")

OUT_H10.mkdir(parents=True, exist_ok=True)
OUT_H20.mkdir(parents=True, exist_ok=True)

files = sorted(SRC_H20.glob("optimizer_input_*.parquet"))
if not files:
    raise FileNotFoundError(SRC_H20)

for f20 in files:
    date = f20.stem.replace("optimizer_input_", "")
    f10 = SRC_H10 / f20.name
    if not f10.exists():
        raise FileNotFoundError(f10)

    h20 = pd.read_parquet(f20)
    h10 = pd.read_parquet(f10)

    # h20 文件作为主表：保留 market/quote/benchmark + h20 signal
    # 删除会和 h10 merge 冲突的 h10 字段
    drop_h20 = [
        "pred_ret_h10", "predz_h10", "label_y_raw_h10",
    ]
    h20 = h20.drop(columns=[c for c in drop_h20 if c in h20.columns], errors="ignore")

    if "pred_ret_h20" not in h20.columns:
        raise KeyError(f"{f20} missing pred_ret_h20")

    # pred_ret 在 h20 目录里指向 h20，方便脚本兼容
    h20["pred_ret"] = h20["pred_ret_h20"]

    # h10 文件只保留 key + h10 signal，避免重复 market columns / duplicate signal columns
    key_candidates = ["date", "datetime", "minute", "securityid", "sid", "ts"]
    keys = [c for c in key_candidates if c in h10.columns]

    keep = keys + [c for c in ["pred_ret_h10", "predz_h10", "label_y_raw_h10"] if c in h10.columns]
    if "pred_ret_h10" not in h10.columns:
        raise KeyError(f"{f10} missing pred_ret_h10")

    h10 = h10[keep].drop_duplicates(keys, keep="last")

    out20 = OUT_H20 / f20.name
    out10 = OUT_H10 / f10.name

    h20.to_parquet(out20, index=False)
    h10.to_parquet(out10, index=False)

    print(
        date,
        "h20_cols=", len(h20.columns),
        "h10_cols=", h10.columns.tolist(),
        "h20_has_h10=", "pred_ret_h10" in h20.columns,
        "h10_nonnull=", h10["pred_ret_h10"].notna().mean(),
        "h20_nonnull=", h20["pred_ret_h20"].notna().mean(),
    )

print("\nDONE")
print("OUT_H10:", OUT_H10)
print("OUT_H20:", OUT_H20)
