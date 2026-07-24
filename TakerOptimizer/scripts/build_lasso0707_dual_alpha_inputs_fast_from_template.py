# -*- coding: utf-8 -*-
from pathlib import Path
import pandas as pd
import numpy as np

DATES = [
    20241217, 20241218, 20241219, 20241220,
    20241223, 20241224, 20241225, 20241226, 20241227,
    20241230, 20241231,
    20250102, 20250103, 20250106, 20250107,
    20250108, 20250109, 20250110, 20250113, 20250114,
]

# 用旧 canonical input 当 market / quotes / benchmark 模板，不重新读每股 quotes
TEMPLATE_H10 = Path("/mnt/data1/fwz/HFT_010-dev_fwz_outputs/TakerPipeline/zzy_pure_cs_h10_res_minute_input_csi2000_weight_v7")
TEMPLATE_H20 = Path("/mnt/data1/fwz/HFT_010-dev_fwz_outputs/TakerPipeline/zzy_pure_cs_h20_res_minute_input_csi2000_weight_v7")

# 新版 lasso 残差预测
PRED10 = Path("/mnt/data1/zzy/optimizer_data/pred/lasso_0707_xgb/10min/test_predictions.parquet")
PRED20 = Path("/mnt/data1/zzy/optimizer_data/pred/lasso_0707_xgb/20min/test_predictions.parquet")

OUT_H10 = Path("/mnt/data1/fwz/HFT_010-dev_fwz_outputs/TakerPipeline/zzy_lasso0707_h10_res_input_csi2000_weight_v7")
OUT_H20 = Path("/mnt/data1/fwz/HFT_010-dev_fwz_outputs/TakerPipeline/zzy_lasso0707_h20_res_input_csi2000_weight_v7")
OUT_H10.mkdir(parents=True, exist_ok=True)
OUT_H20.mkdir(parents=True, exist_ok=True)

for p in [TEMPLATE_H10, TEMPLATE_H20, PRED10, PRED20]:
    if not p.exists():
        raise FileNotFoundError(p)

print("[read new predictions]")
print("PRED10:", PRED10)
print("PRED20:", PRED20)

p10 = pd.read_parquet(PRED10, columns=["date", "sid", "ts", "pred", "pred_z", "y_raw"])
p20 = pd.read_parquet(PRED20, columns=["date", "sid", "ts", "pred", "pred_z", "y_raw"])

for x in [p10, p20]:
    x["date"] = x["date"].astype(int)
    x["sid"] = x["sid"].astype(int)
    x["ts"] = x["ts"].astype(int)

p10 = p10[p10["date"].isin(DATES)].copy()
p20 = p20[p20["date"].isin(DATES)].copy()

p10 = p10.rename(columns={
    "pred": "new_pred_ret_h10",
    "pred_z": "new_predz_h10",
    "y_raw": "new_y_h10",
})
p20 = p20.rename(columns={
    "pred": "new_pred_ret_h20",
    "pred_z": "new_predz_h20",
    "y_raw": "new_y_h20",
})

key = ["date", "sid", "ts"]

print("[pred10]", p10.shape, p10["date"].min(), p10["date"].max())
print("[pred20]", p20.shape, p20["date"].min(), p20["date"].max())

# 分日字典，避免每次扫全表
p10_by_date = {d: g.drop_duplicates(key, keep="last") for d, g in p10.groupby("date")}
p20_by_date = {d: g.drop_duplicates(key, keep="last") for d, g in p20.groupby("date")}

def patch_one_day(date: int):
    f10 = TEMPLATE_H10 / f"optimizer_input_{date}.parquet"
    f20 = TEMPLATE_H20 / f"optimizer_input_{date}.parquet"

    if not f10.exists():
        raise FileNotFoundError(f10)
    if not f20.exists():
        raise FileNotFoundError(f20)

    # 用 h20 模板作为完整 market 模板；如果你希望 h10/h20 两个目录保留原结构，也可分别读
    base = pd.read_parquet(f20)

    base["date"] = base["date"].astype(int)
    base["sid"] = base["sid"].astype(int)
    base["ts"] = base["ts"].astype(int)

    # 清掉旧预测列，保留所有 market/quote/benchmark 字段
    drop_cols = [
        "pred_ret", "pred_ret_h10", "pred_ret_h20",
        "predz_h10", "predz_h20",
        "label_y_raw", "label_y_raw_h10", "label_y_raw_h20",
        "new_pred_ret_h10", "new_pred_ret_h20",
        "new_predz_h10", "new_predz_h20",
        "new_y_h10", "new_y_h20",
    ]
    base = base.drop(columns=[c for c in drop_cols if c in base.columns])

    x = base.merge(p10_by_date[date], on=key, how="left")
    x = x.merge(p20_by_date[date], on=key, how="left")

    x["pred_ret_h10"] = pd.to_numeric(x["new_pred_ret_h10"], errors="coerce")
    x["pred_ret_h20"] = pd.to_numeric(x["new_pred_ret_h20"], errors="coerce")
    x["predz_h10"] = pd.to_numeric(x["new_predz_h10"], errors="coerce")
    x["predz_h20"] = pd.to_numeric(x["new_predz_h20"], errors="coerce")
    x["label_y_raw_h10"] = pd.to_numeric(x["new_y_h10"], errors="coerce")
    x["label_y_raw_h20"] = pd.to_numeric(x["new_y_h20"], errors="coerce")
    x["label_y_raw"] = x["label_y_raw_h20"]

    # 输出两个目录：如果 optimizer 读 pred_ret，则 h10 dir 的 pred_ret=h10，h20 dir 的 pred_ret=h20
    h10 = x.copy()
    h20 = x.copy()

    h10["pred_ret"] = h10["pred_ret_h10"]
    h20["pred_ret"] = h20["pred_ret_h20"]

    # 清理临时列
    tmp_cols = ["new_pred_ret_h10", "new_pred_ret_h20", "new_predz_h10", "new_predz_h20", "new_y_h10", "new_y_h20"]
    h10 = h10.drop(columns=[c for c in tmp_cols if c in h10.columns])
    h20 = h20.drop(columns=[c for c in tmp_cols if c in h20.columns])

    out10 = OUT_H10 / f"optimizer_input_{date}.parquet"
    out20 = OUT_H20 / f"optimizer_input_{date}.parquet"

    h10.to_parquet(out10, index=False)
    h20.to_parquet(out20, index=False)

    # 检查 benchmark 是否保留正确
    first_sum = float(h20.groupby(["date", "datetime"])["benchmark_weight"].sum().iloc[0]) if "benchmark_weight" in h20.columns else np.nan

    print(
        date,
        "rows=", len(h20),
        "bench_sum_first_dt=", round(first_sum, 6),
        "h10_nonnull=", round(float(h20["pred_ret_h10"].notna().mean()), 6),
        "h20_nonnull=", round(float(h20["pred_ret_h20"].notna().mean()), 6),
        "saved"
    )

for d in DATES:
    patch_one_day(d)

print("\nDONE")
print("h10_dir:", OUT_H10)
print("h20_dir:", OUT_H20)
