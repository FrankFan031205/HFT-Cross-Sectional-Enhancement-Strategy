# -*- coding: utf-8 -*-
from pathlib import Path
import importlib.util
import sys
import numpy as np
import pandas as pd

DATES = [
    20241217, 20241218, 20241219, 20241220,
    20241223, 20241224, 20241225, 20241226, 20241227,
    20241230, 20241231,
    20250102, 20250103, 20250106, 20250107,
    20250108, 20250109, 20250110, 20250113, 20250114,
]

VERSION = "lasso_0707_xgb"
WEIGHT_PATH = Path("/mnt/data1/fwz/HFT_010-dev_fwz_outputs/benchmark_weights/csi2000_932000_weight_normalized_v3.parquet")

OUT_H10 = Path("/mnt/data1/fwz/HFT_010-dev_fwz_outputs/TakerPipeline/zzy_lasso0707_h10_res_input_csi2000_weight_v7")
OUT_H20 = Path("/mnt/data1/fwz/HFT_010-dev_fwz_outputs/TakerPipeline/zzy_lasso0707_h20_res_input_csi2000_weight_v7")

OUT_H10.mkdir(parents=True, exist_ok=True)
OUT_H20.mkdir(parents=True, exist_ok=True)


def find_data_loader():
    candidates = [
        Path("data_loader.py"),
        Path("TakerOptimizer/scripts/data_loader.py"),
        Path("TakerOptimizer/scripts/zzy_data_loader.py"),
        Path("/mnt/data1/zzy/optimizer_data/data_loader.py"),
        Path("/mnt/data1/zzy/data_loader.py"),
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "cannot find data_loader.py. Please put the upstream data_loader.py in repo root "
        "or TakerOptimizer/scripts/zzy_data_loader.py"
    )


def import_from_path(path):
    spec = importlib.util.spec_from_file_location("zzy_data_loader", str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["zzy_data_loader"] = mod
    spec.loader.exec_module(mod)
    return mod


def pick_col(df, cands, name):
    for c in cands:
        if c in df.columns:
            return c
    raise KeyError(f"cannot find {name}; candidates={cands}; columns={df.columns.tolist()}")


def hhmmssfff_to_datetime(date_s, ts_real_s):
    # ts_real: 93000000 / 930000000 / 930000 等都尽量兼容
    out = []
    for d, t in zip(date_s.astype(int).astype(str), ts_real_s):
        if pd.isna(t):
            out.append(pd.NaT)
            continue
        s = str(int(t))
        if len(s) >= 8:
            s = s.zfill(9)[:6]
        else:
            s = s.zfill(6)
        out.append(pd.to_datetime(d + " " + s[:2] + ":" + s[2:4] + ":" + s[4:6]))
    return pd.Series(out)


def load_benchmark_weight():
    if not WEIGHT_PATH.exists():
        raise FileNotFoundError(WEIGHT_PATH)

    w = pd.read_parquet(WEIGHT_PATH)
    sid_col = pick_col(w, ["sid", "securityid", "SecurityID", "symbol", "ticker", "code"], "sid")
    weight_col = pick_col(w, ["benchmark_weight", "weight", "index_weight", "w"], "weight")

    w = w.copy()
    w["sid"] = pd.to_numeric(w[sid_col].astype(str).str.extract(r"(\d+)")[0], errors="coerce").astype("Int64")
    w["benchmark_weight"] = pd.to_numeric(w[weight_col], errors="coerce").fillna(0.0)

    if "date" in w.columns:
        w["date"] = w["date"].astype(int)
        w = w[["date", "sid", "benchmark_weight"]].drop_duplicates(["date", "sid"], keep="last")
    else:
        w = w[["sid", "benchmark_weight"]].drop_duplicates(["sid"], keep="last")

    print("[benchmark weight]", WEIGHT_PATH)
    print(w.head())
    print("sum:", w["benchmark_weight"].sum())
    return w


def prep_one_day(df, bench_w, date):
    x = df.copy()

    # Polars -> pandas already done outside
    x["date"] = x["date"].astype(int)
    x["sid"] = x["sid"].astype(int)
    x["ts"] = x["ts"].astype(int)

    if "datetime" not in x.columns:
        x["datetime"] = hhmmssfff_to_datetime(x["date"], x["ts_real"])

    x["securityid"] = x["sid"].astype(str).str.zfill(6)
    x["minute"] = x["datetime"].dt.strftime("%H:%M:%S")

    # join CSI2000 benchmark weight
    if "date" in bench_w.columns:
        x = x.merge(bench_w, on=["date", "sid"], how="left")
    else:
        x = x.merge(bench_w, on=["sid"], how="left")
    x["benchmark_weight"] = x["benchmark_weight"].fillna(0.0)

    # execution / price aliases
    # 保留 task/tbid/tmid，同时给旧 optimizer 兼容 mid_price/bid_price/ask_price
    x["mid_price"] = pd.to_numeric(x["tmid"], errors="coerce")
    x["bid_price"] = pd.to_numeric(x["tbid"], errors="coerce")
    x["ask_price"] = pd.to_numeric(x["task"], errors="coerce")

    # 如果 microstructure 的 ask1/bid1 不存在，就用执行价兜底；存在则不覆盖
    if "ask1" not in x.columns:
        x["ask1"] = x["ask_price"]
    if "bid1" not in x.columns:
        x["bid1"] = x["bid_price"]

    x["spread_bps"] = (x["ask_price"] / x["bid_price"] - 1.0) * 10000.0
    x["turnover_amount"] = pd.to_numeric(x.get("vol", 0.0), errors="coerce").fillna(0.0) * x["mid_price"]

    # 兼容字段
    x["bid_volume1"] = pd.to_numeric(x.get("tbvol", np.nan), errors="coerce")
    x["ask_volume1"] = pd.to_numeric(x.get("tavol", np.nan), errors="coerce")
    x["limit_up"] = 0
    x["limit_down"] = 0
    x["industry"] = 0

    # 新版 residual h10/h20
    if "pred_res_10" not in x.columns or "pred_res_20" not in x.columns:
        raise KeyError(f"missing pred_res_10/pred_res_20; columns={x.columns.tolist()}")

    x["pred_ret_h10"] = pd.to_numeric(x["pred_res_10"], errors="coerce")
    x["pred_ret_h20"] = pd.to_numeric(x["pred_res_20"], errors="coerce")

    # labels optional
    if "y_10" in x.columns:
        x["label_y_raw_h10"] = pd.to_numeric(x["y_10"], errors="coerce")
    if "y_20" in x.columns:
        x["label_y_raw_h20"] = pd.to_numeric(x["y_20"], errors="coerce")
        x["label_y_raw"] = x["label_y_raw_h20"]

    base_cols = [
        "date", "datetime", "minute", "securityid", "sid", "ts", "ts_real",
        "task", "tbid", "tmid", "tavol", "tbvol", "vol",
        "mid_price", "bid_price", "ask_price", "bid1", "ask1",
        "bid_volume1", "ask_volume1", "spread_bps", "turnover_amount",
        "benchmark_weight", "limit_up", "limit_down", "industry",
        "pred_ret_h10", "pred_ret_h20",
    ]

    extra_cols = [c for c in ["label_y_raw", "label_y_raw_h10", "label_y_raw_h20", "pred_ts_10", "pred_ts_20", "predz_ts_10", "predz_ts_20"] if c in x.columns]
    cols = base_cols + extra_cols

    return x[cols].sort_values(["date", "datetime", "securityid", "ts"])


def main():
    loader_path = find_data_loader()
    print("[data_loader]", loader_path)
    dl = import_from_path(loader_path)
    bench_w = load_benchmark_weight()

    for d in DATES:
        print(f"\n===== load date {d} version={VERSION} =====")
        m = dl.load_master(dates=[d], horizons=[10, 20], models=("ts", "res"), version=VERSION, n_workers=48)
        pdf = m.to_pandas()
        out = prep_one_day(pdf, bench_w, d)

        h10 = out.copy()
        h20 = out.copy()

        # 让每个目录里主 signal 名字也符合 horizon
        h10["pred_ret"] = h10["pred_ret_h10"]
        h20["pred_ret"] = h20["pred_ret_h20"]

        p10 = OUT_H10 / f"optimizer_input_{d}.parquet"
        p20 = OUT_H20 / f"optimizer_input_{d}.parquet"

        h10.to_parquet(p10, index=False)
        h20.to_parquet(p20, index=False)

        print("[saved h10]", p10, h10.shape)
        print("[saved h20]", p20, h20.shape)
        print("bench_sum first dt:", out.groupby(["date", "datetime"])["benchmark_weight"].sum().iloc[0])
        print("pred h10 nonnull:", out["pred_ret_h10"].notna().mean(), "pred h20 nonnull:", out["pred_ret_h20"].notna().mean())

    print("\nDONE")
    print("h10_dir:", OUT_H10)
    print("h20_dir:", OUT_H20)


if __name__ == "__main__":
    main()
