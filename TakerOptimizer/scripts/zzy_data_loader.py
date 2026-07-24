# -*- coding: utf-8 -*-
"""data_loader.py —— 多线程拉取优化器输入,在内存里拼成【一张对齐大表】返回。

磁盘上 quotes 与 预测各自分开存(不生成任何合并文件);本模块在【加载时】把它们按
(date, sid, ts) 拼成一张大表作为函数返回值,调用方直接用,不落盘。

主入口:
    from data_loader import load_master
    df = load_master(dates=[20241217, 20241218])                   # 默认 version="v1"(上一版全量因子)
    df = load_master(dates=..., version="lasso_0707_xgb")          # 新版(lasso 筛因子 + xgb)
    # df 即对齐大表,在内存里,没写盘

积木(需要单独某块时):
    from data_loader import load_quotes, load_preds, load_pred, load_overnight
    q  = load_quotes(dates=...)            # 仅 quotes
    p  = load_preds(dates=...)             # 仅全 horizon 预测(宽表)
    p5 = load_pred(5, model="ts", dates=...)   # 单 horizon 单模型原始预测
    o  = load_overnight(model="resid", dates=...)  # 隔夜预测(尾盘→次日早盘复权收益);独立口径,不并入大表

CLI(只演示/查规模,不落盘):
    python data_loader.py --dates 20241217 20241218
"""
import argparse
import os
from concurrent.futures import ThreadPoolExecutor

import polars as pl

PRED_ROOT    = "/mnt/data1/zzy/optimizer_data/pred"               # 残差模型预测(label=cs_zscore)
PRED_TS_ROOT = "/mnt/data1/zzy/optimizer_data/pred_ts"            # 时序模型预测(label=y_raw)
QUOTES_ROOT  = "/mnt/data1/zzy/optimizer_data/quotes"             # 执行数据(每股一文件,多为前向)
MICRO_ROOT   = "/mnt/data1/zzy/optimizer_data/microstructure"     # 因果微观结构层(每股一文件,全因果)
OVERNIGHT_ROOT = "/mnt/data1/zzy/optimizer_data/overnight_pred"   # 隔夜预测(尾盘→次日早盘复权收益;与 pred 并列)

# ★预测按版本嵌套:<root>/<version>/<h>min/test_predictions.parquet
DEFAULT_VERSION = "v1"                                            # 默认版本(上一版全量因子)
KNOWN_VERSIONS  = ("v1", "lasso_0707_xgb")                        # v1=全量因子;lasso_0707_xgb=lasso 筛因子 + xgb

_KEY = ["date", "sid", "ts"]


def _avail_horizons(root, version=DEFAULT_VERSION):
    """某预测根 + version 下有哪些 horizon(<version>/<h>min/test_predictions.parquet 存在)。"""
    vroot = os.path.join(root, version)
    if not os.path.isdir(vroot):
        return []
    return sorted(int(d[:-3]) for d in os.listdir(vroot)
                  if d.endswith("min") and d[:-3].isdigit()
                  and os.path.exists(os.path.join(vroot, d, "test_predictions.parquet")))


def _cast_keys(df):
    return df.with_columns(pl.col("date").cast(pl.Int64), pl.col("sid").cast(pl.Int64), pl.col("ts").cast(pl.Int64))


# ============================== quotes ==============================
def load_quotes(dates=None, quotes_root=QUOTES_ROOT, n_workers=48):
    """多线程读 quotes(每股一文件)→ 一张长表。dates=None → TimeSeries 下全部日期。
    列:date,sid,ts(行号),ts_real(真实时刻),task,tbid,tmid,tavol,tbvol,vol。"""
    ts_root = os.path.join(quotes_root, "TimeSeries")
    if dates is None:
        dates = sorted(int(d) for d in os.listdir(ts_root) if d.isdigit())
    tasks = []
    for d in dates:
        dd = os.path.join(ts_root, str(d))
        if not os.path.isdir(dd):
            continue
        for s in os.listdir(dd):
            fp = os.path.join(dd, s, "quotes.parquet")
            if os.path.exists(fp):
                tasks.append(fp)
    if not tasks:
        raise SystemExit(f"!! 读不到 quotes({quotes_root});检查日期 {dates}")
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        parts = list(ex.map(pl.read_parquet, tasks))
    return _cast_keys(pl.concat(parts, how="vertical"))


# ============================== 微观结构(因果层)==============================
def load_microstructure(dates=None, micro_root=MICRO_ROOT, n_workers=48, required=True):
    """多线程读因果微观结构层(每股一文件)→ 一张长表。dates=None → TimeSeries 下全部日期。
    列:date,sid,ts(行号),ts_real,ask1,bid1,mid,spread,vol_bwd —— 全部决策当下可见(因果),
    与 pred/quotes 同一套 (date,sid,ts) 网格,可直接 join。【择时/大盘状态划分的合法输入】。
    ★ask1/bid1 的唯一真源就是这张表(已从 quotes 移走);load_master 会从这里把 ask1/bid1 接回大表。
    required=False:微观结构未落盘时返回 None(给 load_master 做"有则并入"的优雅退化),不报错。
    落盘脚本:data_analyst/dump_execquotes_zz2000.py --stage micro。"""
    ts_root = os.path.join(micro_root, "TimeSeries")
    if not os.path.isdir(ts_root):
        if required:
            raise SystemExit(f"!! 无 {ts_root};先跑 dump_execquotes_zz2000.py --stage micro。")
        return None
    if dates is None:
        dates = sorted(int(d) for d in os.listdir(ts_root) if d.isdigit())
    tasks = []
    for d in dates:
        dd = os.path.join(ts_root, str(d))
        if not os.path.isdir(dd):
            continue
        for s in os.listdir(dd):
            fp = os.path.join(dd, s, "microstructure.parquet")
            if os.path.exists(fp):
                tasks.append(fp)
    if not tasks:
        if required:
            raise SystemExit(f"!! 读不到 microstructure({micro_root});先跑 dump_execquotes_zz2000.py --stage micro。日期 {dates}")
        return None
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        parts = list(ex.map(pl.read_parquet, tasks))
    return _cast_keys(pl.concat(parts, how="vertical"))


# ============================== 预测 ==============================
def load_pred(horizon, model="ts", version=DEFAULT_VERSION, dates=None,
              pred_root=PRED_ROOT, pred_ts_root=PRED_TS_ROOT):
    """读单 version 单 horizon 单模型的原始预测。model="ts"(时序)/"res"(残差)。
    路径 = <root>/<version>/<h>min/test_predictions.parquet;列:date,ts,sid,pred,pred_z,y_raw。"""
    root = pred_ts_root if model == "ts" else pred_root
    fp = os.path.join(root, version, f"{horizon}min", "test_predictions.parquet")
    if not os.path.exists(fp):
        raise SystemExit(f"!! 无 {fp}(version={version});已知版本 {KNOWN_VERSIONS}")
    df = _cast_keys(pl.read_parquet(fp))
    if dates is not None:
        df = df.filter(pl.col("date").is_in(list(dates)))
    return df.sort(_KEY)


def load_preds(dates=None, horizons=None, models=("ts", "res"), version=DEFAULT_VERSION,
               pred_root=PRED_ROOT, pred_ts_root=PRED_TS_ROOT):
    """某 version 全 horizon 预测拼成【仅预测】宽表(不含 quotes),按 (date,sid,ts) 连。
    路径 = <root>/<version>/<h>min/;列:date,sid,ts + pred_ts_<h>/predz_ts_<h>/y_<h>(时序)、pred_res_<h>(残差)。"""
    ts_hs = horizons if horizons is not None else _avail_horizons(pred_ts_root, version)
    res_hs = horizons if horizons is not None else _avail_horizons(pred_root, version)
    base = None

    def _take(fp, renames):
        d = _cast_keys(pl.read_parquet(fp))
        if dates is not None:
            d = d.filter(pl.col("date").is_in(list(dates)))
        return d.select(_KEY + [pl.col(src).alias(dst) for src, dst in renames])

    if "ts" in models:
        for h in ts_hs:
            fp = os.path.join(pred_ts_root, version, f"{h}min", "test_predictions.parquet")
            if os.path.exists(fp):
                d = _take(fp, [("pred", f"pred_ts_{h}"), ("pred_z", f"predz_ts_{h}"), ("y_raw", f"y_{h}")])
                base = d if base is None else base.join(d, on=_KEY, how="outer_coalesce")
    if "res" in models:
        for h in res_hs:
            fp = os.path.join(pred_root, version, f"{h}min", "test_predictions.parquet")
            if os.path.exists(fp):
                d = _take(fp, [("pred", f"pred_res_{h}")])
                base = d if base is None else base.join(d, on=_KEY, how="outer_coalesce")
    if base is None:
        raise SystemExit(f"!! 没读到任何预测(version={version})。")
    return base


# ============================== 隔夜预测(尾盘→次日早盘,复权跨日)==============================
def load_overnight(model="resid", dates=None, overnight_root=OVERNIGHT_ROOT):
    """读隔夜预测(尾盘 30min 样本 → 次日早盘的复权隔夜收益)。model="resid"(截面中性化 label)/"ts"(原始 y_raw)。
    路径 = <overnight_root>/<model>/test_predictions.parquet;列:date,ts,sid,pred,pred_z,y_raw(列同日内 pred)。

    ⚠️ 独立口径,别并入 load_master:
      - `ts` = 全天绝对行号 **12600+**(尾盘 14:30 起、步长 10),与日内同网格但**只覆盖尾盘段**(约 12600~14210)。
      - `y_raw` = **复权隔夜收益** = 今日 D 尾盘 row=ts 的 mid → 次日 D+1 早盘 row=ts−12600 的 mid,位置对齐
        (14:45↔09:45)× accumAdjFactor(an/at 除权修正)。跨日,和日内 h 分钟前向是两套东西。
    落盘脚本:HFCT_Model/overnight_model/xgbm/train_xgb_overnight_30t10v20t.py。"""
    fp = os.path.join(overnight_root, model, "test_predictions.parquet")
    if not os.path.exists(fp):
        raise SystemExit(f"!! 无 {fp}(model={model});已知 resid / ts")
    df = _cast_keys(pl.read_parquet(fp))
    if dates is not None:
        df = df.filter(pl.col("date").is_in(list(dates)))
    return df.sort(_KEY)


# ============================== 大表(主入口)==============================
def load_master(dates=None, horizons=None, models=("ts", "res"), version=DEFAULT_VERSION, n_workers=48,
                pred_root=PRED_ROOT, pred_ts_root=PRED_TS_ROOT, quotes_root=QUOTES_ROOT,
                micro_root=MICRO_ROOT):
    """【主入口】内存里拼成一张对齐大表并返回(不落盘)。
      base = quotes 全网格(不去尾,是各 horizon 预测的前缀超集),左连全 horizon 预测。
      dates    : 日期列表(int YYYYMMDD);None=全部 test 天。
      horizons : horizon 列表;None=自动检测。
      version  : 预测版本子目录(默认 v1;新版 "lasso_0707_xgb")。
    列:date,sid,ts,ts_real,task,tbid,tmid,tavol,tbvol,vol,
        ask1,bid1(从 microstructure 接回,因果执行门禁;微观结构未落盘时缺这两列),
        pred_ts_<h>,predz_ts_<h>,y_<h>,pred_res_<h>(去尾段为 null)。"""
    q = load_quotes(dates, quotes_root, n_workers)
    p = load_preds(dates, horizons, models, version, pred_root, pred_ts_root)
    m = q.join(p, on=_KEY, how="left")
    # ask1/bid1 唯一真源 = microstructure(已从 quotes 移走)。从那里接回大表,供优化器执行门禁用,值与旧 quotes 一致。
    #   先 drop 掉旧 quotes 可能残留的 ask1/bid1(过渡期老 quotes 文件还带着)→ 防 join 列名冲突;有微观结构则并入。
    m = m.drop([c for c in ("ask1", "bid1") if c in m.columns])
    micro = load_microstructure(dates, micro_root, n_workers, required=False)
    if micro is not None:
        m = m.join(micro.select(_KEY + ["ask1", "bid1"]), on=_KEY, how="left")
    return m.sort(_KEY)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dates", type=int, nargs="*", default=None, help="日期(YYYYMMDD);默认全部 test 天")
    ap.add_argument("--horizons", type=int, nargs="*", default=None, help="horizon(分钟);默认自动检测")
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--version", default=DEFAULT_VERSION, help=f"预测版本子目录(默认 {DEFAULT_VERSION};已知 {KNOWN_VERSIONS})")
    args = ap.parse_args()

    df = load_master(args.dates, args.horizons, version=args.version, n_workers=args.workers)
    print(f"[master] rows={df.height:,}  cols={len(df.columns)}  "
          f"dates={df['date'].n_unique()}  sids={df['sid'].n_unique()}  (内存,未落盘)")
    print("[schema]")
    for c, t in df.schema.items():
        print(f"  {c:<14} {t}")
    print("[head]")
    print(df.head(5))


if __name__ == "__main__":
    main()
