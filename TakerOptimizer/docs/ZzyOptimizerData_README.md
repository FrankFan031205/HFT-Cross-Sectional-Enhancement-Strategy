# ZZY optimizer data 接入说明

这个接入把 `/mnt/data1/zzy/optimizer_data/` 下的预测和执行 quotes 接到本项目的 `TakerOptimizer`。

核心原则：

1. 不接到 `FactorModel`。这批数据已经是模型预测 + 未来 60s 执行统计，应该作为 `TakerOptimizer` 的新数据源。
2. 对齐键必须是 `(date, sid, ts)`。`ts` 是每个 `(date, sid)` 内部的 10 秒行号，会换股/换日重置，不能单独 join。
3. `task/tbid/tmid/tavol/tbvol/vol` 是未来 60 秒统计，不是可实时看到的因子。只用于离线执行价格、成本、容量和回测，不要作为 alpha 特征。
4. 默认用 `pred_res_5` 作为第一版信号，因为残差预测通常比时序预测更适合横截面选股。后面可以系统比较 `res/ts` 和 `2/3/5/10/20/30min`。

## 安装

在项目根目录执行：

```bash
cd /mnt/data1/fwz/HFT_010-dev_fwz
# 假设补丁包解压后当前目录包含 TakerOptimizer/...
cp -r TakerOptimizer/src/zzy_optimizer_data_loader.py TakerOptimizer/src/
cp -r TakerOptimizer/scripts/prepare_zzy_optimizer_input.py TakerOptimizer/scripts/
cp -r TakerOptimizer/scripts/check_zzy_optimizer_input.py TakerOptimizer/scripts/
cp -r TakerOptimizer/config/taker_optimizer_zzy_zz2000_v1.yaml TakerOptimizer/config/
```

## 生成优化器输入

先小样本跑 2 天：

```bash
cd /mnt/data1/fwz/HFT_010-dev_fwz
python TakerOptimizer/scripts/prepare_zzy_optimizer_input.py \
  --dates 20241217 20241218 \
  --horizon 5 \
  --signal-model res \
  --include-y \
  --workers 32 \
  --participation 0.03 \
  --out-dir /mnt/data1/fwz/HFT_010-dev_fwz_outputs/TakerOptimizer/zzy_zz2000_h5_res_v1_test2d
```

检查：

```bash
python TakerOptimizer/scripts/check_zzy_optimizer_input.py \
  --input-dir /mnt/data1/fwz/HFT_010-dev_fwz_outputs/TakerOptimizer/zzy_zz2000_h5_res_v1_test2d
```

全 20 天：

```bash
DATES="20241217 20241218 20241219 20241220 20241223 20241224 20241225 20241226 20241227 20241230 20241231 20250102 20250103 20250106 20250107 20250108 20250109 20250110 20250113 20250114"
python TakerOptimizer/scripts/prepare_zzy_optimizer_input.py \
  --dates $DATES \
  --horizon 5 \
  --signal-model res \
  --include-y \
  --workers 32 \
  --participation 0.03 \
  --out-dir /mnt/data1/fwz/HFT_010-dev_fwz_outputs/TakerOptimizer/zzy_zz2000_h5_res_v1
```

输出列会包含：

- `date, sid, SecurityID, ts, ts_real`
- `pred_ret, signal_raw, signal_z`
- `exec_buy_price, exec_sell_price, exec_mid_price`
- `exec_spread_bps, buy_cost_bps, sell_cost_bps`
- `volume_60s, max_participation_shares, max_participation_notional`
- 可选 `fwd_ret_label`

## 下一步接优化器

现有 optimizer 只需要改数据读取层：把旧的 `pred_ret/fair_price/ask_price/bid_price/vol` 来源换成这里的 parquet。

建议第一版映射：

| 旧字段 | 新字段 |
|---|---|
| alpha / signal | `signal_z` 或 `pred_ret` |
| pred_ret | `pred_ret` |
| ask_price / buy execution price | `exec_buy_price` |
| bid_price / sell execution price | `exec_sell_price` |
| mid_price | `exec_mid_price` |
| spread / cost | `exec_spread_bps` |
| liquidity / capacity | `max_participation_notional` |

如果要保持 live-like，optimizer 决策只能用 `pred_ret/signal_z` 和历史可得信息；`exec_*` 只放到 backtest execution 层。
