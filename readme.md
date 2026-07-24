# HFT_010-dev_fwz

### 1. Project Overview

`HFT_010-dev_fwz` is a research-oriented high-frequency trading project for China A-share equities.

The project covers the full research pipeline from microstructure factor generation, machine-learning factor modeling, market-making quote construction, to execution-aware and inventory-aware backtesting.

The repository is organized into five major modules:

```text
FactorGenerate      -> Generate and evaluate high-frequency microstructure factors
FactorModel         -> Train ML / deep learning models and generate hidden factors
PricingModel        -> Prepare pricing, market return, and markout datasets
MarketMakingModel   -> Convert model predictions into quote decisions
BacktestingModel    -> Simulate fills, costs, inventory constraints, and portfolio replay
```

The end-to-end idea is:

```text
raw market data
    ↓
microstructure factors
    ↓
factor model / hidden factor
    ↓
quote decision
    ↓
execution-aware fill simulation
    ↓
inventory-aware portfolio backtest
```

This project is intended for research and experimentation. It is not a production trading system or a live exchange simulator.

---

### 2. Research Objective

The main objective is to study whether high-frequency microstructure signals can be transformed into profitable market-making decisions under realistic execution assumptions.

The core research questions are:

1. Which order book, trade-flow, order-flow, and cancellation factors have short-horizon predictive power?
2. Can machine-learning models combine weak microstructure factors into stronger hidden signals?
3. Can predicted short-horizon returns be converted into maker-style bid / ask quote decisions?
4. How much performance remains after queue priority, transaction costs, T+1 inventory constraints, and no-short constraints?
5. Which factors or models provide the best trade-off among PnL, turnover, exposure, drawdown, and capital efficiency?

---

### 3. Repository Structure

```text
HFT_010-dev_fwz/
├── FactorGenerate/
│   ├── config/
│   ├── data/
│   ├── logs/
│   ├── outputs/
│   ├── scripts/
│   ├── src/
│   └── README.md
│
├── FactorModel/
│   ├── config/
│   ├── data/
│   ├── logs/
│   ├── models/
│   ├── notebooks/
│   ├── outputs/
│   ├── scripts/
│   ├── src/
│   └── README.md
│
├── PricingModel/
│   ├── config/
│   ├── data/
│   ├── outputs/
│   ├── scripts/
│   ├── src/
│   └── README.md
│
├── MarketMakingModel/
│   ├── config/
│   ├── outputs/
│   ├── scripts/
│   ├── src/
│   └── README.md
│
├── BacktestingModel/
│   ├── config/
│   ├── data/
│   ├── logs/
│   ├── outputs/
│   ├── scripts/
│   ├── src/
│   └── README.md
│
├── datadescribe.md
├── environment.yml
├── export_market_csv.py
└── README.md
```

---

### 4. End-to-End Workflow

The complete workflow is:

```text
Step 1: Generate microstructure factors
        Module: FactorGenerate

Step 2: Evaluate factor predictiveness
        Metrics: IC / RankIC / ICIR / long-short return / horizon sensitivity

Step 3: Train factor models
        Module: FactorModel

Step 4: Generate hidden factor CSV
        Example: hidden_factor_mlp2_h60_202410_100.csv

Step 5: Generate quote decisions
        Module: MarketMakingModel

Step 6: Prepare pricing and markout data
        Module: PricingModel

Step 7: Run execution-aware backtest
        Module: BacktestingModel

Step 8: Compare factor models
        Output: portfolio-level summary metrics
```

A simplified data flow is:

```text
ClickHouse raw data
    ├── Snapshot
    ├── Trade
    ├── Order
    └── Cancel
        ↓
FactorGenerate
        ↓
factor_features_202410_100.csv
        ↓
FactorModel
        ↓
hidden_factor_xxx_h60_202410_100.csv
        ↓
MarketMakingModel
        ↓
quote_decisions_xxx_h60_202410_100.csv
        ↓
BacktestingModel
        ↓
portfolio_replay_xxx_summary.csv
```

---

### 5. Module Descriptions

#### 5.1 FactorGenerate

`FactorGenerate` is responsible for generating and evaluating high-frequency microstructure factors.

It uses raw China A-share market data, including:

```text
Snapshot data
Trade data
Order data
Cancel data
```

Typical factor categories include:

```text
order book imbalance
weighted order book imbalance
depth imbalance
book slope imbalance
microprice deviation
relative spread
short-horizon return
trade imbalance
active buy / sell pressure
trade intensity
large trade imbalance
order imbalance
aggressive order imbalance
cancel pressure
near-price cancel pressure
cancel ratio imbalance
```

Common evaluation metrics include:

```text
IC
RankIC
ICIR
RankICIR
cross-sectional long-short return
quantile return
horizon sensitivity
```

Typical commands:

```bash
cd /home/fwz/projects/HFT_010-dev_fwz/FactorGenerate

python scripts/multi_factor_eval.py   --start 20241001   --end 20241031   --factors all   --sample 100   --horizons 30,60,90,120   --bins 5   --output outputs/multi_factor_eval_202410_100.csv
```

To export factor features for modeling:

```bash
python scripts/dump_factor_features.py   --start 20241022   --end 20241031   --factors all   --sample 100   --horizons 30,60,90,120   --output ../FactorModel/data/raw/factor_features_202410_100.csv
```

---

#### 5.2 FactorModel

`FactorModel` trains machine-learning or deep-learning models on generated microstructure factors.

Typical input:

```text
FactorModel/data/raw/factor_features_202410_100.csv
```

Typical preprocessing:

```text
replace infinite values
fill missing values
winsorize / clip outliers
cross-sectional z-score normalization by datetime
train / validation / test split
```

Supported or planned models include:

```text
Ridge
LightGBM
MLP
Attention model
other neural networks
```

Typical outputs:

```text
FactorModel/outputs/hidden_factor_lgbm_h60_202410_100.csv
FactorModel/outputs/hidden_factor_mlp2_h60_202410_100.csv
FactorModel/outputs/hidden_factor_attention_h60_202410_100.csv
```

These hidden factor files are used by downstream modules as model signals.

---

#### 5.3 PricingModel

`PricingModel` prepares pricing and markout datasets used by market-making and backtesting modules.

Typical outputs:

```text
PricingModel/data/market_return_202410_100.csv
PricingModel/data/pricing_dataset_h60_202410_100_full.csv
```

The pricing dataset contains:

```text
datetime
securityid
bid1 ... bid10
ask1 ... ask10
bid1_volume ... bid10_volume
ask1_volume ... ask10_volume
mid_price
spread
spread_ticks
future_mid / label columns
```

Main usage:

```text
calculate quote prices
evaluate future mid-price markout
merge order book state into backtests
```

---

#### 5.4 MarketMakingModel

`MarketMakingModel` converts model signals into bid / ask quote decisions.

Typical output:

```text
MarketMakingModel/outputs/quote_decisions/quote_decisions_mlp2_h60_202410_100.csv
```

A quote decision file usually contains:

```text
datetime
securityid
bid1
ask1
mid_price
spread
hidden_factor_xxx
raw_pred
pred_used
fair_price
quote_bid
quote_ask
bid_price
ask_price
bid_size
ask_size
bid_edge
ask_edge
risk_state
quote_style
```

The market-making logic usually considers:

```text
predicted alpha
spread
transaction cost
adverse selection buffer
inventory state
risk regime
quote threshold
```

Important distinction:

```text
MarketMakingModel generates quote decisions.
BacktestingModel evaluates quote decisions.
```

---

#### 5.5 BacktestingModel

`BacktestingModel` is an execution-aware backtesting framework.

It evaluates fixed quote decisions and simulates:

```text
touched-fill
queue-aware fill
transaction costs
60s markout PnL
policy filtering
A-share no-short constraint
T+1 sellable inventory
initial inventory pool
max position cap
inventory-aware execution overlay
cash / position / equity portfolio replay
exposure / drawdown / capital efficiency analysis
```

Current final baseline:

```text
fill model: queue-aware
queue_ahead_multiplier: 0.05
policy: abs_top40
initial inventory: 5000 shares per stock
max position: 15000 shares per stock
sell floor: 3000 shares
buy block: 9000 shares
T+1: enabled
short selling: disabled
```

Current final baseline result:

```text
num_trades                     75,705
total_pnl                      +557,231.94
total_turnover                 172.38M
pnl_bps_on_turnover            32.33 bps
max_gross_exposure             115.82M
return_on_max_gross_exposure   0.481%
max_drawdown                   -261,445
short violations               0
```

This is the current recommended realistic inventory-aware backtest setup.

---

### 6. New Factor Backtest Workflow

The project supports generalized new-factor backtesting.

For a new factor, the required inputs are:

```text
tag
quote_decision_path
signal_col
```

Example:

```text
tag: lookback_attention_h60_202410_100
quote_decision_path: ../MarketMakingModel/outputs/quote_decisions/quote_decisions_lookback_attention_h60_202410_100.csv
signal_col: hidden_factor_lookback_attention_h60
```

Recommended command:

```bash
cd /home/fwz/projects/HFT_010-dev_fwz/BacktestingModel

python scripts/register_new_factor_backtest.py   --tag lookback_attention_h60_202410_100   --quote-path ../MarketMakingModel/outputs/quote_decisions/quote_decisions_lookback_attention_h60_202410_100.csv   --signal-col hidden_factor_lookback_attention_h60   --overwrite   --build   --sample   --max-quotes 5000
```

If the small sample succeeds, run the full backtest:

```bash
nohup python scripts/run_backtest_experiment.py   --exp config/experiments/generated/lookback_attention_h60_202410_100.yaml   > logs/lookback_attention_h60_202410_100_backtest_experiment.log 2>&1 &
```

Monitor progress:

```bash
tail -f logs/lookback_attention_h60_202410_100_backtest_experiment.log
```

Summarize all experiments:

```bash
python scripts/summarize_all_experiments.py
```

The comparison output is:

```text
BacktestingModel/outputs/portfolio/all_experiments_summary.csv
```

Key comparison metrics:

```text
total_pnl
total_turnover
pnl_bps_on_turnover
max_gross_exposure
return_on_max_gross_exposure
max_drawdown
num_short_events
num_short_violations_if_no_short
```

---

### 7. Data Sources

The project uses ClickHouse data sources including:

```text
A_share_Trade
A_share_Order
A_share_Cancel
500ms snapshot tables
```

Common fields:

```text
Snapshot:
    time
    SecurityID
    bidprice1 ... bidprice10
    askprice1 ... askprice10
    bidvolume1 ... bidvolume10
    askvolume1 ... askvolume10

Trade:
    time
    SecurityID
    seqno
    price
    qty
    bidseqno
    offerseqno
    side

Order:
    time
    SecurityID
    seqno
    price
    qty
    side
    type

Cancel:
    time
    SecurityID
    seqno
    price
    cancel_qty
    side
    order_seqno
    order_qty
```

Known conventions:

```text
trade side = 0 means active buy
trade side = 1 means active sell
trade price scale = 100
```

Therefore:

```text
trade_price = price / 100
```

---

### 8. Environment

The project is usually run under the conda environment:

```text
hft_py39
```

Activate environment:

```bash
conda activate hft_py39
```

Common Python packages:

```text
pandas
polars
numpy
scikit-learn
lightgbm
xgboost
torch
pyyaml
clickhouse-connect
matplotlib
```

If using `environment.yml`:

```bash
conda env update -f environment.yml
```

---

### 9. Recommended Daily Workflow

A typical research workflow is:

```text
1. Generate or update factors in FactorGenerate
2. Evaluate factor IC and long-short return
3. Train or update FactorModel
4. Generate hidden factor CSV
5. Generate quote decisions in MarketMakingModel
6. Register the quote decision in BacktestingModel
7. Run small-sample backtest
8. Run full backtest in background
9. Summarize all experiments
10. Compare PnL, turnover, exposure, and drawdown
```

---

### 10. Common Commands

Check long-running Python jobs:

```bash
ps -ef | grep python | grep -v grep
```

Check specific backtest jobs:

```bash
ps -ef | grep -E "run_backtest_experiment|run_fill_simulation|enrich_fills|run_attention_pnl_backtest|run_portfolio_replay" | grep -v grep
```

Run a backtest in background:

```bash
nohup python scripts/run_backtest_experiment.py   --exp config/experiments/generated/mlp2_h60_202410_100.yaml   > logs/mlp2_h60_202410_100_backtest_experiment.log 2>&1 &
```

Monitor logs:

```bash
tail -f logs/mlp2_h60_202410_100_backtest_experiment.log
```

---

### 11. Current Limitations

This project is a research-grade system, not a production trading system.

Current limitations include:

```text
latency-aware order arrival is simplified
order lifecycle is simplified
cancel-adjusted queue movement is not fully modeled
partial fill and order size sensitivity can be improved
end-of-day liquidation is not fully modeled
overnight inventory handling is simplified
capital allocation is approximated by max gross exposure
live trading risk checks are not implemented
```

Future improvements:

```text
latency-aware fill simulation
cancel-adjusted queue model
order lifecycle with keep / cancel / amend
end-of-day de-risking
multi-horizon markout analysis
more realistic capital allocation
portfolio-level risk constraints
```

---

### 13. One-sentence Summary

`HFT_010-dev_fwz` is a modular high-frequency market-making research pipeline that starts from raw A-share microstructure data, builds and evaluates predictive factors, trains hidden-factor models, converts predictions into quote decisions, and evaluates them through execution-aware and inventory-aware backtesting.

---

# Chinese Version

## 1. 项目概览

`HFT_010-dev_fwz` 是一个面向中国 A 股高频做市研究的项目。

项目覆盖从微观结构因子生成、机器学习因子建模、做市报价生成，到执行感知回测和库存约束回测的完整研究流程。

主要模块包括：

```text
FactorGenerate      -> 生成和评估高频微观结构因子
FactorModel         -> 训练机器学习 / 深度学习模型并生成 hidden factor
PricingModel        -> 准备定价和 markout 数据
MarketMakingModel   -> 将模型预测值转化为 quote decision
BacktestingModel    -> 模拟成交、费用、库存约束和组合回放
```

整体流程是：

```text
原始市场数据
    ↓
微观结构因子
    ↓
因子模型 / hidden factor
    ↓
quote decision
    ↓
成交模拟
    ↓
库存感知组合回测
```

本项目用于研究和实验，不是实盘交易系统，也不是生产级交易所撮合模拟器。

---

## 2. 研究目标

本项目的核心目标是研究：高频微观结构信号能否在真实执行约束下转化为有效的做市报价。

主要研究问题包括：

1. 哪些盘口、成交、委托、撤单因子具有短周期预测能力？
2. 机器学习模型能否把弱因子组合成更强的 hidden factor？
3. 短周期收益预测能否转化为 bid / ask quote decision？
4. 考虑 queue priority、交易费用、T+1、不可裸卖空后，策略是否仍然有效？
5. 哪些因子或模型在 PnL、turnover、exposure、drawdown 之间表现最均衡？

---

## 3. 项目结构

```text
HFT_010-dev_fwz/
├── FactorGenerate/
├── FactorModel/
├── PricingModel/
├── MarketMakingModel/
├── BacktestingModel/
├── datadescribe.md
├── environment.yml
├── export_market_csv.py
└── README.md
```

各模块职责如下：

```text
FactorGenerate:
    因子生成和因子评价

FactorModel:
    机器学习 / 深度学习建模，生成 hidden factor

PricingModel:
    准备 mid price、future mid、盘口和 markout 数据

MarketMakingModel:
    根据 hidden factor 生成 quote decision

BacktestingModel:
    对 quote decision 做成交模拟、PnL 评估和库存感知组合回放
```

---

## 4. 端到端流程

完整研究流程：

```text
Step 1: FactorGenerate 生成微观结构因子

Step 2: 评估 IC / RankIC / long-short return

Step 3: FactorModel 训练模型

Step 4: 生成 hidden_factor_xxx_h60_202410_100.csv

Step 5: MarketMakingModel 生成 quote_decisions_xxx_h60_202410_100.csv

Step 6: PricingModel 准备 pricing / markout 数据

Step 7: BacktestingModel 做执行感知回测

Step 8: 对不同因子 / 模型做横向比较
```

简化数据流：

```text
ClickHouse 原始数据
    ├── Snapshot
    ├── Trade
    ├── Order
    └── Cancel
        ↓
FactorGenerate
        ↓
factor_features_202410_100.csv
        ↓
FactorModel
        ↓
hidden_factor_xxx_h60_202410_100.csv
        ↓
MarketMakingModel
        ↓
quote_decisions_xxx_h60_202410_100.csv
        ↓
BacktestingModel
        ↓
portfolio_replay_xxx_summary.csv
```

---

## 5. 各模块说明

### 5.1 FactorGenerate

`FactorGenerate` 负责生成和评估高频微观结构因子。

使用的数据包括：

```text
Snapshot 行情快照
Trade 成交数据
Order 委托数据
Cancel 撤单数据
```

典型因子包括：

```text
盘口不平衡
加权盘口不平衡
深度不平衡
microprice deviation
相对 spread
短周期 return
主动买卖不平衡
成交强度
大单成交不平衡
委托不平衡
主动委托不平衡
撤单压力
近价撤单压力
撤单比例不平衡
```

常用评价指标：

```text
IC
RankIC
ICIR
RankICIR
分组 long-short return
quantile return
不同预测周期下的稳定性
```

典型命令：

```bash
cd /home/fwz/projects/HFT_010-dev_fwz/FactorGenerate

python scripts/multi_factor_eval.py   --start 20241001   --end 20241031   --factors all   --sample 100   --horizons 30,60,90,120   --bins 5   --output outputs/multi_factor_eval_202410_100.csv
```

---

### 5.2 FactorModel

`FactorModel` 负责使用因子数据训练机器学习或深度学习模型。

典型输入：

```text
FactorModel/data/raw/factor_features_202410_100.csv
```

常见处理步骤：

```text
替换 inf
填充缺失值
去极值 / clipping
按 datetime 做横截面 z-score 标准化
划分 train / valid / test
```

支持或计划支持的模型包括：

```text
Ridge
LightGBM
MLP
Attention model
其他神经网络模型
```

典型输出：

```text
FactorModel/outputs/hidden_factor_lgbm_h60_202410_100.csv
FactorModel/outputs/hidden_factor_mlp2_h60_202410_100.csv
FactorModel/outputs/hidden_factor_attention_h60_202410_100.csv
```

这些 hidden factor 会作为后续做市和回测的信号输入。

---

### 5.3 PricingModel

`PricingModel` 负责准备定价和 markout 数据。

典型输出：

```text
PricingModel/data/market_return_202410_100.csv
PricingModel/data/pricing_dataset_h60_202410_100_full.csv
```

这些数据包括：

```text
datetime
securityid
bid1 ... bid10
ask1 ... ask10
bid1_volume ... bid10_volume
ask1_volume ... ask10_volume
mid_price
spread
spread_ticks
future_mid / label
```

用途包括：

```text
计算 quote price
评估 future mid markout
为回测合并盘口状态
```

---

### 5.4 MarketMakingModel

`MarketMakingModel` 负责把模型信号转化为 bid / ask quote decision。

典型输出：

```text
MarketMakingModel/outputs/quote_decisions/quote_decisions_mlp2_h60_202410_100.csv
```

quote decision 文件通常包含：

```text
datetime
securityid
bid1
ask1
mid_price
spread
hidden_factor_xxx
raw_pred
pred_used
fair_price
quote_bid
quote_ask
bid_price
ask_price
bid_size
ask_size
bid_edge
ask_edge
risk_state
quote_style
```

重要区别：

```text
MarketMakingModel 负责生成 quote decision
BacktestingModel 负责评估 quote decision
```

---

### 5.5 BacktestingModel

`BacktestingModel` 是执行感知回测框架。

它评估固定的 quote decision，并模拟：

```text
touched-fill
queue-aware fill
交易费用
60s markout PnL
policy filter
A 股不可裸卖空
T+1 可卖库存
初始库存池
单票最大持仓限制
库存感知成交过滤
cash / position / equity 组合回放
exposure / drawdown / capital efficiency
```

当前最终 baseline：

```text
fill model: queue-aware
queue_ahead_multiplier: 0.05
policy: abs_top40
initial inventory: 每只股票 5000 股
max position: 每只股票 15000 股
sell floor: 3000 股
buy block: 9000 股
T+1: 开启
short selling: 禁止
```

当前最终结果：

```text
num_trades                     75,705
total_pnl                      +557,231.94
total_turnover                 172.38M
pnl_bps_on_turnover            32.33 bps
max_gross_exposure             115.82M
return_on_max_gross_exposure   0.481%
max_drawdown                   -261,445
short violations               0
```

这版是目前最推荐的 realistic inventory-aware backtest baseline。

---

## 6. 新因子回测流程

对于一个新因子，需要提供：

```text
tag
quote_decision_path
signal_col
```

例如：

```text
tag: lookback_attention_h60_202410_100
quote_decision_path: ../MarketMakingModel/outputs/quote_decisions/quote_decisions_lookback_attention_h60_202410_100.csv
signal_col: hidden_factor_lookback_attention_h60
```

推荐命令：

```bash
cd /home/fwz/projects/HFT_010-dev_fwz/BacktestingModel

python scripts/register_new_factor_backtest.py   --tag lookback_attention_h60_202410_100   --quote-path ../MarketMakingModel/outputs/quote_decisions/quote_decisions_lookback_attention_h60_202410_100.csv   --signal-col hidden_factor_lookback_attention_h60   --overwrite   --build   --sample   --max-quotes 5000
```

如果小样本通过，再跑全量：

```bash
nohup python scripts/run_backtest_experiment.py   --exp config/experiments/generated/lookback_attention_h60_202410_100.yaml   > logs/lookback_attention_h60_202410_100_backtest_experiment.log 2>&1 &
```

查看进度：

```bash
tail -f logs/lookback_attention_h60_202410_100_backtest_experiment.log
```

汇总所有实验：

```bash
python scripts/summarize_all_experiments.py
```

---

## 7. 数据源说明

项目主要使用 ClickHouse 中的 A 股高频数据：

```text
A_share_Trade
A_share_Order
A_share_Cancel
500ms snapshot tables
```

常见字段：

```text
Snapshot:
    bidprice1 ... bidprice10
    askprice1 ... askprice10
    bidvolume1 ... bidvolume10
    askvolume1 ... askvolume10

Trade:
    time
    SecurityID
    seqno
    price
    qty
    bidseqno
    offerseqno
    side
```

当前约定：

```text
trade side = 0 表示主动买
trade side = 1 表示主动卖
trade price scale = 100
```

因此：

```text
trade_price = price / 100
```

---

## 8. 环境

常用 conda 环境：

```text
hft_py39
```

激活环境：

```bash
conda activate hft_py39
```

常用 Python 包：

```text
pandas
polars
numpy
scikit-learn
lightgbm
xgboost
torch
pyyaml
clickhouse-connect
matplotlib
```

---

## 9. 推荐日常工作流

```text
1. 在 FactorGenerate 中生成或更新因子
2. 评估因子的 IC 和 long-short return
3. 在 FactorModel 中训练模型
4. 生成 hidden factor CSV
5. 在 MarketMakingModel 中生成 quote decision
6. 在 BacktestingModel 中注册新 quote decision
7. 小样本回测
8. 后台全量回测
9. 汇总所有 experiment
10. 比较 PnL、turnover、exposure、drawdown
```

---

## 10. 常用命令

查看 Python 进程：

```bash
ps -ef | grep python | grep -v grep
```

查看回测相关进程：

```bash
ps -ef | grep -E "run_backtest_experiment|run_fill_simulation|enrich_fills|run_attention_pnl_backtest|run_portfolio_replay" | grep -v grep
```

后台运行：

```bash
nohup python scripts/run_backtest_experiment.py   --exp config/experiments/generated/mlp2_h60_202410_100.yaml   > logs/mlp2_h60_202410_100_backtest_experiment.log 2>&1 &
```

查看日志：

```bash
tail -f logs/mlp2_h60_202410_100_backtest_experiment.log
```

---

## 11. 当前限制

本项目是 research-grade system，不是 production trading system。

当前限制包括：

```text
latency-aware order arrival 仍然较简化
order lifecycle 仍然较简化
cancel-adjusted queue movement 尚未完全建模
partial fill 和 order size sensitivity 可以继续优化
end-of-day liquidation 尚未完全建模
overnight inventory handling 仍然简化
capital allocation 主要用 max gross exposure 近似
尚未接入实盘风控
```

后续可优化方向：

```text
latency-aware fill simulation
cancel-adjusted queue model
order lifecycle with keep / cancel / amend
end-of-day de-risking
multi-horizon markout analysis
更真实的 capital allocation
portfolio-level risk constraints
```

---

## 13. 总结

`HFT_010-dev_fwz` 是一个模块化的高频做市研究管线：从 A 股微观结构原始数据出发，生成和评估因子，训练 hidden-factor 模型，将预测值转化为 quote decision，并通过执行感知和库存感知回测评估策略在更接近真实交易约束下的表现。
