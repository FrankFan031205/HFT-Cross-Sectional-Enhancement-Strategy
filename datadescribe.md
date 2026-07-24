# 所有表的数据说明
参照框架的data_loader读取的表来做说明

## 零、清洗股票逐笔说明
- **seqno**：  
  按照“**先申报、后撮合**”原则，根据**事件流顺序**统一排序, 逻辑与深交所保持一致；  
  **每个股票、每个交易日从 0 开始连续递增**。

- **price**：  
**市价单**与**本方最优价**为 0；
部分新股**限价单**价格无约束，做截断处理，最大值为 **1000000**

- **数据完整性**：  
**TLTradeSH**, **TLTradeSZ**, **TLOrderSH**, **TLOrderSZ** 全量保留，无遗漏或过滤

## 一、500ms/df_snapshot
| 字段名 | 含义 |
| ------ | ------ |
| Securityid | 股票代码 |
| timestamp | 快照时间戳 |
| bid/ask price/volume 1/2/... | 买/卖 挂单价/量 1/2/...档位, 目前10档 |
| limit_up/down_price | 当日涨跌停价格 |
| turnoverRate,negMarketValue,... | 参照优矿数据的低频字段 |

## 二、A_share_Order/df_order
| 字段名 | 类型 | 含义说明 |
|------|------|---------|
| SecurityID | int32 | 股票代码 |
| seqno | int32 | 委托事件序列号 |
| time | int32 | 委托时间 |
| price | int32 | 挂单价格 ×100 |
| qty | int32 | 挂单数量 |
| side | int8 | 买卖侧（0: buy，1: sell） |
| type | int8 | 订单类型（0: market，1: limit，2: best；深交所有三类，上交所仅有 limit） |
| timestamp | 当前逐笔归属于哪个快照时间戳 |

## 三、A_share_Trade/df_trade
| 字段名 | 类型 | 含义说明 |
|------|------|---------|
| SecurityID | int32 | 股票代码 |
| seqno | int32 | 成交事件序列号 |
| time | int32 | 成交时间 |
| price | int32 | 成交价格 |
| qty | int32 | 成交数量 |
| bidseqno | int32 | 成交中买方原始委托序列号 |
| offerseqno | int32 | 成交中卖方原始委托序列号 |
| side | int8 | 成交方向（0: buy，1: sell） |
| timestamp | 当前逐笔归属于哪个快照时间戳 |

## 四、A_share_Cancel/df_cancel
| 字段名 | 类型 | 含义说明 |
|------|------|---------|
| SecurityID | int32 | 股票代码 |
| seqno | int32 | 撤单事件序列号 |
| time | int32 | 撤单时间 |
| price | int32 | 被撤订单原始委托价格 ×100 |
| cancel_qty | int32 | 撤单时剩余数量 |
| side | int8 | 买卖侧（0: buy，1: sell） |
| order_seqno | int32 | 被撤订单的原始委托序列号 |
| order_qty | int32 | 被撤订单的原始挂单数量 |
| timestamp | 当前逐笔归属于哪个快照时间戳 |

## 五、stock_index_main_500ms_v2/df_index
股指主力合约快照数据
| 字段名 | 含义说明 |
| ------ | ------ |
| time | 快照时间戳 |
| SecurityID | 合约名（包含当日IC、IF、IM、IH四个主力合约） |
| open/high/low/last/pre_close price | 开/高/低/最新/前一个交易日最后一个 成交价 |
| bid/ask price/volume 1/2/... | 买/卖 挂单价/量 1/2/...档位, 目前5档 |
| volume | 当日累计成交量 |
| position | 当前合约持仓量 | 
| timestamp | 当前逐笔归属于哪个快照时间戳 |

## 六、ETF_Order_Test/df_etf_order
| **字段名称**      | **类型 (CH)**  | **业务含义** | **数据说明**                  | **备注**                    |
| ----------------- | -------------- | ------------ | ----------------------------- | --------------------------- |
| `time`            | int            | 逐笔时间戳   | | |
| `SecurityID`      | int            | 宽基ETF的id（目前只有510050, 510300, 510500, 512100, 563300）   | | |
| `OrderType`       | String         | 订单类型     | A: 新增委托；D: 删除/撤销委托 | 见注1                       |
| `OrderNO`         | UInt64         | 原始订单号   | 该笔订单的唯一编号            | 核心关联键，可关联逐笔成交  |
| `OrderPrice`      | Float64        | 委托价格     | 委托下单的价格                | 对应精度 MDLFloatT<3>       |
| `Balance`         | Float64        | 委托量       | 订单当前剩余的委托数量        | 见注2                       |
| `OrderBSFlag`     | String         | 买卖方向     | B: 买入；S: 卖出              | 仅针对委托订单              |
| `timestamp`       | int            |当前逐笔归属于哪个快照时间戳 |||

## 七、ETF_Trade_Test/df_etf_trade
| **字段名称**      | **类型 (CH)**  | **业务含义** | **取值及含义说明**                            | **备注**                     |
| ----------------- | -------------- | ------------ | --------------------------------------------- | ---------------------------- |
| `time`            | int            | 逐笔时间戳   | | |
| `SecurityID`      | int            | 宽基ETF的id（目前只有510050, 510300, 510500, 512100, 563300）   | | |
| `TradPrice`       | Float64        | 成交价格     | 每股/份的成交价格                             | 对应 MDLFloatT<3>            |
| `TradVolume`      | Float64        | 成交数量     | 成交的股数/份数/张数                          | 对应 MDLDoubleT<3>           |
| `TradeBuyNo`      | Int64          | 买方单号     | 匹配到的买方原始订单号                        | 关键关联键，对应 OrderNO     |
| `TradeSellNo`     | Int64          | 卖方单号     | 匹配到的卖方原始订单号                        | 关键关联键，对应 OrderNO     |
| `TradeBSFlag`     | String         | 主动方方向   | B: 主动买（外盘）；S: 主动卖（内盘）；N: 未知 | 用于判断资金流向             |
| `timestamp`       |int             | 当前逐笔归属于哪个快照时间戳 |||
