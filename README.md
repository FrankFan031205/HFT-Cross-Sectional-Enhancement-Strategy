# HFT Cross-Sectional Enhancement Strategy

This repository contains the code handoff for the HFT cross-sectional enhancement strategy project.

## Main Modules

- AlphaResearch: single-factor and alpha research tools
- FactorGenerate: high-frequency factor generation
- FactorModel: ML / DL factor models
- PricingModel: fair-price and edge estimation
- MarketMakingModel: quote generation and market-making strategy
- BacktestingModel: fill simulation and PnL backtesting
- CrossSectionalModel: cross-sectional signal modelling
- CrossSectionalOptimizer: early cross-sectional optimizer
- TakerOptimizer: pure-CS / taker position optimizer
- TakerModel: NAV evaluation, futures overlay, overnight module
- TakerPipeline: OOS inference and pipeline scripts

## Notes

This repository is code-only. Data, model outputs, logs, large backtest results, and raw market files are intentionally excluded.

The latest pure-CS main version is v18. The enhanced overlay version is v21.
