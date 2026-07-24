import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.align_data import read_table, align_prediction_market
from backtest.maker_backtest import run_backtest


def main():
    config_path = ROOT / "config" / "pricing_config.yaml"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    pred_path = Path(config["paths"]["pred_path"])
    market_path = Path(config["paths"]["market_path"])
    output_dir = Path(config["paths"]["output_dir"])

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "quotes").mkdir(parents=True, exist_ok=True)
    (output_dir / "trades").mkdir(parents=True, exist_ok=True)
    (output_dir / "reports").mkdir(parents=True, exist_ok=True)

    pred = read_table(pred_path)
    market = read_table(market_path)

    df = align_prediction_market(pred, market, config)

    if len(df) == 0:
        raise ValueError("No aligned data. Check timestamp/date/code or merge_mode.")

    quotes, trades, metrics = run_backtest(df, config)

    quotes.to_csv(output_dir / "quotes" / "quotes.csv", index=False)
    trades.to_csv(output_dir / "trades" / "trades.csv", index=False)
    metrics.to_csv(output_dir / "reports" / "metrics.csv", index=False)

    print("Backtest finished.")
    print(metrics.T)


if __name__ == "__main__":
    main()