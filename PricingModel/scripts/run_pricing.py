import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def read_table(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix == ".csv":
        return pd.read_csv(path)

    if path.suffix == ".parquet":
        return pd.read_parquet(path)

    raise ValueError(f"Unsupported file type: {path}")


def calc_linear_calibration(df, signal_col, target_col, split_col, calibrate_splits):
    train = df[df[split_col].isin(calibrate_splits)].copy()
    train = train[[signal_col, target_col]].replace([np.inf, -np.inf], np.nan).dropna()

    if len(train) == 0:
        raise ValueError(f"No valid calibration data for {signal_col}")

    x = train[signal_col].values
    y = train[target_col].values

    x_mean = x.mean()
    y_mean = y.mean()

    var_x = ((x - x_mean) ** 2).mean()

    if var_x == 0:
        raise ValueError(f"Signal variance is zero for {signal_col}")

    beta = ((x - x_mean) * (y - y_mean)).mean() / var_x
    intercept = y_mean - beta * x_mean

    return intercept, beta, len(train)


def calc_fair_price(mid_price, pred_ret, return_type):
    if return_type == "simple":
        return mid_price * (1 + pred_ret)

    if return_type == "log":
        return mid_price * np.exp(pred_ret)

    raise ValueError(f"Unknown return_type: {return_type}")


def calc_metrics_one(df, name, pred_col, signal_col, target_col, split_col):
    rows = []

    for split, g in df.groupby(split_col):
        tmp = g[[signal_col, pred_col, target_col]].replace([np.inf, -np.inf], np.nan).dropna()

        if len(tmp) == 0:
            continue

        signal_rank_ic = tmp[signal_col].corr(tmp[target_col], method="spearman")
        pred_rank_ic = tmp[pred_col].corr(tmp[target_col], method="spearman")
        pred_pearson_ic = tmp[pred_col].corr(tmp[target_col], method="pearson")

        err = tmp[pred_col] - tmp[target_col]

        rows.append(
            {
                "signal": name,
                "split": split,
                "n": len(tmp),
                "signal_rank_ic": signal_rank_ic,
                "pred_ret_rank_ic": pred_rank_ic,
                "pred_ret_pearson_ic": pred_pearson_ic,
                "mse": (err ** 2).mean(),
                "mae": err.abs().mean(),
                "pred_ret_mean": tmp[pred_col].mean(),
                "pred_ret_std": tmp[pred_col].std(),
                "target_mean": tmp[target_col].mean(),
                "target_std": tmp[target_col].std(),
            }
        )

    return rows


def calc_metrics_ensemble(df, pred_col, target_col, split_col):
    rows = []

    for split, g in df.groupby(split_col):
        tmp = g[[pred_col, target_col]].replace([np.inf, -np.inf], np.nan).dropna()

        if len(tmp) == 0:
            continue

        pred_rank_ic = tmp[pred_col].corr(tmp[target_col], method="spearman")
        pred_pearson_ic = tmp[pred_col].corr(tmp[target_col], method="pearson")

        err = tmp[pred_col] - tmp[target_col]

        rows.append(
            {
                "signal": "ensemble",
                "split": split,
                "n": len(tmp),
                "signal_rank_ic": np.nan,
                "pred_ret_rank_ic": pred_rank_ic,
                "pred_ret_pearson_ic": pred_pearson_ic,
                "mse": (err ** 2).mean(),
                "mae": err.abs().mean(),
                "pred_ret_mean": tmp[pred_col].mean(),
                "pred_ret_std": tmp[pred_col].std(),
                "target_mean": tmp[target_col].mean(),
                "target_std": tmp[target_col].std(),
            }
        )

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(ROOT / "config" / "pricing_config.yaml"),
        help="Path to pricing config yaml",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path

    print("using config:", config_path)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    path_cfg = config["paths"]
    data_cfg = config["data"]
    pricing_cfg = config["pricing"]
    calib_cfg = config["calibration"]
    ensemble_cfg = config["ensemble"]

    input_path = ROOT / path_cfg["pricing_dataset_path"]
    output_dir = ROOT / path_cfg["output_dir"]

    output_pricing_dir = output_dir / "pricing"
    output_report_dir = output_dir / "reports"

    output_pricing_dir.mkdir(parents=True, exist_ok=True)
    output_report_dir.mkdir(parents=True, exist_ok=True)

    df = read_table(input_path)

    date_col = data_cfg["date_col"]
    code_col = data_cfg["code_col"]
    timestamp_col = data_cfg["timestamp_col"]
    mid_col = data_cfg["mid_col"]
    target_col = data_cfg["target_col"]
    split_col = data_cfg["split_col"]
    signals = data_cfg["signals"]

    required_cols = [
        date_col,
        code_col,
        timestamp_col,
        mid_col,
        target_col,
        split_col,
    ]

    for c in required_cols:
        if c not in df.columns:
            raise KeyError(f"Missing column: {c}")

    df = df.copy()

    df = df.rename(
        columns={
            date_col: "date",
            code_col: "securityid",
            timestamp_col: "datetime",
            mid_col: "mid_price",
            target_col: "target_ret",
            split_col: "split",
        }
    )

    df["securityid"] = df["securityid"].astype(str).str.zfill(6)

    return_type = pricing_cfg.get("return_type", "simple")
    tick_size = pricing_cfg.get("tick_size", 0.01)
    calibrate_enabled = calib_cfg.get("enabled", True)
    calibrate_splits = calib_cfg.get("calibrate_splits", ["train", "valid"])

    priced = df[
        [
            "date",
            "datetime",
            "securityid",
            "split",
            "mid_price",
            "target_ret",
        ]
    ].copy()

    if "bid1" in df.columns:
        priced["bid1"] = df["bid1"]

    if "ask1" in df.columns:
        priced["ask1"] = df["ask1"]

    calibration_rows = []
    metric_rows = []
    pred_cols_for_ensemble = []
    valid_signal_names = []

    for sig in signals:
        name = sig["name"]
        raw_col = sig["col"]

        if raw_col not in df.columns:
            print(f"Skip missing signal: {raw_col}")
            continue

        signal_col = f"signal_{name}"
        pred_col = f"pred_ret_{name}"
        fair_col = f"fair_price_{name}"
        alpha_price_col = f"alpha_price_{name}"
        alpha_ticks_col = f"alpha_ticks_{name}"

        priced[signal_col] = df[raw_col]

        tmp = pd.DataFrame(
            {
                signal_col: df[raw_col],
                "target_ret": df["target_ret"],
                "split": df["split"],
            }
        )

        if calibrate_enabled:
            intercept, beta, n_calib = calc_linear_calibration(
                df=tmp,
                signal_col=signal_col,
                target_col="target_ret",
                split_col="split",
                calibrate_splits=calibrate_splits,
            )
        else:
            intercept = 0.0
            beta = 1.0
            n_calib = len(tmp)

        priced[pred_col] = intercept + beta * priced[signal_col]
        priced[fair_col] = calc_fair_price(
            mid_price=priced["mid_price"],
            pred_ret=priced[pred_col],
            return_type=return_type,
        )
        priced[alpha_price_col] = priced[fair_col] - priced["mid_price"]
        priced[alpha_ticks_col] = priced[alpha_price_col] / tick_size

        calibration_rows.append(
            {
                "signal": name,
                "raw_col": raw_col,
                "calibrate": calibrate_enabled,
                "calibrate_splits": ",".join(calibrate_splits),
                "intercept": intercept,
                "beta": beta,
                "n_calibration": n_calib,
            }
        )

        metric_df = priced[[signal_col, pred_col, "target_ret", "split"]].copy()
        metric_rows.extend(
            calc_metrics_one(
                df=metric_df,
                name=name,
                pred_col=pred_col,
                signal_col=signal_col,
                target_col="target_ret",
                split_col="split",
            )
        )

        pred_cols_for_ensemble.append(pred_col)
        valid_signal_names.append(name)

    if len(pred_cols_for_ensemble) == 0:
        raise ValueError("No valid signals found.")

    weight_rows = []

    if ensemble_cfg.get("enabled", True):
        method = ensemble_cfg.get("method", "equal_weight")

        if method == "equal_weight":
            weight = 1.0 / len(pred_cols_for_ensemble)
            weights = {col: weight for col in pred_cols_for_ensemble}
        else:
            raise ValueError(f"Unsupported ensemble method: {method}")

        priced["ensemble_pred_ret"] = 0.0

        for name, col in zip(valid_signal_names, pred_cols_for_ensemble):
            w = weights[col]
            priced["ensemble_pred_ret"] += w * priced[col]

            weight_rows.append(
                {
                    "signal": name,
                    "pred_col": col,
                    "weight": w,
                    "method": method,
                }
            )

        priced["ensemble_fair_price"] = calc_fair_price(
            mid_price=priced["mid_price"],
            pred_ret=priced["ensemble_pred_ret"],
            return_type=return_type,
        )
        priced["ensemble_alpha_price"] = priced["ensemble_fair_price"] - priced["mid_price"]
        priced["ensemble_alpha_ticks"] = priced["ensemble_alpha_price"] / tick_size

        metric_rows.extend(
            calc_metrics_ensemble(
                df=priced,
                pred_col="ensemble_pred_ret",
                target_col="target_ret",
                split_col="split",
            )
        )

    calibration = pd.DataFrame(calibration_rows)
    metrics = pd.DataFrame(metric_rows)
    weights = pd.DataFrame(weight_rows)

    priced.to_csv(output_pricing_dir / "priced_dataset.csv", index=False)
    metrics.to_csv(output_report_dir / "pricing_metrics_by_signal.csv", index=False)
    calibration.to_csv(output_report_dir / "pricing_calibration.csv", index=False)
    weights.to_csv(output_report_dir / "pricing_ensemble_weights.csv", index=False)

    print("Pricing finished.")
    print("saved:", output_pricing_dir / "priced_dataset.csv")
    print("saved:", output_report_dir / "pricing_metrics_by_signal.csv")
    print("saved:", output_report_dir / "pricing_calibration.csv")
    print("saved:", output_report_dir / "pricing_ensemble_weights.csv")
    print()
    print("Calibration:")
    print(calibration)
    print()
    print("Metrics:")
    print(metrics)


if __name__ == "__main__":
    main()