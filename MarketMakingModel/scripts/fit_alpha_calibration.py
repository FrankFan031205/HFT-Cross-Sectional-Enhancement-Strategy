import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--pred-col", required=True)
    p.add_argument("--target-col", default="label_60")
    p.add_argument("--datetime-col", default="datetime")
    p.add_argument("--chunksize", type=int, default=500000)
    p.add_argument("--train-start", default=None)
    p.add_argument("--train-end", default=None)
    p.add_argument("--max-fair-shift-ticks", type=float, default=5.0)
    return p.parse_args()


def filter_date(df, datetime_col, start, end):
    if start is None and end is None:
        return df
    if datetime_col not in df.columns:
        return df

    dt = pd.to_datetime(df[datetime_col], errors="coerce")
    mask = pd.Series(True, index=df.index)

    if start is not None:
        mask &= dt >= pd.to_datetime(start)

    if end is not None:
        mask &= dt <= pd.to_datetime(end)

    return df.loc[mask]


def main():
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(input_path)

    header = pd.read_csv(input_path, nrows=0)
    cols = list(header.columns)

    missing = [c for c in [args.pred_col, args.target_col] if c not in cols]
    if missing:
        raise ValueError(
            f"missing columns: {missing}\n"
            f"available columns: {cols}\n"
            f"Calibration needs pred_col and target_col."
        )

    usecols = [args.pred_col, args.target_col]
    if args.datetime_col in cols:
        usecols.append(args.datetime_col)

    n = 0
    sx = 0.0
    sy = 0.0
    sxx = 0.0
    sxy = 0.0
    syy = 0.0

    for chunk in pd.read_csv(input_path, usecols=usecols, chunksize=args.chunksize):
        chunk = filter_date(chunk, args.datetime_col, args.train_start, args.train_end)

        x = pd.to_numeric(chunk[args.pred_col], errors="coerce").to_numpy(float)
        y = pd.to_numeric(chunk[args.target_col], errors="coerce").to_numpy(float)

        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]

        if len(x) == 0:
            continue

        n += len(x)
        sx += x.sum()
        sy += y.sum()
        sxx += (x * x).sum()
        sxy += (x * y).sum()
        syy += (y * y).sum()

    if n < 100:
        raise RuntimeError(f"too few valid rows for calibration: n={n}")

    denom = n * sxx - sx * sx

    if abs(denom) < 1e-20:
        a = 0.0
        b = sy / n
    else:
        a = (n * sxy - sx * sy) / denom
        b = (sy - a * sx) / n

    mean_x = sx / n
    mean_y = sy / n

    var_x = max(sxx / n - mean_x * mean_x, 0.0)
    var_y = max(syy / n - mean_y * mean_y, 0.0)
    cov_xy = sxy / n - mean_x * mean_y

    if var_x > 0 and var_y > 0:
        corr = cov_xy / (var_x ** 0.5 * var_y ** 0.5)
    else:
        corr = 0.0

    result = {
        "method": "global_linear",
        "formula": "calibrated_pred = a * raw_pred + b",
        "pred_col": args.pred_col,
        "target_col": args.target_col,
        "datetime_col": args.datetime_col,
        "train_start": args.train_start,
        "train_end": args.train_end,
        "a": float(a),
        "b": float(b),
        "n": int(n),
        "mean_pred": float(mean_x),
        "mean_target": float(mean_y),
        "std_pred": float(var_x ** 0.5),
        "std_target": float(var_y ** 0.5),
        "corr_pred_target": float(corr),
        "max_fair_shift_ticks": float(args.max_fair_shift_ticks),
    }

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print("saved:", output_path)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
