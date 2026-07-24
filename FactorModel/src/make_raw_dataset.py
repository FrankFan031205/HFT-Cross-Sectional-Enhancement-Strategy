import os
import re
import argparse
import numpy as np
import pandas as pd


def read_data(path):
    if path.endswith(".csv"):
        return pd.read_csv(path)
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    if path.endswith(".pkl"):
        return pd.read_pickle(path)
    raise ValueError(f"unsupported input file type: {path}")


def ensure_dir(path):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def infer_label_cols(df, horizons):
    label_cols = []
    rename_map = {}

    for h in horizons:
        candidates = [
            f"label_{h}",
            f"ret_{h}",
            f"return_{h}",
            f"future_return_{h}",
            f"future_ret_{h}",
            f"y_{h}",
            f"pred_horizon_{h}",
        ]

        found = None
        for c in candidates:
            if c in df.columns:
                found = c
                break

        if found is None:
            continue

        target = f"label_{h}"
        if found != target:
            rename_map[found] = target

        label_cols.append(target)

    if rename_map:
        df = df.rename(columns=rename_map)

    return df, label_cols


def make_labels_by_row_shift(df, datetime_col, symbol_col, mid_col, horizons):
    df = df.sort_values([symbol_col, datetime_col]).copy()

    for h in horizons:
        future_mid = df.groupby(symbol_col)[mid_col].shift(-h)
        df[f"label_{h}"] = (future_mid - df[mid_col]) / df[mid_col]

    return df


def make_labels_by_time_shift(df, datetime_col, symbol_col, mid_col, horizons):
    df = df.copy()
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df = df.sort_values([symbol_col, datetime_col])

    out_list = []

    for sym, g in df.groupby(symbol_col, sort=False):
        g = g.sort_values(datetime_col).copy()

        base_time = g[datetime_col].values.astype("datetime64[ns]")
        base_mid = g[mid_col].values

        for h in horizons:
            target_time = g[datetime_col] + pd.to_timedelta(h, unit="s")
            target_time = target_time.values.astype("datetime64[ns]")

            idx = np.searchsorted(base_time, target_time, side="left")

            label = np.full(len(g), np.nan)
            valid = idx < len(g)

            label[valid] = (base_mid[idx[valid]] - base_mid[valid]) / base_mid[valid]
            g[f"label_{h}"] = label

        out_list.append(g)

    return pd.concat(out_list, axis=0, ignore_index=True)


def infer_mid_col(df):
    candidates = [
        "mid_price",
        "mid",
        "midpx",
        "mid_px",
        "wap",
        "last_mid_price",
    ]

    for c in candidates:
        if c in df.columns:
            return c

    bid_candidates = [
        ("bid1_price", "ask1_price"),
        ("bidprice1", "askprice1"),
        ("bid1", "ask1"),
        ("bid_px1", "ask_px1"),
        ("bid_price_1", "ask_price_1"),
    ]

    for bid_col, ask_col in bid_candidates:
        if bid_col in df.columns and ask_col in df.columns:
            df["mid_price"] = (df[bid_col] + df[ask_col]) / 2.0
            return "mid_price"

    return None


def infer_factor_cols(df, datetime_col, symbol_col, label_cols, factor_regex):
    exclude_exact = set([
        datetime_col,
        symbol_col,
        "date",
        "time",
        "timestamp",
        "datetime",
        "symbol",
        "stock",
        "code",
        "ticker",
        "instrument",
        "mid",
        "mid_price",
        "midpx",
        "mid_px",
        "wap",
        "last_price",
        "last",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "turnover",
    ])

    exclude_exact |= set(label_cols)

    exclude_patterns = [
        r"^label_\d+$",
        r"^ret_\d+$",
        r"^return_\d+$",
        r"^future_return_\d+$",
        r"^future_ret_\d+$",
        r"^y_\d+$",
        r"^bid.*price",
        r"^ask.*price",
        r"^bid.*size",
        r"^ask.*size",
        r"^bid.*volume",
        r"^ask.*volume",
    ]

    numeric_cols = []
    for c in df.columns:
        if c in exclude_exact:
            continue

        skip = False
        for pat in exclude_patterns:
            if re.search(pat, c):
                skip = True
                break
        if skip:
            continue

        if pd.api.types.is_numeric_dtype(df[c]):
            numeric_cols.append(c)

    if factor_regex:
        reg = re.compile(factor_regex)
        matched = [c for c in numeric_cols if reg.search(c)]
        if len(matched) > 0:
            return matched

    return numeric_cols


def clean_output(df, cols, factor_cols, label_cols):
    out = df[cols].copy()

    num_cols = factor_cols + label_cols
    out[num_cols] = out[num_cols].replace([np.inf, -np.inf], np.nan)

    out = out.dropna(subset=label_cols)

    for c in factor_cols:
        if out[c].isna().any():
            out[c] = out[c].fillna(0)

    return out


def save_feature_yaml(feature_cols, output_path):
    ensure_dir(output_path)

    with open(output_path, "w") as f:
        f.write("features:\n")
        for c in feature_cols:
            f.write(f"  - {c}\n")


def save_summary(df, factor_cols, label_cols, output_path):
    ensure_dir(output_path)

    lines = []
    lines.append(f"rows: {len(df)}")
    lines.append(f"num_factors: {len(factor_cols)}")
    lines.append(f"factor_cols: {factor_cols}")
    lines.append(f"label_cols: {label_cols}")
    lines.append("")
    lines.append("label describe:")
    lines.append(str(df[label_cols].describe()))
    lines.append("")
    lines.append("factor missing rate top:")
    lines.append(str(df[factor_cols].isna().mean().sort_values(ascending=False).head(20)))

    with open(output_path, "w") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="FactorModel/data/raw/factor_features_202410_100.csv")
    parser.add_argument("--feature_yaml", default="FactorModel/data/raw/feature_cols_202410_100.yaml")
    parser.add_argument("--summary", default="FactorModel/data/raw/raw_dataset_summary_202410_100.txt")

    parser.add_argument("--datetime_col", default="datetime")
    parser.add_argument("--symbol_col", default="symbol")

    parser.add_argument("--horizons", default="30,60,90,120")
    parser.add_argument("--mid_col", default=None)

    parser.add_argument(
        "--label_mode",
        default="auto",
        choices=["auto", "existing", "row_shift", "time_shift"],
        help="existing: use existing label/ret columns; row_shift: horizon means future rows; time_shift: horizon means future seconds"
    )

    parser.add_argument(
        "--factor_regex",
        default="^fwz",
        help="used only for detecting factor columns. Factor names will NOT be changed."
    )

    args = parser.parse_args()

    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]

    df = read_data(args.input)

    if args.datetime_col not in df.columns:
        raise ValueError(f"datetime_col not found: {args.datetime_col}")

    if args.symbol_col not in df.columns:
        raise ValueError(f"symbol_col not found: {args.symbol_col}")

    label_cols = []

    if args.label_mode in ["auto", "existing"]:
        df, label_cols = infer_label_cols(df, horizons)

    if args.label_mode == "existing" and len(label_cols) == 0:
        raise ValueError("label_mode=existing but no label columns found")

    if args.label_mode in ["auto", "row_shift", "time_shift"] and len(label_cols) == 0:
        mid_col = args.mid_col

        if mid_col is None:
            mid_col = infer_mid_col(df)

        if mid_col is None:
            raise ValueError(
                "No existing label columns found, and no mid price column found. "
                "Please provide --mid_col or make sure label_30/label_60/... already exist."
            )

        if args.label_mode == "time_shift":
            df = make_labels_by_time_shift(
                df=df,
                datetime_col=args.datetime_col,
                symbol_col=args.symbol_col,
                mid_col=mid_col,
                horizons=horizons,
            )
        else:
            df = make_labels_by_row_shift(
                df=df,
                datetime_col=args.datetime_col,
                symbol_col=args.symbol_col,
                mid_col=mid_col,
                horizons=horizons,
            )

        label_cols = [f"label_{h}" for h in horizons]

    factor_cols = infer_factor_cols(
        df=df,
        datetime_col=args.datetime_col,
        symbol_col=args.symbol_col,
        label_cols=label_cols,
        factor_regex=args.factor_regex,
    )

    if len(factor_cols) == 0:
        raise ValueError("No factor columns detected. Try changing --factor_regex.")

    cols = [args.datetime_col, args.symbol_col] + factor_cols + label_cols

    out = clean_output(
        df=df,
        cols=cols,
        factor_cols=factor_cols,
        label_cols=label_cols,
    )

    ensure_dir(args.output)
    out.to_csv(args.output, index=False)

    save_feature_yaml(factor_cols, args.feature_yaml)
    save_summary(out, factor_cols, label_cols, args.summary)

    print("saved raw dataset:", args.output)
    print("saved feature yaml:", args.feature_yaml)
    print("saved summary:", args.summary)
    print("shape:", out.shape)
    print("num factors:", len(factor_cols))
    print("factor cols:")
    for c in factor_cols:
        print("  ", c)
    print("label cols:", label_cols)


if __name__ == "__main__":
    main()