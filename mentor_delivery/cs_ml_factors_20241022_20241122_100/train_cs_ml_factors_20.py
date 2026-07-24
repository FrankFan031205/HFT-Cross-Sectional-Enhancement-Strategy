#!/usr/bin/env python3
"""
Train 20 cross-sectional ML hidden factors from one-month factor data.

Design:
    5 non-deep-learning models x 4 horizons = 20 factors

Models:
    ridge
    lasso_sgd
    elasticnet_sgd
    lgbm
    extratrees

Horizons:
    30, 60, 90, 120

Run from FactorModel root:
    cd /home/fwz/projects/HFT_010-dev_fwz/FactorModel

    python src/train_cs_ml_factors_20.py \
      --input data/raw/factor_features_20241022_20241122_100.csv \
      --feature_cols data/raw/feature_cols_20241022_20241122_100.yaml \
      --tag 20241022_20241122_100
"""

import os
import gc
import argparse
import warnings
import yaml
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, SGDRegressor
from sklearn.ensemble import ExtraTreesRegressor

warnings.filterwarnings("ignore")


def ensure_dir(path):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_feature_cols(path, df_cols=None, feature_prefix="fwz"):
    if path and os.path.exists(path):
        obj = load_yaml(path)
        if isinstance(obj, dict) and "features" in obj:
            return list(obj["features"])
        if isinstance(obj, list):
            return list(obj)
        raise RuntimeError(f"Unsupported feature yaml format: {path}")

    if df_cols is None:
        raise RuntimeError("feature_cols file not found and df_cols is None")

    ignore = {
        "date", "time", "datetime", "securityid", "symbol",
        "label_30", "label_60", "label_90", "label_120",
    }

    features = [
        c for c in df_cols
        if c not in ignore and (c.startswith(feature_prefix) or c.startswith("cf_"))
    ]

    if not features:
        raise RuntimeError("No feature columns detected")

    return features


def safe_corr(x, y, method="spearman"):
    tmp = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(tmp) < 3:
        return np.nan
    if tmp["x"].nunique() < 2 or tmp["y"].nunique() < 2:
        return np.nan
    return tmp["x"].corr(tmp["y"], method=method)


def add_cs_labels(df, horizons, group_col="datetime"):
    print("creating cross-sectional labels...", flush=True)

    for h in horizons:
        raw_col = f"label_{h}"
        cs_col = f"label_{h}_cs"
        rank_col = f"label_{h}_rank"

        if raw_col not in df.columns:
            raise RuntimeError(f"missing label column: {raw_col}")

        g = df.groupby(group_col, sort=False)[raw_col]
        mean = g.transform("mean")
        std = g.transform("std")

        df[cs_col] = ((df[raw_col] - mean) / (std + 1e-8)).astype("float32")
        df[rank_col] = (g.rank(pct=True).astype("float32") - 0.5)

        print(f"  created {cs_col}, {rank_col}", flush=True)

    return df


def preprocess_features(df, feature_cols, group_col="datetime", clip_q=0.01, do_cs_zscore=True):
    print("preprocessing features...", flush=True)

    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")

    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if clip_q is not None and clip_q > 0:
        print(f"  clipping global quantiles: {clip_q}, {1 - clip_q}", flush=True)
        q_low = df[feature_cols].quantile(clip_q)
        q_high = df[feature_cols].quantile(1 - clip_q)
        df[feature_cols] = df[feature_cols].clip(lower=q_low, upper=q_high, axis=1)

    if do_cs_zscore:
        print("  cross-sectional zscore features by datetime", flush=True)
        g = df.groupby(group_col, sort=False)
        mean = g[feature_cols].transform("mean")
        std = g[feature_cols].transform("std")
        df[feature_cols] = ((df[feature_cols] - mean) / (std + 1e-8)).astype("float32")
        df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return df


def make_model(model_name, seed):
    if model_name == "ridge":
        return Ridge(alpha=1.0, fit_intercept=True, random_state=seed)

    if model_name == "lasso_sgd":
        return SGDRegressor(
            loss="squared_error",
            penalty="l1",
            alpha=1e-5,
            max_iter=1000,
            tol=1e-4,
            learning_rate="adaptive",
            eta0=0.01,
            random_state=seed,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=5,
        )

    if model_name == "elasticnet_sgd":
        return SGDRegressor(
            loss="squared_error",
            penalty="elasticnet",
            alpha=1e-5,
            l1_ratio=0.15,
            max_iter=1000,
            tol=1e-4,
            learning_rate="adaptive",
            eta0=0.01,
            random_state=seed,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=5,
        )

    if model_name == "lgbm":
        try:
            from lightgbm import LGBMRegressor
        except Exception as e:
            raise RuntimeError("LightGBM is not available. Install lightgbm or remove lgbm from --models.") from e

        return LGBMRegressor(
            objective="regression",
            boosting_type="gbdt",
            n_estimators=600,
            learning_rate=0.03,
            num_leaves=31,
            max_depth=-1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.0,
            reg_lambda=1.0,
            min_child_samples=50,
            random_state=seed,
            n_jobs=8,
        )

    if model_name == "extratrees":
        return ExtraTreesRegressor(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=100,
            max_features="sqrt",
            bootstrap=False,
            random_state=seed,
            n_jobs=8,
        )

    raise RuntimeError(f"unknown model: {model_name}")


def sample_training_rows(train_idx, max_rows, seed):
    if max_rows is None or max_rows <= 0 or len(train_idx) <= max_rows:
        return train_idx
    rng = np.random.default_rng(seed)
    return rng.choice(train_idx, size=max_rows, replace=False)


def calc_timeseries_eval(df, factor_col, label_col, group_col="datetime", n_bins=5):
    rows = []

    for dt, g in df.groupby(group_col, sort=False):
        if len(g) < n_bins:
            continue

        row = {
            "datetime": dt,
            "n": len(g),
            "ic": safe_corr(g[factor_col], g[label_col], method="pearson"),
            "rankic": safe_corr(g[factor_col], g[label_col], method="spearman"),
        }

        try:
            tmp = g[[factor_col, label_col]].replace([np.inf, -np.inf], np.nan).dropna()
            if tmp[factor_col].nunique() < n_bins:
                raise ValueError("not enough unique factor values")

            tmp["q"] = pd.qcut(tmp[factor_col], q=n_bins, labels=False, duplicates="drop")
            qret = tmp.groupby("q")[label_col].mean()

            for i in range(n_bins):
                row[f"group_{i + 1}"] = qret.get(i, np.nan)

            row["long_short"] = qret.get(n_bins - 1, np.nan) - qret.get(0, np.nan)

        except Exception:
            for i in range(n_bins):
                row[f"group_{i + 1}"] = np.nan
            row["long_short"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def _mean_ir(series):
    if series is None or len(series) == 0:
        return np.nan, np.nan, np.nan
    return series.mean(), series.std(), series.mean() / (series.std() + 1e-12)


def summarize_factor(df, factor_col, raw_label_col, cs_label_col, split_name, n_bins=5):
    d = df[df["split"] == split_name].copy()
    d = d.dropna(subset=[factor_col, raw_label_col, cs_label_col])

    ts_raw = calc_timeseries_eval(d, factor_col, raw_label_col, group_col="datetime", n_bins=n_bins)
    ts_cs = calc_timeseries_eval(d, factor_col, cs_label_col, group_col="datetime", n_bins=n_bins)

    raw_rankic_mean, _, raw_rankicir = _mean_ir(ts_raw["rankic"] if len(ts_raw) else pd.Series(dtype=float))
    cs_rankic_mean, _, cs_rankicir = _mean_ir(ts_cs["rankic"] if len(ts_cs) else pd.Series(dtype=float))
    ls_mean, _, ls_ir = _mean_ir(ts_raw["long_short"] if len(ts_raw) else pd.Series(dtype=float))

    res = {
        "factor": factor_col,
        "split": split_name,
        "rows": len(d),
        "num_datetimes": d["datetime"].nunique(),
        "overall_ic_raw": safe_corr(d[factor_col], d[raw_label_col], method="pearson"),
        "overall_rankic_raw": safe_corr(d[factor_col], d[raw_label_col], method="spearman"),
        "overall_ic_cs": safe_corr(d[factor_col], d[cs_label_col], method="pearson"),
        "overall_rankic_cs": safe_corr(d[factor_col], d[cs_label_col], method="spearman"),
        "mean_cs_rankic_raw_label": raw_rankic_mean,
        "cs_rankicir_raw_label": raw_rankicir,
        "mean_cs_rankic_cs_label": cs_rankic_mean,
        "cs_rankicir_cs_label": cs_rankicir,
        "mean_long_short_raw_label": ls_mean,
        "long_short_ir_raw_label": ls_ir,
    }

    for i in range(n_bins):
        c = f"group_{i + 1}"
        res[f"mean_{c}_raw_label"] = ts_raw[c].mean() if len(ts_raw) else np.nan

    return res


def save_single_factor(df, out_path, factor_col, horizon):
    cols = [
        "date",
        "datetime",
        "securityid",
        f"label_{horizon}",
        f"label_{horizon}_cs",
        factor_col,
        "split",
    ]
    ensure_dir(out_path)
    df[cols].to_csv(out_path, index=False)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", default="data/raw/factor_features_20241022_20241122_100.csv")
    parser.add_argument("--feature_cols", default="data/raw/feature_cols_20241022_20241122_100.yaml")
    parser.add_argument("--tag", default="20241022_20241122_100")

    parser.add_argument("--date_col", default="date")
    parser.add_argument("--datetime_col", default="datetime")
    parser.add_argument("--symbol_col", default="securityid")

    parser.add_argument("--horizons", default="30,60,90,120")
    parser.add_argument("--models", default="ridge,lasso_sgd,elasticnet_sgd,lgbm,extratrees")

    parser.add_argument("--train_start", type=int, default=20241022)
    parser.add_argument("--train_end", type=int, default=20241112)
    parser.add_argument("--valid_start", type=int, default=20241113)
    parser.add_argument("--valid_end", type=int, default=20241115)
    parser.add_argument("--test_start", type=int, default=20241118)
    parser.add_argument("--test_end", type=int, default=20241122)

    parser.add_argument("--clip_q", type=float, default=0.01)
    parser.add_argument("--no_cs_zscore_features", action="store_true")
    parser.add_argument("--max_train_rows", type=int, default=2_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_bins", type=int, default=5)

    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--merged_output", default=None)
    parser.add_argument("--eval_output", default=None)

    args = parser.parse_args()

    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]
    model_names = [x.strip() for x in args.models.split(",") if x.strip()]

    merged_output = args.merged_output or f"{args.output_dir}/ml_cs_hidden_factors_20_{args.tag}.csv"
    eval_output = args.eval_output or f"{args.output_dir}/eval_cs_ml_factors_20_{args.tag}.csv"

    print("=" * 100, flush=True)
    print("Cross-sectional ML factor training", flush=True)
    print("input:", args.input, flush=True)
    print("feature_cols:", args.feature_cols, flush=True)
    print("models:", model_names, flush=True)
    print("horizons:", horizons, flush=True)
    print("tag:", args.tag, flush=True)
    print("=" * 100, flush=True)

    header_cols = pd.read_csv(args.input, nrows=0).columns.tolist()
    feature_cols = load_feature_cols(args.feature_cols, header_cols)

    required_cols = [args.date_col, args.datetime_col, args.symbol_col] + feature_cols + [f"label_{h}" for h in horizons]
    missing = [c for c in required_cols if c not in header_cols]
    if missing:
        raise RuntimeError(f"missing columns in input: {missing[:20]}, total={len(missing)}")

    print("num features:", len(feature_cols), flush=True)

    dtype = {args.symbol_col: str, args.datetime_col: str}

    print("loading data...", flush=True)
    df = pd.read_csv(args.input, usecols=required_cols, dtype=dtype)

    rename_map = {}
    if args.date_col != "date":
        rename_map[args.date_col] = "date"
    if args.datetime_col != "datetime":
        rename_map[args.datetime_col] = "datetime"
    if args.symbol_col != "securityid":
        rename_map[args.symbol_col] = "securityid"
    if rename_map:
        df = df.rename(columns=rename_map)

    df["date"] = df["date"].astype(int)
    df["securityid"] = df["securityid"].astype(str).str.zfill(6)
    df["datetime"] = df["datetime"].astype(str)

    print("raw shape:", df.shape, flush=True)
    print("date range:", df["date"].min(), df["date"].max(), flush=True)
    print("num dates:", df["date"].nunique(), flush=True)
    print("num stocks:", df["securityid"].nunique(), flush=True)

    df = df.sort_values(["date", "datetime", "securityid"]).reset_index(drop=True)

    df = add_cs_labels(df, horizons, group_col="datetime")
    df = preprocess_features(
        df,
        feature_cols=feature_cols,
        group_col="datetime",
        clip_q=args.clip_q,
        do_cs_zscore=not args.no_cs_zscore_features,
    )

    train_mask = (df["date"] >= args.train_start) & (df["date"] <= args.train_end)
    valid_mask = (df["date"] >= args.valid_start) & (df["date"] <= args.valid_end)
    test_mask = (df["date"] >= args.test_start) & (df["date"] <= args.test_end)

    df["split"] = "other"
    df.loc[train_mask, "split"] = "train"
    df.loc[valid_mask, "split"] = "valid"
    df.loc[test_mask, "split"] = "test"

    print("split counts:", flush=True)
    print(df["split"].value_counts(), flush=True)

    train_idx_all = np.where(train_mask.values)[0]
    if len(train_idx_all) == 0:
        raise RuntimeError("empty train set")

    X_all = df[feature_cols].values.astype("float32")

    output_factor_cols = []
    eval_rows = []

    for model_name in model_names:
        for h in horizons:
            print("\n" + "=" * 100, flush=True)
            print(f"training model={model_name}, horizon=h{h}", flush=True)

            raw_label_col = f"label_{h}"
            cs_label_col = f"label_{h}_cs"
            factor_col = f"hidden_factor_cs_{model_name}_h{h}"

            y_all = df[cs_label_col].values.astype("float32")

            train_idx = train_idx_all.copy()
            train_idx = train_idx[np.isfinite(y_all[train_idx])]

            train_idx_fit = sample_training_rows(
                train_idx,
                max_rows=args.max_train_rows,
                seed=args.seed + h + len(model_name),
            )

            print("train rows available:", len(train_idx), flush=True)
            print("train rows used:", len(train_idx_fit), flush=True)

            X_train = X_all[train_idx_fit]
            y_train = y_all[train_idx_fit]

            model = make_model(model_name, seed=args.seed + h)
            model.fit(X_train, y_train)

            print("predicting all rows...", flush=True)
            pred = model.predict(X_all).astype("float32")
            df[factor_col] = pred
            output_factor_cols.append(factor_col)

            single_path = f"{args.output_dir}/hidden_factor_cs_{model_name}_h{h}_{args.tag}.csv"
            save_single_factor(df, single_path, factor_col=factor_col, horizon=h)
            print("saved single factor:", single_path, flush=True)

            for split_name in ["train", "valid", "test"]:
                row = summarize_factor(
                    df,
                    factor_col=factor_col,
                    raw_label_col=raw_label_col,
                    cs_label_col=cs_label_col,
                    split_name=split_name,
                    n_bins=args.n_bins,
                )
                row.update({
                    "model": model_name,
                    "horizon": h,
                    "target_raw_label": raw_label_col,
                    "target_cs_label": cs_label_col,
                    "single_factor_path": single_path,
                })
                eval_rows.append(row)

            if hasattr(model, "coef_"):
                coef_path = f"{args.output_dir}/coef_cs_{model_name}_h{h}_{args.tag}.csv"
                pd.DataFrame({"feature": feature_cols, "coef": model.coef_}).sort_values(
                    "coef", key=lambda s: s.abs(), ascending=False
                ).to_csv(coef_path, index=False)
                print("saved coef:", coef_path, flush=True)

            if model_name == "lgbm" and hasattr(model, "feature_importances_"):
                imp_path = f"{args.output_dir}/feature_importance_cs_{model_name}_h{h}_{args.tag}.csv"
                pd.DataFrame({"feature": feature_cols, "importance": model.feature_importances_}).sort_values(
                    "importance", ascending=False
                ).to_csv(imp_path, index=False)
                print("saved feature importance:", imp_path, flush=True)

            del model, X_train, y_train, pred
            gc.collect()

    keep_cols = ["date", "datetime", "securityid"]
    for h in horizons:
        keep_cols.extend([f"label_{h}", f"label_{h}_cs", f"label_{h}_rank"])
    keep_cols.extend(output_factor_cols)
    keep_cols.append("split")

    ensure_dir(merged_output)
    df[keep_cols].to_csv(merged_output, index=False)
    print("\nsaved merged 20-factor file:", merged_output, flush=True)

    eval_df = pd.DataFrame(eval_rows)
    eval_df = eval_df.sort_values(
        ["split", "mean_cs_rankic_raw_label", "mean_long_short_raw_label"],
        ascending=[True, False, False],
    )

    ensure_dir(eval_output)
    eval_df.to_csv(eval_output, index=False)
    print("saved eval summary:", eval_output, flush=True)

    print("\n=== TEST SET RANKING ===", flush=True)
    test_eval = eval_df[eval_df["split"] == "test"].copy()
    test_eval = test_eval.sort_values(
        ["mean_cs_rankic_raw_label", "mean_long_short_raw_label"],
        ascending=False,
    )

    display_cols = [
        "factor", "model", "horizon", "rows", "num_datetimes",
        "mean_cs_rankic_raw_label", "cs_rankicir_raw_label",
        "mean_long_short_raw_label", "long_short_ir_raw_label",
        "overall_rankic_raw", "overall_rankic_cs",
    ]
    print(test_eval[display_cols].to_string(index=False), flush=True)
    print("\nDONE.", flush=True)


if __name__ == "__main__":
    main()
