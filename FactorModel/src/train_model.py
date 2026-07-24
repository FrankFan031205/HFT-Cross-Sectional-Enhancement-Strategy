import os
import argparse
import yaml
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr, pearsonr


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_features(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)["features"]


def ensure_dir(path):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def calc_ic(y_true, y_pred):
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 3:
        return np.nan, np.nan

    y_true = y_true[mask]
    y_pred = y_pred[mask]

    try:
        ic = pearsonr(y_true, y_pred)[0]
    except Exception:
        ic = np.nan

    try:
        rankic = spearmanr(y_true, y_pred)[0]
    except Exception:
        rankic = np.nan

    return ic, rankic


def calc_cross_sectional_ic(df, pred_col, label_col, group_col):
    rows = []

    for dt, g in df.groupby(group_col, sort=False):
        if len(g) < 3:
            continue

        ic, rankic = calc_ic(
            g[label_col].values.astype(float),
            g[pred_col].values.astype(float),
        )

        rows.append({
            "datetime": dt,
            "ic": ic,
            "rankic": rankic,
            "n": len(g),
        })

    res = pd.DataFrame(rows)

    if len(res) == 0:
        return {
            "mean_ic": np.nan,
            "mean_rankic": np.nan,
            "icir": np.nan,
            "rankicir": np.nan,
        }

    return {
        "mean_ic": res["ic"].mean(),
        "mean_rankic": res["rankic"].mean(),
        "icir": res["ic"].mean() / (res["ic"].std() + 1e-12),
        "rankicir": res["rankic"].mean() / (res["rankic"].std() + 1e-12),
    }


def make_split_mask(df, date_col, start, end):
    return (df[date_col] >= start) & (df[date_col] <= end)


def train_ridge(cfg, X_train, y_train):
    params = cfg["model"].get("params", {})
    params = {k: v for k, v in params.items() if k != "random_state"}
    model = Ridge(**params)
    model.fit(X_train, y_train)
    return model


def train_lgbm(cfg, X_train, y_train, X_valid, y_valid):
    import lightgbm as lgb

    params = cfg["model"].get("params", {}).copy()
    early_stopping_rounds = cfg["train"].get("early_stopping_rounds", 50)
    eval_metric = cfg["train"].get("eval_metric", "l2")

    model = lgb.LGBMRegressor(**params)

    # New LightGBM versions support callbacks
    if hasattr(lgb, "early_stopping") and hasattr(lgb, "log_evaluation"):
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric=eval_metric,
            callbacks=[
                lgb.early_stopping(early_stopping_rounds),
                lgb.log_evaluation(50),
            ],
        )
        return model

    # Older LightGBM versions may support early_stopping_rounds directly
    try:
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric=eval_metric,
            early_stopping_rounds=early_stopping_rounds,
            verbose=50,
        )
        return model
    except TypeError:
        pass

    # Very old fallback: train without early stopping
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric=eval_metric,
    )

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]

    date_col = data_cfg.get("date_col", "date")
    datetime_col = data_cfg["datetime_col"]
    symbol_col = data_cfg["symbol_col"]
    label_col = data_cfg["label_col"]

    feature_cols = load_features(data_cfg["feature_cols_path"])

    print("loading processed data:", data_cfg["processed_data_path"])
    df = pd.read_csv(data_cfg["processed_data_path"])

    print("data shape:", df.shape)
    print("date range:", df[date_col].min(), df[date_col].max())
    print("num features:", len(feature_cols))
    print("label:", label_col)

    train_mask = make_split_mask(
        df, date_col, data_cfg["train_start"], data_cfg["train_end"]
    )
    valid_mask = make_split_mask(
        df, date_col, data_cfg["valid_start"], data_cfg["valid_end"]
    )
    test_mask = make_split_mask(
        df, date_col, data_cfg["test_start"], data_cfg["test_end"]
    )

    print("train rows:", train_mask.sum())
    print("valid rows:", valid_mask.sum())
    print("test rows:", test_mask.sum())

    if train_mask.sum() == 0:
        raise RuntimeError("empty train set")
    if valid_mask.sum() == 0:
        raise RuntimeError("empty valid set")
    if test_mask.sum() == 0:
        raise RuntimeError("empty test set")

    X_train = df.loc[train_mask, feature_cols].astype("float32")
    y_train = df.loc[train_mask, label_col].astype("float32")

    X_valid = df.loc[valid_mask, feature_cols].astype("float32")
    y_valid = df.loc[valid_mask, label_col].astype("float32")

    X_test = df.loc[test_mask, feature_cols].astype("float32")
    y_test = df.loc[test_mask, label_col].astype("float32")

    model_type = model_cfg["type"]

    if model_type == "ridge":
        model = train_ridge(cfg, X_train, y_train)
    elif model_type == "lightgbm":
        model = train_lgbm(cfg, X_train, y_train, X_valid, y_valid)
    else:
        raise RuntimeError(f"unknown model type: {model_type}")

    output_col = cfg["prediction"]["output_col"]

    df[output_col] = np.nan
    df.loc[train_mask, output_col] = model.predict(df.loc[train_mask, feature_cols].astype("float32"))
    df.loc[valid_mask, output_col] = model.predict(df.loc[valid_mask, feature_cols].astype("float32"))
    df.loc[test_mask, output_col] = model.predict(df.loc[test_mask, feature_cols].astype("float32"))

    df["split"] = "other"
    df.loc[train_mask, "split"] = "train"
    df.loc[valid_mask, "split"] = "valid"
    df.loc[test_mask, "split"] = "test"

    metrics = []

    for split_name, mask in [
        ("train", train_mask),
        ("valid", valid_mask),
        ("test", test_mask),
    ]:
        y_true = df.loc[mask, label_col].values.astype(float)
        y_pred = df.loc[mask, output_col].values.astype(float)

        mse = mean_squared_error(y_true, y_pred)
        ic, rankic = calc_ic(y_true, y_pred)

        cs = calc_cross_sectional_ic(
            df.loc[mask, [datetime_col, label_col, output_col]],
            pred_col=output_col,
            label_col=label_col,
            group_col=datetime_col,
        )

        metrics.append({
            "split": split_name,
            "rows": int(mask.sum()),
            "mse": mse,
            "overall_ic": ic,
            "overall_rankic": rankic,
            "mean_cs_ic": cs["mean_ic"],
            "mean_cs_rankic": cs["mean_rankic"],
            "cs_icir": cs["icir"],
            "cs_rankicir": cs["rankicir"],
        })

    metrics_df = pd.DataFrame(metrics)
    print(metrics_df)

    pred_cols = [
        date_col,
        datetime_col,
        symbol_col,
        label_col,
        output_col,
        "split",
    ]

    pred_path = cfg["prediction"]["output_path"]
    ensure_dir(pred_path)
    df[pred_cols].to_csv(pred_path, index=False)
    print("saved prediction:", pred_path)

    model_path = cfg["train"]["model_path"]
    ensure_dir(model_path)
    joblib.dump(model, model_path)
    print("saved model:", model_path)

    metrics_path = cfg["output"]["metrics_path"]
    ensure_dir(metrics_path)
    metrics_df.to_csv(metrics_path, index=False)
    print("saved metrics:", metrics_path)

    if model_type == "lightgbm":
        fi = pd.DataFrame({
            "feature": feature_cols,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)

        fi_path = cfg["output"]["feature_importance_path"]
        ensure_dir(fi_path)
        fi.to_csv(fi_path, index=False)
        print("saved feature importance:", fi_path)

    if model_type == "ridge":
        coef = pd.DataFrame({
            "feature": feature_cols,
            "coef": model.coef_,
        }).sort_values("coef", key=lambda x: x.abs(), ascending=False)

        coef_path = cfg["output"]["coef_path"]
        ensure_dir(coef_path)
        coef.to_csv(coef_path, index=False)
        print("saved ridge coef:", coef_path)


if __name__ == "__main__":
    main()