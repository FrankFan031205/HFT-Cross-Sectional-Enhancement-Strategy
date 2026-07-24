import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def repo_root():
    return Path(__file__).resolve().parents[2]


def resolve_path(p):
    p = Path(p)
    return p if p.is_absolute() else repo_root() / p


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_date_series(s):
    return s.astype(str).str.slice(0, 8)


def in_range(date_s, start, end):
    return (date_s >= str(start)) & (date_s <= str(end))


def normalize_securityid(s):
    return s.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)


def detect_columns(path, cfg):
    header = pd.read_csv(path, nrows=0)
    cols = header.columns.tolist()

    c = cfg["columns"]
    suffix = c["feature_suffix"]

    feature_cols = [
        x for x in cols
        if x.endswith(suffix)
        and x not in [
            c["label_col"],
            c["target_col"],
            c["label_excess_col"],
        ]
    ]

    base_cols = [
        c["datetime_col"],
        c["symbol_col"],
        c["label_col"],
        c["target_col"],
        c["label_excess_col"],
    ]

    missing = [x for x in base_cols if x not in cols]
    if missing:
        raise ValueError("missing required columns: {}".format(missing))

    if not feature_cols:
        raise ValueError("no feature columns found with suffix {}".format(suffix))

    print("feature count:", len(feature_cols))
    print("first 20 features:", feature_cols[:20])

    return base_cols, feature_cols


def count_rows(path, cfg):
    c = cfg["columns"]
    tr = cfg["split"]
    chunksize = int(cfg["training"]["chunksize"])

    counts = {"train": 0, "valid": 0, "test": 0, "all": 0}

    for chunk in pd.read_csv(path, usecols=[c["datetime_col"]], chunksize=chunksize):
        d = get_date_series(chunk[c["datetime_col"]])
        counts["train"] += int(in_range(d, tr["train_start"], tr["train_end"]).sum())
        counts["valid"] += int(in_range(d, tr["valid_start"], tr["valid_end"]).sum())
        counts["test"] += int(in_range(d, tr["test_start"], tr["test_end"]).sum())
        counts["all"] += len(chunk)

    return counts


def load_sample(path, cfg, base_cols, feature_cols, split_name, start, end, max_rows, total_count):
    c = cfg["columns"]
    tcfg = cfg["training"]

    chunksize = int(tcfg["chunksize"])
    random_state = int(tcfg["random_state"])
    fillna_value = float(tcfg["fillna_value"])

    frac = min(1.0, float(max_rows) / float(max(total_count, 1)))

    print("\nloading {} sample".format(split_name))
    print("date range:", start, end)
    print("total rows:", total_count, "max rows:", max_rows, "sample frac:", frac)

    parts = []
    rng = np.random.RandomState(random_state)

    usecols = base_cols + feature_cols

    for i, chunk in enumerate(pd.read_csv(path, usecols=usecols, chunksize=chunksize, low_memory=False)):
        d = get_date_series(chunk[c["datetime_col"]])
        chunk = chunk.loc[in_range(d, start, end)]

        if chunk.empty:
            continue

        if frac < 1.0:
            seed = int(rng.randint(0, 2**31 - 1))
            chunk = chunk.sample(frac=frac, random_state=seed)

        parts.append(chunk)

        if (i + 1) % 10 == 0:
            print(split_name, "chunk", i + 1, "sampled rows so far:", sum(len(x) for x in parts))

    if not parts:
        raise RuntimeError("empty {} sample".format(split_name))

    df = pd.concat(parts, axis=0, ignore_index=True)

    if len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=random_state).reset_index(drop=True)

    df[c["target_col"]] = pd.to_numeric(df[c["target_col"]], errors="coerce")
    df = df.dropna(subset=[c["target_col"]]).reset_index(drop=True)

    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(fillna_value)
    y = df[c["target_col"]].astype(float)

    print(split_name, "final shape:", X.shape)

    return X, y


def train_ridge(X_train, y_train, cfg):
    from sklearn.linear_model import Ridge

    alpha = float(cfg["models"]["ridge"].get("alpha", 10.0))
    model = Ridge(alpha=alpha, random_state=int(cfg["training"]["random_state"]))
    model.fit(X_train, y_train)
    return model


def train_lgbm(X_train, y_train, X_valid, y_valid, cfg):
    try:
        import lightgbm as lgb
    except Exception as e:
        print("LightGBM import failed, skip lgbm. error:", repr(e))
        return None

    params = dict(cfg["models"]["lgbm"].get("params", {}))
    model = lgb.LGBMRegressor(**params)

    X_train_np = X_train.to_numpy(dtype=np.float32, copy=True)
    X_valid_np = X_valid.to_numpy(dtype=np.float32, copy=True)
    y_train_np = np.asarray(y_train, dtype=np.float32)
    y_valid_np = np.asarray(y_valid, dtype=np.float32)

    try:
        model.fit(
            X_train_np,
            y_train_np,
            eval_set=[(X_valid_np, y_valid_np)],
            eval_metric="l2",
        )
    except TypeError:
        model.fit(X_train_np, y_train_np)

    return model

def safe_corr(a, b):
    a = pd.Series(a)
    b = pd.Series(b)
    m = a.notna() & b.notna()
    if m.sum() < 3:
        return np.nan
    return a[m].corr(b[m], method="spearman")


def quick_eval(model, X, y, name):
    pred = model.predict(X)
    corr = safe_corr(pred, y)
    print("{} validation spearman: {:.6f}".format(name, corr))
    return corr


def save_model(model, path):
    try:
        import joblib
        joblib.dump(model, path)
        print("saved model:", path)
    except Exception as e:
        print("failed to save model:", path, repr(e))


def predict_full(path, cfg, base_cols, feature_cols, models):
    c = cfg["columns"]
    tcfg = cfg["training"]

    chunksize = int(tcfg["chunksize"])
    fillna_value = float(tcfg["fillna_value"])

    output_path = resolve_path(cfg["data"]["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    usecols = base_cols + feature_cols
    first = True
    total = 0

    if output_path.exists():
        output_path.unlink()

    for idx, chunk in enumerate(pd.read_csv(path, usecols=usecols, chunksize=chunksize, low_memory=False)):
        out = chunk[base_cols].copy()
        out[c["symbol_col"]] = normalize_securityid(out[c["symbol_col"]])

        X = chunk[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(fillna_value)

        for name, model in models.items():
            if model is None:
                continue
            out[name] = model.predict(X)

        out.to_csv(output_path, mode="w" if first else "a", index=False, header=first)

        first = False
        total += len(out)

        print("pred chunk", idx + 1, "rows:", len(out), "total:", total)

    print("saved hidden factors:", output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="CrossSectionalModel/config/cs_hidden_factor_h60.yaml",
    )
    args = parser.parse_args()

    cfg = load_yaml(resolve_path(args.config))
    data_path = resolve_path(cfg["data"]["cs_dataset_path"])
    model_dir = resolve_path(cfg["data"]["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)

    base_cols, feature_cols = detect_columns(data_path, cfg)

    counts = count_rows(data_path, cfg)
    print("\nrow counts:", counts)

    split = cfg["split"]
    train_max = int(cfg["training"]["max_train_rows"])
    valid_max = int(cfg["training"]["max_valid_rows"])

    X_train, y_train = load_sample(
        data_path,
        cfg,
        base_cols,
        feature_cols,
        "train",
        split["train_start"],
        split["train_end"],
        train_max,
        counts["train"],
    )

    X_valid, y_valid = load_sample(
        data_path,
        cfg,
        base_cols,
        feature_cols,
        "valid",
        split["valid_start"],
        split["valid_end"],
        valid_max,
        counts["valid"],
    )

    models = {}

    if bool(cfg["models"]["ridge"].get("enabled", True)):
        print("\ntraining ridge")
        ridge = train_ridge(X_train, y_train, cfg)
        quick_eval(ridge, X_valid, y_valid, "hidden_cs_ridge_h60")
        save_model(ridge, model_dir / "hidden_cs_ridge_h60.pkl")
        models["hidden_cs_ridge_h60"] = ridge

    if bool(cfg["models"]["lgbm"].get("enabled", True)):
        print("\ntraining lgbm")
        lgbm = train_lgbm(X_train, y_train, X_valid, y_valid, cfg)
        if lgbm is not None:
            quick_eval(lgbm, X_valid, y_valid, "hidden_cs_lgbm_h60")
            save_model(lgbm, model_dir / "hidden_cs_lgbm_h60.pkl")
            models["hidden_cs_lgbm_h60"] = lgbm

    if not models:
        raise RuntimeError("no model trained")

    print("\npredicting full dataset")
    predict_full(data_path, cfg, base_cols, feature_cols, models)


if __name__ == "__main__":
    main()
