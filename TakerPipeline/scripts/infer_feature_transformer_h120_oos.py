import argparse
import inspect
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "FactorModel" / "src"))

from train_cs_dl_factors_20_v2 import FeatureTransformer, preprocess_features, add_cs_labels


def ensure_dir_for_file(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def infer_feature_cols(raw_columns, horizon):
    key_cols = {"date", "datetime", "securityid"}
    label_cols = {f"label_{h}" for h in [30, 60, 90, 120]}
    label_cols.add(f"label_{horizon}")
    cols = []
    for c in raw_columns:
        if c in key_cols:
            continue
        if c in label_cols:
            continue
        if str(c).startswith("label_"):
            continue
        cols.append(c)
    return cols


def call_preprocess_features(df, feature_cols, clip_q):
    sig = inspect.signature(preprocess_features)
    params = sig.parameters

    kwargs = {}
    if "feature_cols" in params:
        kwargs["feature_cols"] = feature_cols
    if "group_col" in params:
        kwargs["group_col"] = "datetime"
    if "clip_q" in params:
        kwargs["clip_q"] = clip_q

    # training default is cs-zscore enabled unless --no_cs_zscore_features is passed
    for name in [
        "do_cs_zscore",
        "cs_zscore",
        "cs_zscore_features",
        "use_cs_zscore",
        "enable_cs_zscore",
    ]:
        if name in params:
            kwargs[name] = True

    try:
        return preprocess_features(df, **kwargs)
    except TypeError:
        # fallback for older signature
        return preprocess_features(
            df,
            feature_cols=feature_cols,
            group_col="datetime",
            clip_q=clip_q,
        )


def read_oos_raw(raw_path, feature_cols, horizon, start_date, end_date, chunksize):
    usecols = ["date", "datetime", "securityid"] + feature_cols + [f"label_{horizon}"]
    parts = []

    print("reading raw feature file:", raw_path, flush=True)
    print("usecols:", len(usecols), flush=True)
    print("date filter:", start_date, end_date, flush=True)

    for i, chunk in enumerate(pd.read_csv(raw_path, usecols=usecols, chunksize=chunksize, low_memory=False)):
        chunk["date"] = chunk["date"].astype(int)
        chunk = chunk[(chunk["date"] >= start_date) & (chunk["date"] <= end_date)].copy()

        if not chunk.empty:
            parts.append(chunk)

        if i % 10 == 0:
            print("  chunk", i, "kept rows:", len(chunk), flush=True)

    if not parts:
        raise RuntimeError("No OOS rows found in raw feature file.")

    df = pd.concat(parts, ignore_index=True)

    df["date"] = df["date"].astype(int)
    df["securityid"] = df["securityid"].astype(str).str.zfill(6)
    df["datetime"] = df["datetime"].astype(str)

    df = df.sort_values(["date", "securityid", "datetime"]).reset_index(drop=True)

    print("OOS raw shape:", df.shape, flush=True)
    print("OOS date range:", df["date"].min(), df["date"].max(), flush=True)
    print("OOS symbols:", df["securityid"].nunique(), flush=True)
    print("OOS datetimes:", df["datetime"].nunique(), flush=True)

    return df


def make_model_from_state_dict(ckpt, device):
    input_dim = ckpt["feature_embedding"].shape[0]
    embed_dim = ckpt["feature_embedding"].shape[1]

    # From checkpoint shapes:
    # encoder.layers.0.linear1.weight = (ff_dim, embed_dim)
    ff_dim = ckpt["encoder.layers.0.linear1.weight"].shape[0]

    # in_proj_weight = (3 * embed_dim, embed_dim), heads not directly stored.
    # Training default is 4 heads.
    model = FeatureTransformer(
        input_dim=input_dim,
        embed_dim=embed_dim,
        num_heads=4,
        num_layers=2,
        ff_dim=ff_dim,
        dropout=0.10,
    )

    model.load_state_dict(ckpt, strict=True)
    model.to(device)
    model.eval()

    print("loaded FeatureTransformer:", flush=True)
    print("  input_dim:", input_dim, flush=True)
    print("  embed_dim:", embed_dim, flush=True)
    print("  ff_dim:", ff_dim, flush=True)
    print("  num_layers: 2", flush=True)
    print("  num_heads: 4", flush=True)

    return model


@torch.no_grad()
def predict(model, X, batch_size, device):
    X = np.asarray(X, dtype=np.float32)
    ds = TensorDataset(torch.from_numpy(X))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    preds = []
    for i, (xb,) in enumerate(loader):
        xb = xb.to(device)
        y = model(xb).detach().cpu().numpy()
        preds.append(y)

        if i % 200 == 0:
            print("  pred batch:", i, flush=True)

    return np.concatenate(preds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--raw_feature_path", default=None)
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--chunksize", type=int, default=1_000_000)
    parser.add_argument("--clip_q", type=float, default=0.01)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    project_root = Path(cfg["paths"]["project_root"])
    model_path = cfg["model"]["frozen_model_path"]
    out_path = cfg["model"]["oos_hidden_factor_path"]

    raw_path = args.raw_feature_path
    if raw_path is None:
        raw_path = str(project_root / "FactorModel" / "data" / "raw" / "factor_features_20241022_20250114_742.csv")

    horizon = int(cfg["model"]["horizon"])
    start_date = int(cfg["dates"]["test_start_date"])
    end_date = int(cfg["dates"]["test_end_date"])

    ensure_dir_for_file(out_path)

    header = pd.read_csv(raw_path, nrows=0).columns.tolist()
    feature_cols = infer_feature_cols(header, horizon)

    print("=" * 100, flush=True)
    print("Frozen FeatureTransformer OOS inference", flush=True)
    print("raw_path:", raw_path, flush=True)
    print("model_path:", model_path, flush=True)
    print("out_path:", out_path, flush=True)
    print("horizon:", horizon, flush=True)
    print("num inferred feature cols:", len(feature_cols), flush=True)
    print("feature cols sample:", feature_cols[:10], flush=True)
    print("=" * 100, flush=True)

    ckpt = torch.load(model_path, map_location="cpu")
    expected_input_dim = ckpt["feature_embedding"].shape[0]

    if len(feature_cols) != expected_input_dim:
        raise RuntimeError(
            f"feature dim mismatch: inferred {len(feature_cols)}, checkpoint expects {expected_input_dim}"
        )

    feature_cols_path = str(Path(out_path).with_suffix(".feature_cols.yaml"))
    with open(feature_cols_path, "w") as f:
        yaml.safe_dump({"feature_cols": feature_cols}, f, sort_keys=False)
    print("saved feature cols:", feature_cols_path, flush=True)

    df = read_oos_raw(
        raw_path=raw_path,
        feature_cols=feature_cols,
        horizon=horizon,
        start_date=start_date,
        end_date=end_date,
        chunksize=args.chunksize,
    )

    print("adding cs label for output only...", flush=True)
    df = add_cs_labels(df, [horizon], group_col="datetime")

    print("preprocessing features...", flush=True)
    df = call_preprocess_features(df, feature_cols=feature_cols, clip_q=args.clip_q)

    X = df[feature_cols].astype("float32").values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("device:", device, flush=True)
    model = make_model_from_state_dict(ckpt, device=device)

    print("predicting...", flush=True)
    pred = predict(model, X, batch_size=args.batch_size, device=device)

    factor_col = "hidden_factor_cs_dl_feature_transformer_h120"
    df[factor_col] = pred.astype("float32")
    df["split"] = "oos"

    out_cols = [
        "date",
        "datetime",
        "securityid",
        f"label_{horizon}",
        f"label_{horizon}_cs",
        factor_col,
        "split",
    ]

    out = df[out_cols].copy()

    # match previous files: allow pandas to read securityid as int later if it wants,
    # but keep zero-padded string in csv.
    out.to_csv(out_path, index=False)

    print("saved:", out_path, flush=True)
    print("output shape:", out.shape, flush=True)
    print("date range:", out["date"].min(), out["date"].max(), flush=True)
    print("symbols:", out["securityid"].nunique(), flush=True)
    print("factor describe:", flush=True)
    print(out[factor_col].describe(), flush=True)


if __name__ == "__main__":
    main()
