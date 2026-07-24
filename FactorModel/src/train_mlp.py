import os
import argparse
import random
import yaml
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from scipy.stats import pearsonr, spearmanr


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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_mask(df, date_col, start, end):
    return (df[date_col] >= start) & (df[date_col] <= end)


def calc_ic(y_true, y_pred):
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 3:
        return np.nan, np.nan

    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if np.unique(y_true).size < 2 or np.unique(y_pred).size < 2:
        return np.nan, np.nan

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


class MLP(nn.Module):
    def __init__(self, n_features, hidden_dims, dropout=0.1, batch_norm=True):
        super().__init__()

        layers = []
        in_dim = n_features

        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h

        layers.append(nn.Linear(in_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, layers_per_block=2, dropout=0.1, layer_norm=True):
        super().__init__()

        layers = []
        for _ in range(layers_per_block):
            if layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.net(x)


class ResidualMLP(nn.Module):
    def __init__(
        self,
        n_features,
        input_proj_dim=128,
        hidden_dim=128,
        num_blocks=30,
        layers_per_block=2,
        dropout=0.1,
        layer_norm=True,
    ):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(n_features, input_proj_dim),
            nn.ReLU(),
            nn.Linear(input_proj_dim, hidden_dim),
            nn.ReLU(),
        )

        self.blocks = nn.Sequential(*[
            ResidualBlock(
                hidden_dim=hidden_dim,
                layers_per_block=layers_per_block,
                dropout=dropout,
                layer_norm=layer_norm,
            )
            for _ in range(num_blocks)
        ])

        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = self.input_proj(x)
        h = self.blocks(h)
        return self.out(h).squeeze(-1)


def evaluate_loss(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_n = 0

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            pred = model(xb)
            loss = criterion(pred, yb)

            n = xb.shape[0]
            total_loss += loss.item() * n
            total_n += n

    return total_loss / max(total_n, 1)


def predict_array(model, X, batch_size, device):
    model.eval()

    preds = []

    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i:i + batch_size]).to(device)
            pred = model(xb)
            preds.append(pred.detach().cpu().numpy())

    return np.concatenate(preds)


def build_model(cfg, n_features):
    model_type = cfg["model"]["type"]
    params = cfg["model"].get("params", {})

    if model_type == "mlp":
        return MLP(
            n_features=n_features,
            hidden_dims=params.get("hidden_dims", [128, 64]),
            dropout=float(params.get("dropout", 0.1)),
            batch_norm=bool(params.get("batch_norm", True)),
        )

    if model_type == "residual_mlp":
        return ResidualMLP(
            n_features=n_features,
            input_proj_dim=int(params.get("input_proj_dim", 128)),
            hidden_dim=int(params.get("hidden_dim", 128)),
            num_blocks=int(params.get("num_blocks", 30)),
            layers_per_block=int(params.get("layers_per_block", 2)),
            dropout=float(params.get("dropout", 0.1)),
            layer_norm=bool(params.get("layer_norm", True)),
        )

    raise RuntimeError(f"unknown model type: {model_type}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["project"].get("seed", 42))

    data_cfg = cfg["data"]
    train_cfg = cfg["train"]

    date_col = data_cfg.get("date_col", "date")
    datetime_col = data_cfg["datetime_col"]
    symbol_col = data_cfg["symbol_col"]
    label_col = data_cfg["label_col"]

    feature_cols = load_features(data_cfg["feature_cols_path"])

    usecols = [date_col, datetime_col, symbol_col, label_col] + feature_cols

    print("loading processed data:", data_cfg["processed_data_path"], flush=True)
    df = pd.read_csv(data_cfg["processed_data_path"], usecols=usecols)

    print("data shape:", df.shape, flush=True)
    print("date range:", df[date_col].min(), df[date_col].max(), flush=True)
    print("num features:", len(feature_cols), flush=True)
    print("label:", label_col, flush=True)

    train_mask = make_mask(df, date_col, data_cfg["train_start"], data_cfg["train_end"])
    valid_mask = make_mask(df, date_col, data_cfg["valid_start"], data_cfg["valid_end"])
    test_mask = make_mask(df, date_col, data_cfg["test_start"], data_cfg["test_end"])

    print("train rows:", int(train_mask.sum()), flush=True)
    print("valid rows:", int(valid_mask.sum()), flush=True)
    print("test rows:", int(test_mask.sum()), flush=True)

    if train_mask.sum() == 0:
        raise RuntimeError("empty train set")
    if valid_mask.sum() == 0:
        raise RuntimeError("empty valid set")
    if test_mask.sum() == 0:
        raise RuntimeError("empty test set")

    X_train = df.loc[train_mask, feature_cols].values.astype("float32")
    y_train = df.loc[train_mask, label_col].values.astype("float32")

    X_valid = df.loc[valid_mask, feature_cols].values.astype("float32")
    y_valid = df.loc[valid_mask, label_col].values.astype("float32")

    X_test = df.loc[test_mask, feature_cols].values.astype("float32")
    y_test = df.loc[test_mask, label_col].values.astype("float32")

    label_scale = float(train_cfg.get("label_scale", 10000.0))

    y_train_scaled = y_train * label_scale
    y_valid_scaled = y_valid * label_scale

    requested_device = train_cfg.get("device", "cpu")
    if requested_device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("device:", device, flush=True)

    batch_size = int(train_cfg.get("batch_size", 32768))
    num_workers = int(train_cfg.get("num_workers", 0))
    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train_scaled)),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    valid_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_valid), torch.from_numpy(y_valid_scaled)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    model = build_model(cfg, len(feature_cols)).to(device)

    criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-5)),
    )

    epochs = int(train_cfg.get("epochs", 8))
    patience = int(train_cfg.get("patience", 2))

    best_valid_loss = float("inf")
    best_state = None
    bad_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()

        total_loss = 0.0
        total_n = 0

        for step, (xb, yb) in enumerate(train_loader, 1):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            n = xb.shape[0]
            total_loss += loss.item() * n
            total_n += n

            if step % 100 == 0:
                print(
                    f"epoch {epoch} step {step} "
                    f"train_loss={total_loss / total_n:.8f}",
                    flush=True,
                )

        train_loss = total_loss / max(total_n, 1)
        valid_loss = evaluate_loss(model, valid_loader, criterion, device)

        print(
            f"epoch {epoch} finished "
            f"train_loss={train_loss:.8f} "
            f"valid_loss={valid_loss:.8f}",
            flush=True,
        )

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print("early stopping", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model_path = train_cfg["model_path"]
    ensure_dir(model_path)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_cols": feature_cols,
            "config": cfg,
            "label_scale": label_scale,
        },
        model_path,
    )
    print("saved model:", model_path, flush=True)

    output_col = cfg["prediction"]["output_col"]

    print("predicting train...", flush=True)
    pred_train = predict_array(model, X_train, batch_size, device) / label_scale

    print("predicting valid...", flush=True)
    pred_valid = predict_array(model, X_valid, batch_size, device) / label_scale

    print("predicting test...", flush=True)
    pred_test = predict_array(model, X_test, batch_size, device) / label_scale

    df[output_col] = np.nan
    df["split"] = "other"

    df.loc[train_mask, output_col] = pred_train
    df.loc[valid_mask, output_col] = pred_valid
    df.loc[test_mask, output_col] = pred_test

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

        mse = np.mean((y_true - y_pred) ** 2)
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
    print(metrics_df, flush=True)

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
    print("saved prediction:", pred_path, flush=True)

    metrics_path = cfg["output"]["metrics_path"]
    ensure_dir(metrics_path)
    metrics_df.to_csv(metrics_path, index=False)
    print("saved metrics:", metrics_path, flush=True)


if __name__ == "__main__":
    main()