#!/usr/bin/env python3
"""
Train 20 second-batch deep-learning cross-sectional hidden factors.

Design:
    5 new deep-learning models x 4 horizons = 20 factors

Models:
    lstm
    cnn_gru
    temporal_transformer
    feature_transformer
    denoise_mlp

Horizons:
    h30, h60, h90, h120

Input:
    data/raw/factor_features_20241022_20241122_100.csv
    data/raw/feature_cols_20241022_20241122_100.yaml

Output:
    outputs/hidden_factor_cs_dl_<model>_h<horizon>_<tag>.csv
    outputs/ml_cs_dl_v2_hidden_factors_20_<tag>.csv
    outputs/eval_cs_dl_v2_factors_20_<tag>.csv
    models/cs_dl_v2_<model>_h<horizon>_<tag>.pt

Run from FactorModel root:

    cd /home/fwz/projects/HFT_010-dev_fwz/FactorModel

    nohup python -u src/train_cs_dl_factors_20_v2.py \
      --input data/raw/factor_features_20241022_20241122_100.csv \
      --feature_cols data/raw/feature_cols_20241022_20241122_100.yaml \
      --tag 20241022_20241122_100 \
      > logs/train_cs_dl_factors_20_v2_20241022_20241122_100.log 2>&1 &
"""

import os
import gc
import math
import yaml
import argparse
import warnings
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_feature_cols(path: str) -> List[str]:
    with open(path, "r") as f:
        obj = yaml.safe_load(f)

    if isinstance(obj, dict) and "features" in obj:
        return list(obj["features"])
    if isinstance(obj, list):
        return list(obj)

    raise RuntimeError(f"Unsupported feature yaml format: {path}")


def safe_corr(x, y, method="spearman"):
    tmp = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()

    if len(tmp) < 3:
        return np.nan
    if tmp["x"].nunique() < 2 or tmp["y"].nunique() < 2:
        return np.nan

    if method == "spearman":
        return tmp["x"].corr(tmp["y"], method="spearman")
    return tmp["x"].corr(tmp["y"], method="pearson")


def add_cs_labels(df: pd.DataFrame, horizons: List[int], group_col: str = "datetime") -> pd.DataFrame:
    print("creating cross-sectional labels...", flush=True)

    for h in horizons:
        raw_col = f"label_{h}"
        cs_col = f"label_{h}_cs"
        rank_col = f"label_{h}_rank"

        if raw_col not in df.columns:
            raise RuntimeError(f"missing label column: {raw_col}")

        mean = df.groupby(group_col, sort=False)[raw_col].transform("mean")
        std = df.groupby(group_col, sort=False)[raw_col].transform("std")

        df[cs_col] = ((df[raw_col] - mean) / (std + 1e-8)).astype("float32")

        df[rank_col] = (
            df.groupby(group_col, sort=False)[raw_col]
              .rank(pct=True)
              .astype("float32")
            - 0.5
        )

        print(f"  {cs_col}, {rank_col}", flush=True)

    return df


def preprocess_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    group_col: str = "datetime",
    clip_q: float = 0.01,
    do_cs_zscore: bool = True,
) -> pd.DataFrame:
    print("preprocessing features...", flush=True)

    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")

    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if clip_q is not None and clip_q > 0:
        print(f"  global clipping: {clip_q}, {1 - clip_q}", flush=True)
        q_low = df[feature_cols].quantile(clip_q)
        q_high = df[feature_cols].quantile(1 - clip_q)
        df[feature_cols] = df[feature_cols].clip(lower=q_low, upper=q_high, axis=1)

    if do_cs_zscore:
        print("  cross-sectional zscore by datetime", flush=True)
        g = df.groupby(group_col, sort=False)
        mean = g[feature_cols].transform("mean")
        std = g[feature_cols].transform("std")

        df[feature_cols] = ((df[feature_cols] - mean) / (std + 1e-8)).astype("float32")
        df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return df


class TabularDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, indices: np.ndarray):
        self.X = X
        self.y = y
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx], dtype=torch.float32)


class SequenceDataset(Dataset):
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        end_indices: np.ndarray,
        start_indices: np.ndarray,
    ):
        self.X = X
        self.y = y
        self.end_indices = np.asarray(end_indices, dtype=np.int64)
        self.start_indices = np.asarray(start_indices, dtype=np.int64)

    def __len__(self):
        return len(self.end_indices)

    def __getitem__(self, i):
        end = self.end_indices[i]
        start = self.start_indices[i]
        return torch.from_numpy(self.X[start:end + 1]), torch.tensor(self.y[end], dtype=torch.float32)


class DenoiseMLP(nn.Module):
    """
    Robust tabular neural model.

    During training:
        x_noisy = x + N(0, noise_std)

    During evaluation:
        raw x is used.
    """

    def __init__(
        self,
        input_dim: int,
        hidden1: int = 128,
        hidden2: int = 64,
        dropout: float = 0.10,
        noise_std: float = 0.03,
    ):
        super().__init__()
        self.noise_std = noise_std

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden2, 1),
        )

    def forward(self, x):
        if self.training and self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std
        return self.net(x).squeeze(-1)


class FeatureTransformer(nn.Module):
    """
    Feature Transformer for current cross-sectional factor vector.

    Treat each scalar factor as one token:
        x_i + feature embedding -> TransformerEncoder -> pooled representation -> prediction
    """

    def __init__(
        self,
        input_dim: int,
        embed_dim: int = 32,
        num_heads: int = 4,
        num_layers: int = 2,
        ff_dim: int = 64,
        dropout: float = 0.10,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim

        self.feature_embedding = nn.Parameter(torch.randn(input_dim, embed_dim) * 0.02)

        self.value_proj = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.ReLU(),
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )

        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, 1),
        )

    def forward(self, x):
        token = self.value_proj(x.unsqueeze(-1)) + self.feature_embedding.unsqueeze(0)
        z = self.encoder(token)
        pooled = z.mean(dim=1)
        return self.head(pooled).squeeze(-1)


class LSTMModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden: int = 64,
        num_layers: int = 1,
        dropout: float = 0.10,
    ):
        super().__init__()

        self.rnn = nn.LSTM(
            input_dim,
            hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        out, _ = self.rnn(x)
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)


class CNNGRUModel(nn.Module):
    """
    Local temporal convolution + recurrent model.

    Sequence:
        lookback x features
        -> Conv1D over time
        -> GRU
        -> prediction
    """

    def __init__(
        self,
        input_dim: int,
        conv_channels: int = 64,
        rnn_hidden: int = 64,
        kernel_size: int = 3,
        dropout: float = 0.10,
    ):
        super().__init__()

        pad = kernel_size // 2

        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, conv_channels, kernel_size=kernel_size, padding=pad),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv1d(conv_channels, conv_channels, kernel_size=kernel_size, padding=pad),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.gru = nn.GRU(
            conv_channels,
            rnn_hidden,
            num_layers=1,
            batch_first=True,
        )

        self.head = nn.Sequential(
            nn.LayerNorm(rnn_hidden),
            nn.Linear(rnn_hidden, rnn_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(rnn_hidden, 1),
        )

    def forward(self, x):
        # x: [B, T, F]
        z = x.transpose(1, 2)      # [B, F, T]
        z = self.conv(z)           # [B, C, T]
        z = z.transpose(1, 2)      # [B, T, C]
        out, _ = self.gru(z)
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)


class TemporalTransformer(nn.Module):
    """
    Lightweight Transformer over historical factor sequence.

    Sequence:
        lookback x features
        -> linear projection
        -> positional encoding
        -> TransformerEncoder
        -> last-token prediction
    """

    def __init__(
        self,
        input_dim: int,
        lookback: int = 10,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        ff_dim: int = 128,
        dropout: float = 0.10,
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, lookback, embed_dim) * 0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )

        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, 1),
        )

    def forward(self, x):
        # x: [B, T, F]
        z = self.input_proj(x)
        z = z + self.pos_embedding[:, :z.size(1), :]
        z = self.encoder(z)
        last = z[:, -1, :]
        return self.head(last).squeeze(-1)


def make_model(model_name: str, input_dim: int, args):
    if model_name == "lstm":
        return LSTMModel(
            input_dim=input_dim,
            hidden=args.rnn_hidden,
            num_layers=args.rnn_layers,
            dropout=args.dropout,
        )

    if model_name == "cnn_gru":
        return CNNGRUModel(
            input_dim=input_dim,
            conv_channels=args.cnn_channels,
            rnn_hidden=args.rnn_hidden,
            kernel_size=args.cnn_kernel_size,
            dropout=args.dropout,
        )

    if model_name == "temporal_transformer":
        return TemporalTransformer(
            input_dim=input_dim,
            lookback=args.lookback,
            embed_dim=args.temporal_embed_dim,
            num_heads=args.temporal_heads,
            num_layers=args.temporal_layers,
            ff_dim=args.temporal_ff_dim,
            dropout=args.dropout,
        )

    if model_name == "feature_transformer":
        return FeatureTransformer(
            input_dim=input_dim,
            embed_dim=args.feature_embed_dim,
            num_heads=args.feature_heads,
            num_layers=args.feature_layers,
            ff_dim=args.feature_ff_dim,
            dropout=args.dropout,
        )

    if model_name == "denoise_mlp":
        return DenoiseMLP(
            input_dim=input_dim,
            hidden1=args.mlp_hidden1,
            hidden2=args.mlp_hidden2,
            dropout=args.dropout,
            noise_std=args.noise_std,
        )

    raise RuntimeError(f"unknown model: {model_name}")


def sample_indices(indices: np.ndarray, max_rows: int, seed: int) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)

    if max_rows is None or max_rows <= 0 or len(indices) <= max_rows:
        return indices

    rng = np.random.default_rng(seed)
    return rng.choice(indices, size=max_rows, replace=False)


def make_sequence_index(df: pd.DataFrame, lookback: int) -> Tuple[Dict[int, int], np.ndarray]:
    """
    Build sequence index for temporal models.

    For each stock-day:
        valid sample t uses [t-lookback+1, ..., t]
    """
    print(f"building sequence index, lookback={lookback}...", flush=True)

    end_to_start = {}
    valid_ends = []

    for _, g in df.groupby(["date", "securityid"], sort=False):
        idx = g.index.to_numpy()

        if len(idx) < lookback:
            continue

        for pos in range(lookback - 1, len(idx)):
            end = int(idx[pos])
            start = int(idx[pos - lookback + 1])
            end_to_start[end] = start
            valid_ends.append(end)

    valid_ends = np.asarray(valid_ends, dtype=np.int64)
    print("sequence valid rows:", len(valid_ends), flush=True)

    return end_to_start, valid_ends


def train_model(model, train_loader, valid_loader, args, device):
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    criterion = nn.MSELoss()

    best_state = None
    best_valid = np.inf
    bad_count = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True).float()
            yb = yb.to(device, non_blocking=True).float()

            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()
            train_losses.append(float(loss.item()))

        model.eval()
        valid_losses = []

        with torch.no_grad():
            for xb, yb in valid_loader:
                xb = xb.to(device, non_blocking=True).float()
                yb = yb.to(device, non_blocking=True).float()

                pred = model(xb)
                loss = criterion(pred, yb)
                valid_losses.append(float(loss.item()))

        train_loss = float(np.mean(train_losses)) if train_losses else np.nan
        valid_loss = float(np.mean(valid_losses)) if valid_losses else np.nan

        print(
            f"epoch={epoch}, train_loss={train_loss:.8f}, valid_loss={valid_loss:.8f}",
            flush=True,
        )

        if valid_loss < best_valid:
            best_valid = valid_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_count = 0
        else:
            bad_count += 1

        if bad_count >= args.patience:
            print("early stopping", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_valid


def predict_tabular(model, X: np.ndarray, indices: np.ndarray, batch_size: int, device):
    model.eval()
    pred_all = np.full(len(X), np.nan, dtype=np.float32)

    dummy_y = np.zeros(len(X), dtype=np.float32)
    ds = TabularDataset(X, dummy_y, indices)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    cursor = 0

    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device, non_blocking=True).float()
            pred = model(xb).detach().cpu().numpy().astype(np.float32)

            bs = len(pred)
            idx = ds.indices[cursor:cursor + bs]
            pred_all[idx] = pred
            cursor += bs

    return pred_all


def predict_sequence(
    model,
    X: np.ndarray,
    end_indices: np.ndarray,
    start_indices: np.ndarray,
    batch_size: int,
    device,
):
    model.eval()
    pred_all = np.full(len(X), np.nan, dtype=np.float32)

    dummy_y = np.zeros(len(X), dtype=np.float32)
    ds = SequenceDataset(X, dummy_y, end_indices=end_indices, start_indices=start_indices)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    cursor = 0

    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device, non_blocking=True).float()
            pred = model(xb).detach().cpu().numpy().astype(np.float32)

            bs = len(pred)
            idx = ds.end_indices[cursor:cursor + bs]
            pred_all[idx] = pred
            cursor += bs

    return pred_all


def calc_timeseries_eval(
    df: pd.DataFrame,
    factor_col: str,
    label_col: str,
    group_col: str = "datetime",
    n_bins: int = 5,
) -> pd.DataFrame:
    rows = []

    for dt, g in df.groupby(group_col, sort=False):
        g = g.dropna(subset=[factor_col, label_col])

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

            tmp["q"] = pd.qcut(
                tmp[factor_col],
                q=n_bins,
                labels=False,
                duplicates="drop",
            )

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


def summarize_factor(
    df: pd.DataFrame,
    factor_col: str,
    raw_label_col: str,
    cs_label_col: str,
    split_name: str,
    n_bins: int = 5,
):
    d = df[df["split"] == split_name].copy()
    d = d.dropna(subset=[factor_col, raw_label_col, cs_label_col])

    ts_raw = calc_timeseries_eval(
        d,
        factor_col=factor_col,
        label_col=raw_label_col,
        group_col="datetime",
        n_bins=n_bins,
    )

    ts_cs = calc_timeseries_eval(
        d,
        factor_col=factor_col,
        label_col=cs_label_col,
        group_col="datetime",
        n_bins=n_bins,
    )

    def ir(s):
        if len(s) == 0:
            return np.nan
        return s.mean() / (s.std() + 1e-12)

    res = {
        "factor": factor_col,
        "split": split_name,
        "rows": len(d),
        "num_datetimes": d["datetime"].nunique(),

        "overall_ic_raw": safe_corr(d[factor_col], d[raw_label_col], method="pearson"),
        "overall_rankic_raw": safe_corr(d[factor_col], d[raw_label_col], method="spearman"),
        "overall_ic_cs": safe_corr(d[factor_col], d[cs_label_col], method="pearson"),
        "overall_rankic_cs": safe_corr(d[factor_col], d[cs_label_col], method="spearman"),

        "mean_cs_rankic_raw_label": ts_raw["rankic"].mean() if len(ts_raw) else np.nan,
        "cs_rankicir_raw_label": ir(ts_raw["rankic"]) if len(ts_raw) else np.nan,

        "mean_cs_rankic_cs_label": ts_cs["rankic"].mean() if len(ts_cs) else np.nan,
        "cs_rankicir_cs_label": ir(ts_cs["rankic"]) if len(ts_cs) else np.nan,

        "mean_long_short_raw_label": ts_raw["long_short"].mean() if len(ts_raw) else np.nan,
        "long_short_ir_raw_label": ir(ts_raw["long_short"]) if len(ts_raw) else np.nan,
    }

    for i in range(n_bins):
        c = f"group_{i + 1}"
        res[f"mean_{c}_raw_label"] = ts_raw[c].mean() if len(ts_raw) else np.nan

    return res


def save_single_factor(df: pd.DataFrame, out_path: str, factor_col: str, horizon: int):
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

    parser.add_argument("--horizons", default="30,60,90,120")
    parser.add_argument(
        "--models",
        default="lstm,cnn_gru,temporal_transformer,feature_transformer,denoise_mlp",
    )

    parser.add_argument("--train_start", type=int, default=20241022)
    parser.add_argument("--train_end", type=int, default=20241112)
    parser.add_argument("--valid_start", type=int, default=20241113)
    parser.add_argument("--valid_end", type=int, default=20241115)
    parser.add_argument("--test_start", type=int, default=20241118)
    parser.add_argument("--test_end", type=int, default=20241122)

    parser.add_argument("--clip_q", type=float, default=0.01)
    parser.add_argument("--no_cs_zscore_features", action="store_true")

    parser.add_argument("--max_train_rows", type=int, default=900_000)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--seq_batch_size", type=int, default=1024)
    parser.add_argument("--lookback", type=int, default=10)

    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--grad_clip", type=float, default=3.0)

    parser.add_argument("--rnn_hidden", type=int, default=64)
    parser.add_argument("--rnn_layers", type=int, default=1)

    parser.add_argument("--cnn_channels", type=int, default=64)
    parser.add_argument("--cnn_kernel_size", type=int, default=3)

    parser.add_argument("--temporal_embed_dim", type=int, default=64)
    parser.add_argument("--temporal_heads", type=int, default=4)
    parser.add_argument("--temporal_layers", type=int, default=2)
    parser.add_argument("--temporal_ff_dim", type=int, default=128)

    parser.add_argument("--feature_embed_dim", type=int, default=32)
    parser.add_argument("--feature_heads", type=int, default=4)
    parser.add_argument("--feature_layers", type=int, default=2)
    parser.add_argument("--feature_ff_dim", type=int, default=64)

    parser.add_argument("--mlp_hidden1", type=int, default=128)
    parser.add_argument("--mlp_hidden2", type=int, default=64)
    parser.add_argument("--noise_std", type=float, default=0.03)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_bins", type=int, default=5)
    parser.add_argument("--output_dir", default="outputs")

    args = parser.parse_args()

    set_seed(args.seed)

    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]
    model_names = [x.strip() for x in args.models.split(",") if x.strip()]

    allowed_models = {
        "lstm",
        "cnn_gru",
        "temporal_transformer",
        "feature_transformer",
        "denoise_mlp",
    }

    bad_models = [m for m in model_names if m not in allowed_models]
    if bad_models:
        raise RuntimeError(f"unsupported models: {bad_models}. allowed={sorted(allowed_models)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 100, flush=True)
    print("Cross-sectional deep-learning factor training V2", flush=True)
    print("device:", device, flush=True)
    print("input:", args.input, flush=True)
    print("feature_cols:", args.feature_cols, flush=True)
    print("models:", model_names, flush=True)
    print("horizons:", horizons, flush=True)
    print("tag:", args.tag, flush=True)
    print("=" * 100, flush=True)

    feature_cols = load_feature_cols(args.feature_cols)

    required_cols = ["date", "datetime", "securityid"] + feature_cols + [f"label_{h}" for h in horizons]

    header = pd.read_csv(args.input, nrows=0).columns.tolist()
    missing = [c for c in required_cols if c not in header]
    if missing:
        raise RuntimeError(f"missing columns: {missing[:30]}, total={len(missing)}")

    dtype = {
        "securityid": str,
        "datetime": str,
    }

    print("loading data...", flush=True)
    df = pd.read_csv(args.input, usecols=required_cols, dtype=dtype)

    df["date"] = df["date"].astype(int)
    df["securityid"] = df["securityid"].astype(str).str.zfill(6)
    df["datetime"] = df["datetime"].astype(str)

    df = df.sort_values(["date", "securityid", "datetime"]).reset_index(drop=True)

    print("raw shape:", df.shape, flush=True)
    print("date range:", df["date"].min(), df["date"].max(), flush=True)
    print("num dates:", df["date"].nunique(), flush=True)
    print("num stocks:", df["securityid"].nunique(), flush=True)
    print("num features:", len(feature_cols), flush=True)

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
    valid_idx_all = np.where(valid_mask.values)[0]
    all_indices = np.arange(len(df), dtype=np.int64)

    if len(train_idx_all) == 0:
        raise RuntimeError("empty train set")
    if len(valid_idx_all) == 0:
        raise RuntimeError("empty valid set")

    X_all = df[feature_cols].values.astype("float32")

    sequence_models = {"lstm", "cnn_gru", "temporal_transformer"}
    sequence_needed = any(m in sequence_models for m in model_names)

    end_to_start = None
    valid_seq_ends = None

    if sequence_needed:
        end_to_start, valid_seq_ends = make_sequence_index(df, lookback=args.lookback)

    output_factor_cols = []
    eval_rows = []

    for model_name in model_names:
        is_seq = model_name in sequence_models

        for h in horizons:
            print("\n" + "=" * 100, flush=True)
            print(f"training model={model_name}, horizon=h{h}", flush=True)

            raw_label_col = f"label_{h}"
            cs_label_col = f"label_{h}_cs"
            factor_col = f"hidden_factor_cs_dl_{model_name}_h{h}"

            y_all = df[cs_label_col].values.astype("float32")

            if is_seq:
                seq_train_idx = np.array(
                    [i for i in valid_seq_ends if train_mask.values[i] and np.isfinite(y_all[i])],
                    dtype=np.int64,
                )
                seq_valid_idx = np.array(
                    [i for i in valid_seq_ends if valid_mask.values[i] and np.isfinite(y_all[i])],
                    dtype=np.int64,
                )

                train_idx = sample_indices(
                    seq_train_idx,
                    max_rows=args.max_train_rows,
                    seed=args.seed + h + len(model_name),
                )
                valid_idx = seq_valid_idx

                train_starts = np.array([end_to_start[int(i)] for i in train_idx], dtype=np.int64)
                valid_starts = np.array([end_to_start[int(i)] for i in valid_idx], dtype=np.int64)

                train_ds = SequenceDataset(X_all, y_all, train_idx, train_starts)
                valid_ds = SequenceDataset(X_all, y_all, valid_idx, valid_starts)

                batch_size = args.seq_batch_size

            else:
                train_idx = train_idx_all[np.isfinite(y_all[train_idx_all])]
                valid_idx = valid_idx_all[np.isfinite(y_all[valid_idx_all])]

                train_idx = sample_indices(
                    train_idx,
                    max_rows=args.max_train_rows,
                    seed=args.seed + h + len(model_name),
                )

                train_ds = TabularDataset(X_all, y_all, train_idx)
                valid_ds = TabularDataset(X_all, y_all, valid_idx)

                batch_size = args.batch_size

            print("train rows used:", len(train_idx), flush=True)
            print("valid rows:", len(valid_idx), flush=True)

            train_loader = DataLoader(
                train_ds,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                drop_last=False,
            )

            valid_loader = DataLoader(
                valid_ds,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                drop_last=False,
            )

            model = make_model(model_name, input_dim=len(feature_cols), args=args)

            model, best_valid = train_model(
                model,
                train_loader=train_loader,
                valid_loader=valid_loader,
                args=args,
                device=device,
            )

            print("best_valid_loss:", best_valid, flush=True)
            print("predicting all rows...", flush=True)

            if is_seq:
                pred_end_idx = valid_seq_ends
                pred_start_idx = np.array([end_to_start[int(i)] for i in pred_end_idx], dtype=np.int64)

                pred = predict_sequence(
                    model,
                    X_all,
                    end_indices=pred_end_idx,
                    start_indices=pred_start_idx,
                    batch_size=args.seq_batch_size,
                    device=device,
                )

            else:
                pred = predict_tabular(
                    model,
                    X_all,
                    indices=all_indices,
                    batch_size=args.batch_size,
                    device=device,
                )

            df[factor_col] = pred
            output_factor_cols.append(factor_col)

            single_path = f"{args.output_dir}/hidden_factor_cs_dl_{model_name}_h{h}_{args.tag}.csv"
            save_single_factor(df, single_path, factor_col=factor_col, horizon=h)
            print("saved single factor:", single_path, flush=True)

            model_path = f"models/cs_dl_v2_{model_name}_h{h}_{args.tag}.pt"
            ensure_dir(model_path)
            torch.save(model.state_dict(), model_path)
            print("saved model:", model_path, flush=True)

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
                    "best_valid_loss": best_valid,
                    "target_raw_label": raw_label_col,
                    "target_cs_label": cs_label_col,
                    "single_factor_path": single_path,
                    "model_path": model_path,
                })
                eval_rows.append(row)

            del model, train_loader, valid_loader, train_ds, valid_ds, pred
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    keep_cols = ["date", "datetime", "securityid"]

    for h in horizons:
        keep_cols.extend([
            f"label_{h}",
            f"label_{h}_cs",
            f"label_{h}_rank",
        ])

    keep_cols.extend(output_factor_cols)
    keep_cols.append("split")

    merged_output = f"{args.output_dir}/ml_cs_dl_v2_hidden_factors_{len(output_factor_cols)}_{args.tag}.csv"
    ensure_dir(merged_output)
    df[keep_cols].to_csv(merged_output, index=False)

    eval_output = f"{args.output_dir}/eval_cs_dl_v2_factors_{len(output_factor_cols)}_{args.tag}.csv"
    eval_df = pd.DataFrame(eval_rows)
    eval_df.to_csv(eval_output, index=False)

    print("\nsaved merged DL V2 factor file:", merged_output, flush=True)
    print("saved eval summary:", eval_output, flush=True)

    print("\n=== TEST SET RANKING ===", flush=True)
    test_eval = eval_df[eval_df["split"] == "test"].copy()
    test_eval = test_eval.sort_values(
        ["mean_cs_rankic_raw_label", "mean_long_short_raw_label"],
        ascending=False,
    )

    display_cols = [
        "factor",
        "model",
        "horizon",
        "rows",
        "num_datetimes",
        "mean_cs_rankic_raw_label",
        "cs_rankicir_raw_label",
        "mean_long_short_raw_label",
        "long_short_ir_raw_label",
        "overall_rankic_raw",
        "overall_rankic_cs",
        "best_valid_loss",
    ]

    print(test_eval[display_cols].to_string(index=False), flush=True)
    print("\nDONE.", flush=True)


if __name__ == "__main__":
    main()
