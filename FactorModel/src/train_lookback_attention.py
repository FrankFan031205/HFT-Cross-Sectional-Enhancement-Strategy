import os
import argparse
import random
import yaml
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

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


class LookbackDataset(Dataset):
    def __init__(self, X, y_scaled, end_indices, seq_len):
        self.X = X
        self.y_scaled = y_scaled
        self.end_indices = end_indices.astype(np.int64)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.end_indices)

    def __getitem__(self, idx):
        end = self.end_indices[idx]
        start = end - self.seq_len + 1

        x = self.X[start:end + 1]
        y = self.y_scaled[end]

        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32), end


class SelfAttentionBlock(nn.Module):
    def __init__(self, d_model, n_heads, ffn_dim, dropout=0.1):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, attn_mask=None, return_attn=False):
        h = self.norm1(x)

        if return_attn:
            try:
                attn_out, attn_w = self.attn(
                    h,
                    h,
                    h,
                    attn_mask=attn_mask,
                    need_weights=True,
                    average_attn_weights=True,
                )
            except TypeError:
                attn_out, attn_w = self.attn(
                    h,
                    h,
                    h,
                    attn_mask=attn_mask,
                    need_weights=True,
                )
        else:
            attn_out, attn_w = self.attn(
                h,
                h,
                h,
                attn_mask=attn_mask,
                need_weights=False,
            )

        x = x + attn_out
        x = x + self.ffn(self.norm2(x))

        if return_attn:
            return x, attn_w

        return x, None


class LookbackAttentionNN(nn.Module):
    def __init__(
        self,
        n_features,
        seq_len=60,
        d_model=64,
        n_heads=4,
        num_layers=2,
        ffn_dim=128,
        head_hidden_dims=(128, 64),
        dropout=0.1,
        batch_norm=True,
    ):
        super().__init__()

        self.seq_len = seq_len

        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)

        self.blocks = nn.ModuleList([
            SelfAttentionBlock(
                d_model=d_model,
                n_heads=n_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        layers = []
        in_dim = d_model

        for h in head_hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h

        layers.append(nn.Linear(in_dim, 1))
        self.head = nn.Sequential(*layers)

    def causal_mask(self, device):
        mask = torch.full((self.seq_len, self.seq_len), float("-inf"), device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask

    def forward(self, x, return_attention=False):
        batch_size, seq_len, _ = x.shape

        pos = torch.arange(seq_len, device=x.device)
        h = self.input_proj(x) + self.pos_emb(pos).unsqueeze(0)

        attn_mask = self.causal_mask(x.device)

        last_attn = None

        for i, block in enumerate(self.blocks):
            need_attn = return_attention and (i == len(self.blocks) - 1)
            h, attn_w = block(h, attn_mask=attn_mask, return_attn=need_attn)

            if need_attn:
                last_attn = attn_w

        last_state = h[:, -1, :]
        out = self.head(last_state).squeeze(-1)

        if return_attention:
            if last_attn is None:
                return out, None

            if last_attn.dim() == 4:
                last_attn = last_attn.mean(dim=1)

            last_query_attn = last_attn[:, -1, :]
            return out, last_query_attn

        return out


def build_sequence_indices(df, date_col, symbol_col, datetime_col, seq_len):
    end_indices = []

    for _, g in df.groupby([date_col, symbol_col], sort=False):
        idx = g.index.to_numpy()

        if len(idx) < seq_len:
            continue

        for j in range(seq_len - 1, len(idx)):
            end_indices.append(idx[j])

    return np.array(end_indices, dtype=np.int64)


def evaluate_loss(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_n = 0

    with torch.no_grad():
        for xb, yb, _ in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            pred = model(xb)
            loss = criterion(pred, yb)

            n = xb.shape[0]
            total_loss += loss.item() * n
            total_n += n

    return total_loss / max(total_n, 1)


def predict_loader(model, loader, label_scale, device, return_attention=False):
    model.eval()

    preds = []
    indices = []

    attn_sum = None
    attn_n = 0

    with torch.no_grad():
        for xb, _, end_idx in loader:
            xb = xb.to(device, non_blocking=True)

            if return_attention:
                pred, attn = model(xb, return_attention=True)

                if attn is not None:
                    attn_np = attn.detach().cpu().numpy()
                    if attn_sum is None:
                        attn_sum = attn_np.sum(axis=0)
                    else:
                        attn_sum += attn_np.sum(axis=0)
                    attn_n += attn_np.shape[0]
            else:
                pred = model(xb)

            preds.append(pred.detach().cpu().numpy() / label_scale)
            indices.append(end_idx.numpy())

    preds = np.concatenate(preds)
    indices = np.concatenate(indices)

    if return_attention and attn_sum is not None:
        mean_attn = attn_sum / max(attn_n, 1)
    else:
        mean_attn = None

    return indices, preds, mean_attn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["project"].get("seed", 42))

    data_cfg = cfg["data"]
    model_cfg = cfg["model"].get("params", {})
    train_cfg = cfg["train"]

    date_col = data_cfg.get("date_col", "date")
    datetime_col = data_cfg["datetime_col"]
    symbol_col = data_cfg["symbol_col"]
    label_col = data_cfg["label_col"]

    feature_cols = load_features(data_cfg["feature_cols_path"])

    seq_len = int(model_cfg.get("seq_len", 60))

    usecols = [date_col, datetime_col, symbol_col, label_col] + feature_cols

    print("loading processed data:", data_cfg["processed_data_path"], flush=True)
    df = pd.read_csv(data_cfg["processed_data_path"], usecols=usecols)

    df = df.sort_values([date_col, symbol_col, datetime_col]).reset_index(drop=True)

    print("data shape:", df.shape, flush=True)
    print("date range:", df[date_col].min(), df[date_col].max(), flush=True)
    print("num features:", len(feature_cols), flush=True)
    print("label:", label_col, flush=True)
    print("seq_len:", seq_len, flush=True)

    train_mask = make_mask(df, date_col, data_cfg["train_start"], data_cfg["train_end"]).values
    valid_mask = make_mask(df, date_col, data_cfg["valid_start"], data_cfg["valid_end"]).values
    test_mask = make_mask(df, date_col, data_cfg["test_start"], data_cfg["test_end"]).values

    print("building sequence indices...", flush=True)
    all_end_indices = build_sequence_indices(
        df=df,
        date_col=date_col,
        symbol_col=symbol_col,
        datetime_col=datetime_col,
        seq_len=seq_len,
    )

    train_end_indices = all_end_indices[train_mask[all_end_indices]]
    valid_end_indices = all_end_indices[valid_mask[all_end_indices]]
    test_end_indices = all_end_indices[test_mask[all_end_indices]]

    print("train sequences:", len(train_end_indices), flush=True)
    print("valid sequences:", len(valid_end_indices), flush=True)
    print("test sequences:", len(test_end_indices), flush=True)

    if len(train_end_indices) == 0:
        raise RuntimeError("empty train sequence set")
    if len(valid_end_indices) == 0:
        raise RuntimeError("empty valid sequence set")
    if len(test_end_indices) == 0:
        raise RuntimeError("empty test sequence set")

    X = df[feature_cols].values.astype("float32")
    y = df[label_col].values.astype("float32")

    label_scale = float(train_cfg.get("label_scale", 10000.0))
    y_scaled = y * label_scale

    requested_device = train_cfg.get("device", "cpu")
    if requested_device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("device:", device, flush=True)

    batch_size = int(train_cfg.get("batch_size", 4096))
    num_workers = int(train_cfg.get("num_workers", 0))
    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        LookbackDataset(X, y_scaled, train_end_indices, seq_len),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    valid_loader = DataLoader(
        LookbackDataset(X, y_scaled, valid_end_indices, seq_len),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        LookbackDataset(X, y_scaled, test_end_indices, seq_len),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    model = LookbackAttentionNN(
        n_features=len(feature_cols),
        seq_len=seq_len,
        d_model=int(model_cfg.get("d_model", 64)),
        n_heads=int(model_cfg.get("n_heads", 4)),
        num_layers=int(model_cfg.get("num_layers", 2)),
        ffn_dim=int(model_cfg.get("ffn_dim", 128)),
        head_hidden_dims=tuple(model_cfg.get("head_hidden_dims", [128, 64])),
        dropout=float(model_cfg.get("dropout", 0.1)),
        batch_norm=bool(model_cfg.get("batch_norm", True)),
    ).to(device)

    criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-5)),
    )

    epochs = int(train_cfg.get("epochs", 6))
    patience = int(train_cfg.get("patience", 2))

    best_valid_loss = float("inf")
    best_state = None
    bad_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()

        total_loss = 0.0
        total_n = 0

        for step, (xb, yb, _) in enumerate(train_loader, 1):
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
            "seq_len": seq_len,
        },
        model_path,
    )
    print("saved model:", model_path, flush=True)

    output_col = cfg["prediction"]["output_col"]

    df[output_col] = np.nan
    df["split"] = "other"

    df.loc[train_mask, "split"] = "train"
    df.loc[valid_mask, "split"] = "valid"
    df.loc[test_mask, "split"] = "test"

    print("predicting train...", flush=True)
    idx_train, pred_train, _ = predict_loader(
        model, train_loader, label_scale, device, return_attention=False
    )

    print("predicting valid...", flush=True)
    idx_valid, pred_valid, _ = predict_loader(
        model, valid_loader, label_scale, device, return_attention=False
    )

    print("predicting test...", flush=True)
    idx_test, pred_test, mean_attn = predict_loader(
        model, test_loader, label_scale, device, return_attention=True
    )

    df.loc[idx_train, output_col] = pred_train
    df.loc[idx_valid, output_col] = pred_valid
    df.loc[idx_test, output_col] = pred_test

    metrics = []

    for split_name, idx in [
        ("train", idx_train),
        ("valid", idx_valid),
        ("test", idx_test),
    ]:
        y_true = df.loc[idx, label_col].values.astype(float)
        y_pred = df.loc[idx, output_col].values.astype(float)

        mse = np.mean((y_true - y_pred) ** 2)
        ic, rankic = calc_ic(y_true, y_pred)

        cs = calc_cross_sectional_ic(
            df.loc[idx, [datetime_col, label_col, output_col]],
            pred_col=output_col,
            label_col=label_col,
            group_col=datetime_col,
        )

        metrics.append({
            "split": split_name,
            "rows": int(len(idx)),
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

    attention_path = cfg["output"].get("attention_path")
    if attention_path and mean_attn is not None:
        ensure_dir(attention_path)

        lag_names = [f"lag_{seq_len - 1 - i}" for i in range(seq_len)]

        attn_df = pd.DataFrame({
            "lag": lag_names,
            "position": list(range(seq_len)),
            "mean_attention_test": mean_attn,
        })

        attn_df.to_csv(attention_path, index=False)
        print("saved attention summary:", attention_path, flush=True)
        print(attn_df.sort_values("mean_attention_test", ascending=False).head(20), flush=True)


if __name__ == "__main__":
    main()