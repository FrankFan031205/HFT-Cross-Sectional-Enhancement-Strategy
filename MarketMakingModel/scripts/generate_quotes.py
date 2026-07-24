import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.strategy import FairValueMarketMaker


COLUMN_ALIASES = {
    "datetime": ["datetime", "time", "timestamp"],
    "securityid": ["securityid", "SecurityID", "symbol", "code"],
    "bid1": ["bid1", "bidprice1", "bid_price_1", "BidPrice1"],
    "ask1": ["ask1", "askprice1", "ask_price_1", "AskPrice1"],
    "mid_price": ["mid_price", "midprice", "mid"],
    "spread": ["spread"],
    "bid1_volume": ["bid1_volume", "bidvolume1", "bid_volume_1", "BidVolume1"],
    "ask1_volume": ["ask1_volume", "askvolume1", "ask_volume_1", "AskVolume1"],
    "limit_up_price": ["limit_up_price", "limit_up"],
    "limit_down_price": ["limit_down_price", "limit_down"],
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--market-data", default=None)
    p.add_argument("--prediction", default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--chunksize", type=int, default=500000)
    return p.parse_args()


def load_yaml(path: str) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def read_header(path: str) -> List[str]:
    return list(pd.read_csv(path, nrows=0).columns)


def resolve_col(
    available: List[str],
    configured: Optional[str],
    aliases: List[str],
    required: bool,
    role: str,
) -> Optional[str]:
    available_set = set(available)

    if configured and configured in available_set:
        return configured

    candidates = []
    if configured:
        candidates.append(configured)
    candidates.extend(aliases)

    for c in candidates:
        if c in available_set:
            return c

    if required:
        raise ValueError(
            f"Cannot find required column for {role}. "
            f"Configured={configured}, aliases={aliases}, available={available}"
        )

    return configured


def norm_datetime_key_series(s: pd.Series) -> pd.Series:
    def conv(x):
        if pd.isna(x):
            return ""

        t = str(x).strip()

        if "_" in t:
            date_part, time_part = t.split("_", 1)
            digits = "".join(ch for ch in time_part if ch.isdigit())

            if len(digits) >= 6:
                # Align 500ms market rows to 1-second prediction rows.
                # Examples:
                # 20241022_093000000 -> 20241022_093000000
                # 20241022_093000500 -> 20241022_093000000
                # 20241022_093001500 -> 20241022_093001000
                digits = digits[:6] + "000"
                return f"{date_part}_{digits}"

            return t

        return t

    return s.map(conv)


def norm_securityid_series(s: pd.Series) -> pd.Series:
    def conv(x):
        if pd.isna(x):
            return ""

        try:
            return str(int(float(x))).zfill(6)
        except Exception:
            t = str(x).strip()
            if t.endswith(".0"):
                t = t[:-2]
            if t.isdigit():
                return t.zfill(6)
            return t

    return s.map(conv)


def add_merge_keys(df: pd.DataFrame, datetime_col: str, symbol_col: str) -> pd.DataFrame:
    df = df.copy()
    df["_merge_datetime"] = norm_datetime_key_series(df[datetime_col])
    df["_merge_securityid"] = norm_securityid_series(df[symbol_col])
    return df


def update_config_columns(cfg: Dict, market_cols: List[str], pred_cols: Optional[List[str]]) -> Dict:
    c = cfg.setdefault("columns", {})

    c["datetime_col"] = resolve_col(
        market_cols,
        c.get("datetime_col", "datetime"),
        COLUMN_ALIASES["datetime"],
        True,
        "market datetime",
    )

    c["symbol_col"] = resolve_col(
        market_cols,
        c.get("symbol_col", "securityid"),
        COLUMN_ALIASES["securityid"],
        True,
        "market securityid",
    )

    c["bid_col"] = resolve_col(
        market_cols,
        c.get("bid_col", "bid1"),
        COLUMN_ALIASES["bid1"],
        True,
        "bid1",
    )

    c["ask_col"] = resolve_col(
        market_cols,
        c.get("ask_col", "ask1"),
        COLUMN_ALIASES["ask1"],
        True,
        "ask1",
    )

    c["mid_col"] = resolve_col(
        market_cols,
        c.get("mid_col", "mid_price"),
        COLUMN_ALIASES["mid_price"],
        False,
        "mid_price",
    )

    c["spread_col"] = resolve_col(
        market_cols,
        c.get("spread_col", "spread"),
        COLUMN_ALIASES["spread"],
        False,
        "spread",
    )

    c["bid_volume_col"] = resolve_col(
        market_cols,
        c.get("bid_volume_col", "bid1_volume"),
        COLUMN_ALIASES["bid1_volume"],
        False,
        "bid1_volume",
    )

    c["ask_volume_col"] = resolve_col(
        market_cols,
        c.get("ask_volume_col", "ask1_volume"),
        COLUMN_ALIASES["ask1_volume"],
        False,
        "ask1_volume",
    )

    c["limit_up_col"] = resolve_col(
        market_cols,
        c.get("limit_up_col", "limit_up_price"),
        COLUMN_ALIASES["limit_up_price"],
        False,
        "limit_up_price",
    )

    c["limit_down_col"] = resolve_col(
        market_cols,
        c.get("limit_down_col", "limit_down_price"),
        COLUMN_ALIASES["limit_down_price"],
        False,
        "limit_down_price",
    )

    pred_col = c.get("pred_col", "pred_ret")

    if pred_cols is not None:
        pred_aliases = [
            "pred_ret",
            "pred",
            "prediction",
            "y_pred",
            "score",
            "factor",
            "hidden_factor",
            "hidden_factor_attention_h60",
            "hidden_factor_mlp2_h60",
            "hidden_factor_mlp60_h60",
            "hidden_factor_lgbm_h60",
            "hidden_factor_ridge_h60",
        ]

        c["pred_col"] = resolve_col(
            pred_cols,
            pred_col,
            pred_aliases,
            True,
            "prediction",
        )
    else:
        c["pred_col"] = resolve_col(
            market_cols,
            pred_col,
            [
                "pred_ret",
                "pred",
                "prediction",
                "score",
                "hidden_factor",
                "hidden_factor_attention_h60",
                "hidden_factor_mlp2_h60",
                "hidden_factor_mlp60_h60",
                "hidden_factor_lgbm_h60",
                "hidden_factor_ridge_h60",
            ],
            True,
            "prediction",
        )

    return cfg


def load_prediction_table(pred_path: Optional[str], cfg: Dict) -> Optional[pd.DataFrame]:
    if pred_path is None:
        return None

    pred_path_obj = Path(pred_path)
    if not pred_path_obj.exists():
        raise FileNotFoundError(f"prediction file not found: {pred_path}")

    pred_cols = read_header(pred_path)

    pred_datetime = resolve_col(
        pred_cols,
        cfg["columns"].get("datetime_col", "datetime"),
        COLUMN_ALIASES["datetime"],
        True,
        "prediction datetime",
    )

    pred_symbol = resolve_col(
        pred_cols,
        cfg["columns"].get("symbol_col", "securityid"),
        COLUMN_ALIASES["securityid"],
        True,
        "prediction securityid",
    )

    pred_col = cfg["columns"]["pred_col"]

    if pred_col not in pred_cols:
        raise ValueError(
            f"prediction column {pred_col} not in {pred_path}. "
            f"available={pred_cols}"
        )

    keep = [pred_datetime, pred_symbol, pred_col]

    for extra in ["date", "label_60", "label_30", "label_90", "label_120", "split"]:
        if extra in pred_cols and extra not in keep:
            keep.append(extra)

    pred = pd.read_csv(pred_path, usecols=keep)

    pred = pred.rename(
        columns={
            pred_datetime: cfg["columns"]["datetime_col"],
            pred_symbol: cfg["columns"]["symbol_col"],
        }
    )

    pred = add_merge_keys(
        pred,
        cfg["columns"]["datetime_col"],
        cfg["columns"]["symbol_col"],
    )

    pred = pred.drop_duplicates(
        ["_merge_datetime", "_merge_securityid"],
        keep="last",
    )

    print("loaded prediction:", pred_path, pred.shape)
    print("prediction pred_col:", pred_col)
    print("prediction valid ratio:", pred[pred_col].notna().mean())

    return pred


def prepare_output_row(row_dict: Dict, decision_dict: Dict, output_cols: Optional[List[str]]) -> Dict:
    out = {}
    out.update(row_dict)
    out.update(decision_dict)

    out.pop("_merge_datetime", None)
    out.pop("_merge_securityid", None)

    if output_cols:
        for c in output_cols:
            if c not in out:
                out[c] = None
        return {c: out.get(c) for c in output_cols}

    return out


def generate_for_chunk(chunk: pd.DataFrame, strategy: FairValueMarketMaker, cfg: Dict) -> pd.DataFrame:
    output_cols = cfg.get("output", {}).get("columns", None)
    position_col = cfg.get("inventory", {}).get("position_col", None)

    rows = []

    for _, row in chunk.iterrows():
        position = None

        if position_col and position_col in row:
            try:
                position = int(row[position_col])
            except Exception:
                position = None

        decision = strategy.generate_quote(row, position=position)
        d = prepare_output_row(row.to_dict(), decision.to_dict(), output_cols)
        rows.append(d)

    return pd.DataFrame(rows)


def merge_prediction_into_chunk(chunk: pd.DataFrame, pred_table: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    pred_col = cfg["columns"]["pred_col"]

    chunk = chunk.merge(
        pred_table,
        on=["_merge_datetime", "_merge_securityid"],
        how="left",
        suffixes=("", "_pred"),
    )

    pred_col_ext = pred_col + "_pred"

    if pred_col_ext in chunk.columns:
        if pred_col in chunk.columns:
            chunk[pred_col] = chunk[pred_col].combine_first(chunk[pred_col_ext])
        else:
            chunk[pred_col] = chunk[pred_col_ext]

    if pred_col not in chunk.columns:
        raise ValueError(f"prediction column still missing after merge: {pred_col}")

    return chunk


def main():
    args = parse_args()

    cfg = load_yaml(args.config)

    market_path = args.market_data or cfg.get("data", {}).get("market_data_path")
    pred_path = args.prediction or cfg.get("data", {}).get("prediction_path")

    if market_path is None:
        raise ValueError("market_data_path is required")

    if not Path(market_path).exists():
        raise FileNotFoundError(f"market data not found: {market_path}")

    market_cols = read_header(market_path)
    pred_cols = read_header(pred_path) if pred_path else None

    cfg = update_config_columns(cfg, market_cols, pred_cols)

    pred_table = None
    if pred_path is not None:
        pred_table = load_prediction_table(pred_path, cfg)

    output_path = args.output
    if output_path is None:
        out_dir = cfg.get("data", {}).get("output_dir", ".")
        out_file = cfg.get("data", {}).get("output_file", "quote_decisions.csv")
        output_path = str(Path(out_dir) / out_file)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    strategy = FairValueMarketMaker(cfg)

    first = True
    total_written = 0
    chunk_id = 0

    market_datetime = cfg["columns"]["datetime_col"]
    market_symbol = cfg["columns"]["symbol_col"]
    pred_col = cfg["columns"]["pred_col"]

    print("market data:", market_path)
    print("prediction:", pred_path)
    print("output:", output_path)
    print("datetime_col:", market_datetime)
    print("symbol_col:", market_symbol)
    print("pred_col:", pred_col)

    for chunk in pd.read_csv(market_path, chunksize=args.chunksize):
        chunk = add_merge_keys(chunk, market_datetime, market_symbol)

        if pred_table is not None:
            chunk = merge_prediction_into_chunk(chunk, pred_table, cfg)

        if len(chunk) == 0:
            chunk_id += 1
            continue

        if args.max_rows is not None:
            remain = args.max_rows - total_written
            if remain <= 0:
                break
            chunk = chunk.head(remain)

        valid_pred_ratio = chunk[pred_col].notna().mean() if pred_col in chunk.columns else 0.0

        out = generate_for_chunk(chunk, strategy, cfg)

        out.to_csv(
            output_path,
            mode="w" if first else "a",
            header=first,
            index=False,
        )

        first = False
        total_written += len(out)

        bid_rate = out["quote_bid"].mean() if "quote_bid" in out.columns and len(out) else 0.0
        ask_rate = out["quote_ask"].mean() if "quote_ask" in out.columns and len(out) else 0.0

        risk_top = ""
        if "risk_state" in out.columns and len(out):
            vc = out["risk_state"].value_counts(dropna=False).head(3)
            risk_top = ", ".join([f"{k}:{v}" for k, v in vc.items()])

        print(
            f"chunk={chunk_id}, rows={len(out)}, total_written={total_written}, "
            f"valid_pred_ratio={valid_pred_ratio:.4f}, "
            f"bid_quote_rate={bid_rate:.4f}, ask_quote_rate={ask_rate:.4f}, "
            f"risk_top=[{risk_top}]"
        )

        chunk_id += 1

        if args.max_rows is not None and total_written >= args.max_rows:
            break

    print("saved:", output_path)
    print("total rows:", total_written)


if __name__ == "__main__":
    main()
