import sys
import argparse
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from CrossSectionalOptimizer.src.io_utils import (
    load_config,
    resolve_path,
    ensure_parent,
    read_csv_smart,
)
from CrossSectionalOptimizer.src.optimizer import solve_one_timestamp
from CrossSectionalOptimizer.src.quote_mapping import add_quote_mapping


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="CrossSectionalOptimizer/config/optimizer_1min.yaml")
    parser.add_argument("--max_timestamps", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(resolve_path(args.config))
    c = cfg["columns"]
    d = cfg["data"]

    datetime_col = c["datetime_col"]
    symbol_col = c["symbol_col"]

    input_path = resolve_path(d["optimizer_input_path"])
    if not input_path.exists():
        raise FileNotFoundError(f"optimizer input not found: {input_path}")

    print("reading optimizer input:", input_path)
    df = read_csv_smart(input_path)

    df[datetime_col] = pd.to_datetime(df[datetime_col], errors="coerce")
    df[symbol_col] = df[symbol_col].astype(str).str.extract(r"(\d+)")[0].str.zfill(6)

    df = df.dropna(subset=[datetime_col, symbol_col])
    df = df.sort_values([datetime_col, symbol_col])

    all_timestamps = df[datetime_col].drop_duplicates().sort_values().tolist()

    if args.max_timestamps is not None:
        use_timestamps = all_timestamps[:args.max_timestamps]
        use_set = set(use_timestamps)
        df = df[df[datetime_col].isin(use_set)].copy()
    else:
        use_timestamps = all_timestamps

    print("num timestamps to optimize:", len(use_timestamps))
    print("input rows used:", len(df))

    prev_weight_map = {}
    outputs = []
    diagnostics = []

    grouped = df.groupby(datetime_col, sort=True)

    for k, (ts, df_t) in enumerate(grouped, 1):
        df_t = df_t.copy()

        solved = solve_one_timestamp(df_t, prev_weight_map, cfg)
        solved[datetime_col] = ts

        merged = df_t.merge(solved, on=[datetime_col, symbol_col], how="left")
        merged["target_weight"] = merged["target_weight"].fillna(0.0)

        merged = add_quote_mapping(merged, cfg)

        prev_weight_map = dict(zip(
            merged[symbol_col].astype(str),
            merged["target_weight"].astype(float)
        ))

        outputs.append(merged)

        diagnostics.append({
            "datetime": ts,
            "n_names": len(merged),
            "gross": merged["target_weight"].abs().sum(),
            "net": merged["target_weight"].sum(),
            "max_abs_weight": merged["target_weight"].abs().max(),
            "quote_rate": (merged["quote_intensity"] > 0).mean(),
            "status_top": merged["optimizer_status"].mode().iloc[0] if "optimizer_status" in merged.columns else "unknown",
            "size_exposure": (merged["target_weight"] * merged.get("size_z", 0.0)).sum(),
            "liquidity_exposure": (merged["target_weight"] * merged.get("liquidity_z", 0.0)).sum(),
            "volatility_exposure": (merged["target_weight"] * merged.get("volatility_z", 0.0)).sum(),
        })

        if k % 100 == 0:
            print(f"optimized {k}/{len(use_timestamps)} timestamps")

    if not outputs:
        raise ValueError("no optimizer output generated")

    out = pd.concat(outputs, ignore_index=True)
    diag = pd.DataFrame(diagnostics)

    out_path = ensure_parent(d["optimizer_output_path"])
    diag_path = ensure_parent(d["diagnostic_path"])

    out.to_csv(out_path, index=False)
    diag.to_csv(diag_path, index=False)

    print("saved optimizer output:", out_path)
    print("saved diagnostics:", diag_path)
    print("output shape:", out.shape)
    print(out[[datetime_col, symbol_col, "target_weight", "quote_intensity", "bid_aggressiveness", "ask_aggressiveness", "optimizer_status"]].head())


if __name__ == "__main__":
    main()
