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
from CrossSectionalOptimizer.src.risk_exposure import (
    add_trade_date,
    add_style_exposures,
)


def normalize_symbol(df, symbol_col):
    if symbol_col not in df.columns:
        for cand in ["SecurityID", "security_id", "ticker", "symbol"]:
            if cand in df.columns:
                df = df.rename(columns={cand: symbol_col})
                break

    if symbol_col not in df.columns:
        raise ValueError(f"symbol column not found: {symbol_col}")

    df[symbol_col] = (
        df[symbol_col]
        .astype(str)
        .str.extract(r"(\d+)")[0]
        .str.zfill(6)
    )
    return df


def parse_datetime_series(s):
    x = s.astype(str).str.strip()

    # Main format in our project:
    # 20241022_093000000 = YYYYMMDD_HHMMSSmmm
    out = pd.to_datetime(x, format="%Y%m%d_%H%M%S%f", errors="coerce")

    # Fallback: remove non-digits and parse compact datetime
    mask = out.isna()
    if mask.any():
        y = x[mask].str.replace(r"\D", "", regex=True)

        parsed = pd.Series(pd.NaT, index=y.index, dtype="datetime64[ns]")

        formats = [
            (8, "%Y%m%d"),
            (12, "%Y%m%d%H%M"),
            (14, "%Y%m%d%H%M%S"),
            (17, "%Y%m%d%H%M%S%f"),
        ]

        for length, fmt in formats:
            m = y.str.len() == length
            if m.any():
                parsed.loc[m] = pd.to_datetime(y[m], format=fmt, errors="coerce")

        out.loc[mask] = parsed

    return out


def normalize_datetime(df, datetime_col):
    if datetime_col not in df.columns:
        for cand in ["timestamp", "time", "datetime"]:
            if cand in df.columns:
                df = df.rename(columns={cand: datetime_col})
                break

    if datetime_col not in df.columns:
        raise ValueError(f"datetime column not found: {datetime_col}")

    raw_sample = df[datetime_col].head(3).tolist()
    df[datetime_col] = parse_datetime_series(df[datetime_col])

    print("datetime raw sample:", raw_sample)
    print("datetime parsed sample:", df[datetime_col].dropna().head(3).tolist())
    print("datetime parse NaT:", df[datetime_col].isna().sum(), "/", len(df))

    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="CrossSectionalOptimizer/config/optimizer_1min.yaml")
    parser.add_argument("--max_rows", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(resolve_path(args.config))
    c = cfg["columns"]
    d = cfg["data"]

    datetime_col = c["datetime_col"]
    symbol_col = c["symbol_col"]
    alpha_col = c["alpha_col"]

    alpha_path = resolve_path(d["alpha_path"])
    market_path = resolve_path(d["market_path"])
    industry_map_path = resolve_path(d["industry_map_path"])

    if not alpha_path.exists():
        raise FileNotFoundError(f"alpha_path not found: {alpha_path}")

    read_kwargs = {}
    if args.max_rows is not None:
        read_kwargs["nrows"] = args.max_rows

    print("reading alpha:", alpha_path)
    alpha = read_csv_smart(alpha_path, **read_kwargs)
    alpha.columns = [x.strip() for x in alpha.columns]

    alpha = normalize_symbol(alpha, symbol_col)
    alpha = normalize_datetime(alpha, datetime_col)

    if alpha_col not in alpha.columns:
        candidates = [
            x for x in alpha.columns
            if ("hidden_factor" in x) or ("alpha" in x) or ("pred" in x)
        ]
        if not candidates:
            raise ValueError(f"alpha_col {alpha_col} not found. Columns: {list(alpha.columns)}")
        print(f"alpha_col {alpha_col} not found, using {candidates[0]} instead")
        alpha = alpha.rename(columns={candidates[0]: alpha_col})

    alpha = alpha.dropna(subset=[datetime_col, symbol_col])
    print("alpha shape:", alpha.shape)

    if market_path.exists():
        print("reading market:", market_path)
        market = read_csv_smart(market_path, **read_kwargs)
        market.columns = [x.strip() for x in market.columns]

        market = normalize_symbol(market, symbol_col)
        market = normalize_datetime(market, datetime_col)

        keep_cols = [datetime_col, symbol_col]

        candidate_market_cols = [
            c.get("mid_col", "mid_price"),
            c.get("spread_col", "spread"),
            c.get("market_value_col", "marketValue"),
            c.get("turnover_col", "turnoverRate"),
            c.get("volatility_col", "volatility_60"),
            "negMarketValue",
            "relative_spread",
            "ATR6",
            "ATR14",
            "VOL5",
            "VOL10",
            "VOL20",
        ]

        for col in candidate_market_cols:
            if col in market.columns and col not in keep_cols:
                keep_cols.append(col)

        market = market[keep_cols].drop_duplicates(subset=[datetime_col, symbol_col])
        print("market shape:", market.shape)
        print("market keep cols:", keep_cols)

        df = alpha.merge(market, on=[datetime_col, symbol_col], how="left")
    else:
        print("market_path not found, continue with alpha only:", market_path)
        df = alpha.copy()

    df = add_trade_date(df, datetime_col)

    if industry_map_path.exists():
        print("reading industry map:", industry_map_path)
        ind = read_csv_smart(industry_map_path)
        ind["securityid"] = ind["securityid"].astype(str).str.zfill(6)

        ind_keep = ["trade_date", "securityid", "industryID1", "industryName1"]
        ind = ind[ind_keep].drop_duplicates(subset=["trade_date", "securityid"])

        df = df.merge(
            ind,
            left_on=["trade_date", symbol_col],
            right_on=["trade_date", "securityid"],
            how="left",
            suffixes=("", "_ind")
        )

        if "securityid_ind" in df.columns:
            df = df.drop(columns=["securityid_ind"])
    else:
        print("industry map not found, use UNKNOWN industry")
        df["industryID1"] = "UNKNOWN"
        df["industryName1"] = "UNKNOWN"

    df["industryID1"] = df["industryID1"].fillna("UNKNOWN")
    df["industryName1"] = df["industryName1"].fillna("UNKNOWN")

    df = add_style_exposures(df, cfg)

    out_path = ensure_parent(d["optimizer_input_path"])
    df.to_csv(out_path, index=False)

    print("saved:", out_path)
    print("shape:", df.shape)
    print("num timestamps:", df[datetime_col].nunique())
    print("num stocks:", df[symbol_col].nunique())

    show_cols = [
        datetime_col,
        symbol_col,
        alpha_col,
        "industryID1",
        "industryName1",
        "size_z",
        "liquidity_z",
        "volatility_z",
    ]
    show_cols = [x for x in show_cols if x in df.columns]
    print(df[show_cols].head())


if __name__ == "__main__":
    main()
