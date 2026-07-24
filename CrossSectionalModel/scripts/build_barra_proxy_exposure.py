import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def read_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def mkdir_parent(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def find_col(cols, candidates):
    low = {c.lower(): c for c in cols}
    for x in candidates:
        if x.lower() in low:
            return low[x.lower()]
    return None


def normalize_date_value(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if not s:
        return np.nan
    try:
        if "-" in s or "/" in s:
            return int(pd.to_datetime(s).strftime("%Y%m%d"))
        return int(float(s))
    except Exception:
        try:
            return int(pd.to_datetime(s).strftime("%Y%m%d"))
        except Exception:
            return np.nan


def make_securityid_from_ticker(s):
    raw = s.astype("string").str.strip()
    extracted = raw.str.extract(r"(\d+)")[0]
    return pd.to_numeric(extracted, errors="coerce").astype("Int64")


def cs_zscore(df, col, out_col):
    x = pd.to_numeric(df[col], errors="coerce")
    mean = x.groupby(df["date"]).transform("mean")
    std = x.groupby(df["date"]).transform("std")
    z = (x - mean) / std.replace(0, np.nan)
    df[out_col] = z.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def load_daily_market(daily_market_dir, start_date, end_date):
    daily_market_dir = Path(daily_market_dir)
    files = sorted(daily_market_dir.glob("daily_market_*.csv"))

    if not files:
        raise FileNotFoundError(f"no daily_market_*.csv found in {daily_market_dir}")

    dfs = []

    for i, p in enumerate(files, 1):
        header = pd.read_csv(p, nrows=0)
        cols = header.columns.tolist()

        ticker_col = find_col(cols, ["ticker", "Ticker"])
        secid_col = find_col(cols, ["SecID", "secID"])
        trade_date_col = find_col(cols, ["tradeDate", "date"])

        close_col = find_col(cols, ["closePrice", "close"])
        accum_col = find_col(cols, ["accumAdjFactor"])

        market_value_col = find_col(cols, ["marketValue", "totalMarketValue"])
        neg_market_value_col = find_col(cols, ["negMarketValue", "negMarketvalue"])
        turnover_rate_col = find_col(cols, ["turnoverRate"])
        turnover_value_col = find_col(cols, ["turnoverValue", "dealAmount"])
        pb_col = find_col(cols, ["PB", "pb"])
        pe_col = find_col(cols, ["PE", "pe", "PE1"])

        usecols = []
        for c in [
            ticker_col,
            secid_col,
            trade_date_col,
            close_col,
            accum_col,
            market_value_col,
            neg_market_value_col,
            turnover_rate_col,
            turnover_value_col,
            pb_col,
            pe_col,
        ]:
            if c is not None and c not in usecols:
                usecols.append(c)

        if ticker_col is None and secid_col is None:
            continue

        df = pd.read_csv(p, usecols=usecols)

        if ticker_col is not None:
            df["securityid"] = make_securityid_from_ticker(df[ticker_col])
            df["ticker"] = df[ticker_col].astype(str).str.zfill(6)
        else:
            df["securityid"] = make_securityid_from_ticker(df[secid_col])
            df["ticker"] = df[secid_col].astype(str).str.extract(r"(\d+)")[0].str.zfill(6)

        if trade_date_col is not None:
            df["date"] = df[trade_date_col].map(normalize_date_value)
        else:
            s = p.stem.replace("daily_market_", "")
            df["date"] = normalize_date_value(s)

        df = df.dropna(subset=["date", "securityid"])
        df["date"] = df["date"].astype(int)
        df["securityid"] = df["securityid"].astype(int)

        if close_col is not None:
            df["closePrice"] = pd.to_numeric(df[close_col], errors="coerce")
        else:
            df["closePrice"] = np.nan

        if accum_col is not None:
            df["accumAdjFactor"] = pd.to_numeric(df[accum_col], errors="coerce")
        else:
            df["accumAdjFactor"] = 1.0

        if market_value_col is not None:
            df["marketValue"] = pd.to_numeric(df[market_value_col], errors="coerce")
        else:
            df["marketValue"] = np.nan

        if neg_market_value_col is not None:
            df["negMarketValue"] = pd.to_numeric(df[neg_market_value_col], errors="coerce")
        else:
            df["negMarketValue"] = np.nan

        if turnover_rate_col is not None:
            df["turnoverRate"] = pd.to_numeric(df[turnover_rate_col], errors="coerce")
        else:
            df["turnoverRate"] = np.nan

        if turnover_value_col is not None:
            df["turnoverValue"] = pd.to_numeric(df[turnover_value_col], errors="coerce")
        else:
            df["turnoverValue"] = np.nan

        if pb_col is not None:
            df["PB"] = pd.to_numeric(df[pb_col], errors="coerce")
        else:
            df["PB"] = np.nan

        if pe_col is not None:
            df["PE"] = pd.to_numeric(df[pe_col], errors="coerce")
        else:
            df["PE"] = np.nan

        keep = [
            "date",
            "securityid",
            "ticker",
            "closePrice",
            "accumAdjFactor",
            "marketValue",
            "negMarketValue",
            "turnoverRate",
            "turnoverValue",
            "PB",
            "PE",
        ]

        dfs.append(df[keep])

        if i % 20 == 0:
            print(f"loaded daily_market files: {i}/{len(files)}")

    if not dfs:
        raise RuntimeError("no valid daily market data loaded")

    out = pd.concat(dfs, ignore_index=True)
    out = out[(out["date"] >= int(start_date)) & (out["date"] <= int(end_date))]
    out = out.drop_duplicates(["date", "securityid"], keep="last")
    out = out.sort_values(["securityid", "date"]).reset_index(drop=True)

    return out


def load_industry(industry_path):
    industry_path = Path(industry_path)

    if not industry_path.exists():
        print(f"industry file not found: {industry_path}, use UNKNOWN industry")
        return None

    df = pd.read_csv(industry_path)
    cols = df.columns.tolist()

    ticker_col = find_col(cols, ["ticker"])
    secid_col = find_col(cols, ["SecID", "secID"])

    if ticker_col is not None:
        df["securityid"] = make_securityid_from_ticker(df[ticker_col])
    elif secid_col is not None:
        df["securityid"] = make_securityid_from_ticker(df[secid_col])
    else:
        return None

    df = df.dropna(subset=["securityid"])
    df["securityid"] = df["securityid"].astype(int)

    into_col = find_col(cols, ["intoDate"])
    out_col = find_col(cols, ["outDate"])
    isnew_col = find_col(cols, ["isNew"])

    if into_col is not None:
        df["intoDate"] = df[into_col].map(normalize_date_value)
    else:
        df["intoDate"] = 0

    if out_col is not None:
        df["outDate"] = df[out_col].map(normalize_date_value)
    else:
        df["outDate"] = 99991231

    df["intoDate"] = pd.to_numeric(df["intoDate"], errors="coerce").fillna(0).astype(int)
    df["outDate"] = pd.to_numeric(df["outDate"], errors="coerce").fillna(99991231).astype(int)

    if isnew_col is not None:
        df["isNew"] = pd.to_numeric(df[isnew_col], errors="coerce").fillna(0).astype(int)
    else:
        df["isNew"] = 0

    for col in ["industryID1", "industryName1", "industryID2", "industryName2", "industryID", "industryName"]:
        if col not in df.columns:
            df[col] = np.nan

    keep = [
        "securityid",
        "intoDate",
        "outDate",
        "isNew",
        "industryID1",
        "industryName1",
        "industryID2",
        "industryName2",
        "industryID",
        "industryName",
    ]

    return df[keep]


def merge_industry(panel, industry):
    if industry is None or len(industry) == 0:
        panel["industryID1"] = "UNKNOWN"
        panel["industryName1"] = "UNKNOWN"
        panel["industryID2"] = "UNKNOWN"
        panel["industryName2"] = "UNKNOWN"
        return panel

    keys = panel[["date", "securityid"]].drop_duplicates()
    tmp = keys.merge(industry, on="securityid", how="left")

    valid = (
        (tmp["date"] >= tmp["intoDate"].fillna(0))
        & (tmp["date"] <= tmp["outDate"].fillna(99991231))
    )

    tmp = tmp[valid].copy()

    if len(tmp) == 0:
        panel["industryID1"] = "UNKNOWN"
        panel["industryName1"] = "UNKNOWN"
        panel["industryID2"] = "UNKNOWN"
        panel["industryName2"] = "UNKNOWN"
        return panel

    tmp = tmp.sort_values(["date", "securityid", "isNew", "intoDate"])
    tmp = tmp.drop_duplicates(["date", "securityid"], keep="last")

    keep = [
        "date",
        "securityid",
        "industryID1",
        "industryName1",
        "industryID2",
        "industryName2",
        "industryID",
        "industryName",
    ]

    panel = panel.merge(tmp[keep], on=["date", "securityid"], how="left")

    panel["industryID1"] = panel["industryID1"].fillna(panel["industryID"]).fillna("UNKNOWN").astype(str)
    panel["industryName1"] = panel["industryName1"].fillna(panel["industryName"]).fillna("UNKNOWN").astype(str)
    panel["industryID2"] = panel["industryID2"].fillna("UNKNOWN").astype(str)
    panel["industryName2"] = panel["industryName2"].fillna("UNKNOWN").astype(str)

    panel = panel.drop(columns=[c for c in ["industryID", "industryName"] if c in panel.columns])

    return panel


def build_barra_proxy(cfg):
    data_cfg = cfg["data"]
    date_cfg = cfg["date"]

    start_date = int(date_cfg["start_date"])
    end_date = int(date_cfg["end_date"])

    daily = load_daily_market(data_cfg["daily_market_dir"], start_date, end_date)

    daily["adj_close"] = daily["closePrice"] * daily["accumAdjFactor"].fillna(1.0)

    g = daily.groupby("securityid", group_keys=False)
    daily["ret_1d"] = g["adj_close"].pct_change()
    daily["momentum_20d_raw"] = g["adj_close"].pct_change(20)
    daily["volatility_20d_raw"] = g["ret_1d"].rolling(20, min_periods=5).std().reset_index(level=0, drop=True)

    mv = daily["marketValue"].where(daily["marketValue"].notna(), daily["negMarketValue"])
    daily["size_raw"] = np.log(pd.to_numeric(mv, errors="coerce").where(lambda x: x > 0))

    tv = daily["turnoverValue"]
    daily["liquidity_raw"] = np.log(pd.to_numeric(tv, errors="coerce").where(lambda x: x > 0))
    daily.loc[daily["liquidity_raw"].isna(), "liquidity_raw"] = pd.to_numeric(
        daily.loc[daily["liquidity_raw"].isna(), "turnoverRate"], errors="coerce"
    )

    pb = pd.to_numeric(daily["PB"], errors="coerce")
    pe = pd.to_numeric(daily["PE"], errors="coerce")
    daily["value_raw"] = -np.log(pb.where(pb > 0))
    daily.loc[daily["value_raw"].isna(), "value_raw"] = -np.log(pe.where(pe > 0))

    daily["momentum_raw"] = daily["momentum_20d_raw"]
    daily["volatility_raw"] = daily["volatility_20d_raw"]

    for raw, z in [
        ("size_raw", "size_z"),
        ("liquidity_raw", "liquidity_z"),
        ("value_raw", "value_z"),
        ("momentum_raw", "momentum_z"),
        ("volatility_raw", "volatility_z"),
    ]:
        daily[raw] = pd.to_numeric(daily[raw], errors="coerce")
        cs_zscore(daily, raw, z)

    industry = load_industry(data_cfg["industry_path"])
    daily = merge_industry(daily, industry)

    keep = [
        "date",
        "securityid",
        "ticker",
        "industryID1",
        "industryName1",
        "industryID2",
        "industryName2",
        "marketValue",
        "negMarketValue",
        "turnoverRate",
        "turnoverValue",
        "PB",
        "PE",
        "size_z",
        "liquidity_z",
        "value_z",
        "momentum_z",
        "volatility_z",
    ]

    out = daily[keep].copy()
    output_path = data_cfg["output_path"]
    mkdir_parent(output_path)
    out.to_csv(output_path, index=False)

    print("===== barra proxy summary =====")
    print("output:", output_path)
    print("rows:", len(out))
    print("date range:", out["date"].min(), out["date"].max())
    print("num dates:", out["date"].nunique())
    print("num securities:", out["securityid"].nunique())
    print("columns:", out.columns.tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = read_yaml(args.config)
    build_barra_proxy(cfg)


if __name__ == "__main__":
    main()
