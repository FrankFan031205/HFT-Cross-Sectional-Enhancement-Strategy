import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import yaml


def root():
    return Path(__file__).resolve().parents[2]


def rpath(p):
    p = Path(p)
    return p if p.is_absolute() else root() / p


def load_yaml(p):
    with open(p, "r") as f:
        return yaml.safe_load(f)


def first_col(cols, candidates, required=True):
    for c in candidates:
        if c in cols:
            return c
    if required:
        raise ValueError(f"missing columns from candidates={candidates}, available={list(cols)}")
    return None


def norm_sid(s):
    return s.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)


def norm_side(s, buy_values, sell_values):
    buy = set(str(x).lower() for x in buy_values)
    sell = set(str(x).lower() for x in sell_values)
    x = s.astype(str).str.lower()
    out = pd.Series("unknown", index=s.index)
    out[x.isin(buy)] = "buy"
    out[x.isin(sell)] = "sell"
    return out


def bucket_rank(x):
    if pd.isna(x):
        return "no_cs_match"
    x = float(x)
    if x < 0.2:
        return "rank_00_20"
    if x < 0.4:
        return "rank_20_40"
    if x < 0.6:
        return "rank_40_60"
    if x < 0.7:
        return "rank_60_70"
    if x < 0.8:
        return "rank_70_80"
    if x < 0.9:
        return "rank_80_90"
    return "rank_90_100"


def load_cs(path, cfg):
    c = cfg["columns"]
    cols = ["cs_rank", "cs_score_z", "cs_group", "target_inventory_scale",
            "long_candidate", "avoid_candidate", "reduce_candidate"]
    usecols = [c["datetime_col"], c["symbol_col"]] + cols

    print(f"loading cs signal: {path}")
    df = pd.read_csv(path, usecols=usecols, low_memory=False)
    df = df.rename(columns={c["datetime_col"]: "datetime", c["symbol_col"]: "securityid"})
    df["datetime"] = df["datetime"].astype(str)
    df["securityid"] = norm_sid(df["securityid"])
    df["cs_rank"] = pd.to_numeric(df["cs_rank"], errors="coerce")
    df["cs_score_z"] = pd.to_numeric(df["cs_score_z"], errors="coerce")
    df["target_inventory_scale"] = pd.to_numeric(df["target_inventory_scale"], errors="coerce")
    print("cs shape:", df.shape)
    return df


def load_trades(path, cfg, name):
    print(f"loading {name}: {path}")
    head = pd.read_csv(path, nrows=5)
    cols = head.columns

    c = cfg["columns"]
    dt_col = c["datetime_col"]
    sid_col = c["symbol_col"]
    side_col = first_col(cols, c["side_candidates"])
    pnl_col = first_col(cols, c["pnl_candidates"])
    bps_col = first_col(cols, c["bps_candidates"], required=False)
    notional_col = first_col(cols, c["notional_candidates"], required=False)

    usecols = [dt_col, sid_col, side_col, pnl_col]
    for x in [bps_col, notional_col]:
        if x is not None and x not in usecols:
            usecols.append(x)

    df = pd.read_csv(path, usecols=usecols, low_memory=False)

    rename = {
        dt_col: "datetime",
        sid_col: "securityid",
        side_col: "raw_side",
        pnl_col: "net_pnl",
    }
    if bps_col is not None:
        rename[bps_col] = "net_pnl_bps"
    if notional_col is not None:
        rename[notional_col] = "notional"

    df = df.rename(columns=rename)
    df["datetime"] = df["datetime"].astype(str)
    df["securityid"] = norm_sid(df["securityid"])

    sv = cfg["side_values"]
    df["side_norm"] = norm_side(df["raw_side"], sv["buy"], sv["sell"])
    df["net_pnl"] = pd.to_numeric(df["net_pnl"], errors="coerce")

    if "net_pnl_bps" in df.columns:
        df["net_pnl_bps"] = pd.to_numeric(df["net_pnl_bps"], errors="coerce")
    if "notional" in df.columns:
        df["notional"] = pd.to_numeric(df["notional"], errors="coerce")

    df["source"] = name
    print(name, "shape:", df.shape)
    print(name, "side counts:")
    print(df["side_norm"].value_counts(dropna=False))
    return df


def summarize(df, group_cols):
    agg = {
        "num_trades": ("net_pnl", "count"),
        "total_net_pnl": ("net_pnl", "sum"),
        "avg_net_pnl":

        "win_rate": ("net_pnl", lambda x: (x > 0).mean

        "avg_target_inventory_scale": ("target_inventory_scal
    }
    if "n
        agg["avg_net_pnl
        agg["media
    if "notional" i
        agg



def analyz
    print(f"\nmerging cs signal for {name}")
 



    overall 
    by_side = s
    by_bucket = summarize(m, ["source", "side_n


    buy_bucket = 

    matched_buy = buy[buy[
    rows = []
    for th in [0.2


        rows.append({
            "source": nam
            "threshold":
            "matched_
            "kept_bu
            "fil
            "kept_buy_ratio": len(keep) / len(matched_buy) if len(matched_buy) else np.nan,
     
            "ke


            "fil
            "filtered_win_rate": (filt["net_pnl"] > 0).mean() if len(filt)



    outdir.mkdir(parents=True, exist_ok=T
    overall.to_csv(outdir / f"{name}_overall.csv"
    by_side.to_csv(outdir / f"{name}_by_side.csv", 
    by_bucket.to_csv(outdir / f"{
    buy_bucket.to_csv(outdir / f"{name}_buy_b
    threshold.to_csv(outdir 
    m.head(200000).to_csv(outd

    print(f"\


    print(threshold

    return buy_bu


def main():
    ap = argpa
    ap.add_argum
    args = ap.p

    cfg = load_yaml(rpath(
    outdir = rp
    outdir.mkdir(parents=Tr

    cs = load_

    baseline = load_trades
    base_bucket

    cs_path = rpath(cfg["d
    if cs_path.
        csf = load_trades(c
        cs_buck

    report = outdir / "cs_
    with open(report, "w


        f.write(base_bu
        f.write("\n\n#
        f.write(base_th.to_markdown(index=False))
        f.write("\n")
  
    print("saved report:", report)


if __name__ == "__main__":
    
