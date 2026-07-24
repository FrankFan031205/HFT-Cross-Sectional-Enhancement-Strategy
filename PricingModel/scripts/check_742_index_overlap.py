import os
import re
import sys
import subprocess
import pandas as pd

UNIVERSE_PATH = "PricingModel/data/universe_742.csv"
OUT_DIR = "PricingModel/data/index_overlap"

INDEX_MAP = {
    "hs300": "000300",    # 沪深300，对应 IF
    "csi500": "000905",   # 中证500，对应 IC
    "csi1000": "000852",  # 中证1000，对应 IM
}

INDEX_NAME = {
    "hs300": "沪深300",
    "csi500": "中证500",
    "csi1000": "中证1000",
}

os.makedirs(OUT_DIR, exist_ok=True)

def norm_code(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    m = re.search(r"(\d{6})", s)
    return m.group(1) if m else None

def ensure_akshare():
    try:
        import akshare as ak
        return ak
    except ImportError:
        print("[INFO] akshare not installed, installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "akshare"])
        import akshare as ak
        return ak

def find_code_col(df):
    candidates = [
        "成分券代码", "品种代码", "证券代码", "股票代码", "代码",
        "con_code", "symbol", "securityid", "SecurityID"
    ]
    for c in candidates:
        if c in df.columns:
            return c

    best_col, best_cnt = None, -1
    for c in df.columns:
        cnt = df[c].astype(str).str.extract(r"(\d{6})")[0].notna().sum()
        if cnt > best_cnt:
            best_col, best_cnt = c, cnt
    return best_col

def find_name_col(df):
    candidates = ["成分券名称", "品种名称", "证券简称", "股票简称", "名称", "name"]
    for c in candidates:
        if c in df.columns:
            return c
    return None

def load_universe():
    u = pd.read_csv(UNIVERSE_PATH, dtype=str)

    if "securityid" in u.columns:
        col = "securityid"
    elif "SecurityID" in u.columns:
        col = "SecurityID"
    else:
        # fallback: 找最像6位股票代码的列
        best_col, best_cnt = None, -1
        for c in u.columns:
            cnt = u[c].astype(str).str.extract(r"(\d{6})")[0].notna().sum()
            if cnt > best_cnt:
                best_col, best_cnt = c, cnt
        col = best_col

    if col is None:
        raise RuntimeError(f"Cannot find security id column in {UNIVERSE_PATH}; cols={list(u.columns)}")

    out = pd.DataFrame()
    out["securityid"] = u[col].map(norm_code)
    out = out.dropna(subset=["securityid"]).drop_duplicates("securityid")
    out = out.sort_values("securityid").reset_index(drop=True)
    return out

def get_index_members(ak, index_code):
    try:
        df = ak.index_stock_cons_csindex(symbol=index_code)
    except Exception as e1:
        print(f"[WARN] index_stock_cons_csindex failed for {index_code}: {e1}")
        try:
            df = ak.index_stock_cons_sina(symbol=index_code)
        except Exception as e2:
            raise RuntimeError(f"failed to fetch {index_code}: {e1}; {e2}")

    code_col = find_code_col(df)
    name_col = find_name_col(df)

    if code_col is None:
        raise RuntimeError(f"Cannot find code column for {index_code}; cols={list(df.columns)}")

    out = pd.DataFrame()
    out["securityid"] = df[code_col].map(norm_code)
    out["name"] = df[name_col].astype(str) if name_col else ""
    out = out.dropna(subset=["securityid"]).drop_duplicates("securityid")
    out = out.sort_values("securityid").reset_index(drop=True)
    return out

def main():
    universe = load_universe()
    universe_set = set(universe["securityid"])

    print("[UNIVERSE]", UNIVERSE_PATH)
    print("universe_count =", len(universe))
    print(universe.head(20).to_string(index=False))

    ak = ensure_akshare()

    flags = universe.copy()
    summary_rows = []

    for key, index_code in INDEX_MAP.items():
        index_name = INDEX_NAME[key]

        print(f"\n[FETCH] {index_name} {index_code}")
        members = get_index_members(ak, index_code)
        member_set = set(members["securityid"])

        members["index_key"] = key
        members["index_code"] = index_code
        members["index_name"] = index_name

        overlap = members[members["securityid"].isin(universe_set)].copy()
        overlap = overlap.sort_values("securityid").reset_index(drop=True)

        flags[f"in_{key}"] = flags["securityid"].isin(member_set)

        member_path = os.path.join(OUT_DIR, f"{key}_{index_code}_members_latest.csv")
        overlap_path = os.path.join(OUT_DIR, f"universe_742_overlap_{key}_{index_code}.csv")

        members.to_csv(member_path, index=False)
        overlap.to_csv(overlap_path, index=False)

        summary_rows.append({
            "index_key": key,
            "index_code": index_code,
            "index_name": index_name,
            "index_member_count": len(members),
            "universe_count": len(universe),
            "overlap_count": len(overlap),
            "overlap_ratio_in_universe": len(overlap) / len(universe),
            "overlap_file": overlap_path,
        })

        print(f"[RESULT] {index_name}: {len(overlap)} / {len(universe)}")
        print(overlap[["securityid", "name"]].head(50).to_string(index=False))

    def index_group(row):
        tags = []
        if row["in_hs300"]:
            tags.append("hs300")
        if row["in_csi500"]:
            tags.append("csi500")
        if row["in_csi1000"]:
            tags.append("csi1000")
        return ",".join(tags) if tags else "none"

    flags["index_group"] = flags.apply(index_group, axis=1)

    flags_path = os.path.join(OUT_DIR, "universe_742_index_flags_latest.csv")
    summary_path = os.path.join(OUT_DIR, "universe_742_index_overlap_summary_latest.csv")

    flags.to_csv(flags_path, index=False)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(summary_path, index=False)

    print("\n===== SUMMARY =====")
    print(summary.to_string(index=False))

    print("\n===== GROUP COUNTS =====")
    print(flags["index_group"].value_counts().to_string())

    print("\n[SAVED]")
    print(summary_path)
    print(flags_path)

if __name__ == "__main__":
    main()
