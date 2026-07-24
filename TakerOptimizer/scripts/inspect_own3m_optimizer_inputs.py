# -*- coding: utf-8 -*-
from pathlib import Path
import pandas as pd
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", action="append", default=[
        "/mnt/data1/fwz/HFT_010-dev_fwz_outputs",
        "/mnt/data1/fwz/HFT_010-dev_fwz",
    ])
    ap.add_argument("--min-files", type=int, default=20)
    args = ap.parse_args()

    seen = set()
    dirs = []
    for r in args.root:
        r = Path(r)
        if not r.exists():
            continue
        for f in r.rglob("optimizer_input_*.parquet"):
            d = f.parent
            if d not in seen:
                seen.add(d)
                dirs.append(d)

    rows = []
    for d in sorted(dirs):
        fs = sorted(d.glob("optimizer_input_*.parquet"))
        if len(fs) < args.min_files:
            continue
        try:
            df = pd.read_parquet(fs[0])
            cols = df.columns.tolist()
            date_min = df["date"].min() if "date" in df.columns else ""
            date_max = df["date"].max() if "date" in df.columns else ""
            rows.append({
                "dir": str(d),
                "n_files": len(fs),
                "first_file": fs[0].name,
                "last_file": fs[-1].name,
                "shape0": df.shape,
                "date0_min": date_min,
                "date0_max": date_max,
                "has_h10": any("h10" in c.lower() for c in cols),
                "has_h20": any("h20" in c.lower() for c in cols),
                "pred_cols": [c for c in cols if "pred" in c.lower() or "score" in c.lower() or "alpha" in c.lower()],
                "core_cols": [c for c in ["date","datetime","sid","securityid","SecurityID","mid_price","tmid","benchmark_weight"] if c in cols],
            })
        except Exception as e:
            rows.append({
                "dir": str(d),
                "n_files": len(fs),
                "error": repr(e),
            })

    out = pd.DataFrame(rows).sort_values(["n_files","dir"], ascending=[False, True])
    pd.set_option("display.max_colwidth", 180)
    print(out.to_string(index=False))

if __name__ == "__main__":
    main()
