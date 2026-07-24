#!/usr/bin/env python3
import os
import glob
import subprocess
import pandas as pd

PROJECT_ROOT = "/mnt/data1/fwz/HFT_010-dev_fwz"
FACTOR_DIR = f"{PROJECT_ROOT}/FactorGenerate"

UNIV_DIR = f"{PROJECT_ROOT}/FactorModel/data/raw/csi1000_daily_universe_from_market_return"
OUT_DIR = f"{PROJECT_ROOT}/FactorModel/data/raw/factor_features_week2_20_csi1000_by_date"
LOG_DIR = f"{PROJECT_ROOT}/FactorModel/logs/factor_features_week2_20_csi1000_by_date"

FINAL_OUT = f"{PROJECT_ROOT}/FactorModel/data/raw/factor_features_week2_20_20241022_20250114_csi1000.csv"
FEATURE_YAML = "../FactorModel/data/raw/feature_cols_week2_20.yaml"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

univ_files = sorted(glob.glob(f"{UNIV_DIR}/factor_features_*_csi1000.csv"))
if not univ_files:
    raise RuntimeError(f"no daily universe files found in {UNIV_DIR}")

failed = []

for univ_path in univ_files:
    date = int(os.path.basename(univ_path).split("_")[2])

    out_path = f"{OUT_DIR}/factor_features_week2_20_{date}_csi1000.csv"
    log_path = f"{LOG_DIR}/dump_factor_features_week2_20_{date}_csi1000.log"

    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        try:
            head = pd.read_csv(out_path, nrows=5)
            if "date" in head.columns and len(head.columns) > 10:
                print(f"[SKIP] {date} exists", flush=True)
                continue
        except Exception:
            pass

    n = pd.read_csv(univ_path, usecols=["securityid"], dtype={"securityid": str})["securityid"].nunique()

    print("\n" + "=" * 100, flush=True)
    print("[RUN]", date, "universe size:", n, flush=True)
    print("univ:", univ_path, flush=True)
    print("out:", out_path, flush=True)
    print("log:", log_path, flush=True)

    cmd = [
        "python", "-u", "dump_factor_features_fixed_universe.py",
        "--start", str(date),
        "--end", str(date),
        "--universe_file", os.path.relpath(univ_path, FACTOR_DIR),
        "--feature_yaml", FEATURE_YAML,
        "--output", os.path.relpath(out_path, FACTOR_DIR),
        "--overwrite",
    ]

    with open(log_path, "w") as f:
        ret = subprocess.run(
            cmd,
            cwd=FACTOR_DIR,
            stdout=f,
            stderr=subprocess.STDOUT,
        )

    if ret.returncode != 0:
        print("[FAILED]", date, "returncode:", ret.returncode, flush=True)
        failed.append(date)
        try:
            lines = open(log_path).readlines()
            print("last 80 lines:")
            print("".join(lines[-80:]))
        except Exception:
            pass
        break

    print("[DONE]", date, flush=True)

if failed:
    print("failed dates:", failed, flush=True)
    raise SystemExit(1)

print("\n[CONCAT]", flush=True)

daily_files = sorted(glob.glob(f"{OUT_DIR}/factor_features_week2_20_*_csi1000.csv"))

if os.path.exists(FINAL_OUT):
    os.remove(FINAL_OUT)

first = True
total = 0

for path in daily_files:
    print("concat:", os.path.basename(path), flush=True)
    for chunk in pd.read_csv(path, chunksize=1_000_000, dtype={"securityid": str}):
        chunk.to_csv(FINAL_OUT, mode="w" if first else "a", header=first, index=False)
        first = False
        total += len(chunk)
        print("written:", total, flush=True)

print("FINAL:", FINAL_OUT, flush=True)
print("rows:", total, flush=True)
