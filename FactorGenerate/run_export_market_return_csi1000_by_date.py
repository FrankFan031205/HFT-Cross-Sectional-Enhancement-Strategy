#!/usr/bin/env python3
import os
import sys
import yaml
import shutil
import subprocess
from copy import deepcopy

from utils import get_date_security_info


PROJECT_ROOT = "/mnt/data1/fwz/HFT_010-dev_fwz"
FACTOR_DIR = f"{PROJECT_ROOT}/FactorGenerate"

TEMPLATE_CONFIG = f"{PROJECT_ROOT}/PricingModel/config/export_market_return_20241022_20241122_100.yaml"

UNIVERSE_FILE = "../PricingModel/data/universe_csi1000.csv"

START = 20241022
END = 20250114

OUT_DIR = f"{PROJECT_ROOT}/PricingModel/data/market_return_20241022_20250114_csi1000_by_date"
CONFIG_DIR = f"{PROJECT_ROOT}/PricingModel/config/export_market_return_csi1000_by_date"
LOG_DIR = f"{PROJECT_ROOT}/PricingModel/data/market_return_20241022_20250114_csi1000_by_date/logs"

FORCE = os.environ.get("FORCE", "0") == "1"


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def replace_string(x, date):
    y = x
    y = y.replace("/home/fwz/projects/HFT_010-dev_fwz", PROJECT_ROOT)
    y = y.replace("universe_100.csv", "universe_csi1000.csv")
    y = y.replace("universe_742.csv", "universe_csi1000.csv")
    y = y.replace("20241022_20241122_100", f"{date}_csi1000")
    y = y.replace("202410_100", f"{date}_csi1000")
    y = y.replace("_100.csv", "_csi1000.csv")
    y = y.replace("_742.csv", "_csi1000.csv")
    return y


def patch_dates_and_paths(obj, date):
    """
    Recursively patch common date/universe/path fields.
    This keeps compatibility with the existing export config format.
    """
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            lk = str(k).lower()

            if lk in {
                "start",
                "end",
                "start_date",
                "end_date",
                "begin",
                "begin_date",
                "finish",
                "finish_date",
                "date",
            } and isinstance(v, int):
                out[k] = int(date)

            elif "universe" in lk and isinstance(v, str):
                out[k] = UNIVERSE_FILE

            elif "sample" in lk and isinstance(v, int):
                out[k] = None

            else:
                out[k] = patch_dates_and_paths(v, date)

        return out

    if isinstance(obj, list):
        return [patch_dates_and_paths(v, date) for v in obj]

    if isinstance(obj, str):
        return replace_string(obj, date)

    return obj


def force_universe_fields(cfg):
    """
    Try to handle several possible universe config schemas.
    Existing load_universe() will use whichever keys it recognizes.
    """
    cfg.setdefault("universe", {})

    if isinstance(cfg["universe"], dict):
        cfg["universe"]["universe_file"] = UNIVERSE_FILE
        cfg["universe"]["file"] = UNIVERSE_FILE
        cfg["universe"]["path"] = UNIVERSE_FILE
        cfg["universe"]["csv_path"] = UNIVERSE_FILE
        cfg["universe"]["sample"] = None
        cfg["universe"]["sample_size"] = None
        cfg["universe"]["n_symbols"] = None

    return cfg


def make_config(template, date):
    cfg = patch_dates_and_paths(deepcopy(template), date)
    cfg = force_universe_fields(cfg)

    cfg.setdefault("output", {})
    cfg["output"]["csv_path"] = f"../PricingModel/data/market_return_20241022_20250114_csi1000_by_date/market_return_{date}_csi1000.csv"
    cfg["output"]["error_log_path"] = f"../PricingModel/data/market_return_20241022_20250114_csi1000_by_date/logs/market_return_{date}_csi1000_errors.txt"

    return cfg


def main():
    ensure_dir(OUT_DIR)
    ensure_dir(CONFIG_DIR)
    ensure_dir(LOG_DIR)

    if not os.path.exists(TEMPLATE_CONFIG):
        raise FileNotFoundError(TEMPLATE_CONFIG)

    if not os.path.exists(f"{PROJECT_ROOT}/PricingModel/data/universe_csi1000.csv"):
        raise FileNotFoundError(f"{PROJECT_ROOT}/PricingModel/data/universe_csi1000.csv")

    with open(TEMPLATE_CONFIG, "r") as f:
        template = yaml.safe_load(f)

    if "output" not in template or "csv_path" not in template["output"]:
        raise RuntimeError(f"Template config is not valid for export_market_return_csv.py: {TEMPLATE_CONFIG}")

    dates = get_date_security_info.get_date_list(START, END)
    print("num dates:", len(dates))
    print("dates:", dates)

    failed = []

    for date in dates:
        date = int(date)
        out_csv = f"{OUT_DIR}/market_return_{date}_csi1000.csv"
        log_path = f"{LOG_DIR}/export_market_return_{date}_csi1000.log"
        cfg_path = f"{CONFIG_DIR}/export_market_return_{date}_csi1000.yaml"

        if os.path.exists(out_csv) and os.path.getsize(out_csv) > 0 and not FORCE:
            print(f"[SKIP] {date} exists: {out_csv}", flush=True)
            continue

        cfg = make_config(template, date)

        with open(cfg_path, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

        cmd = [
            sys.executable,
            "-u",
            "export_market_return_csv.py",
            "--config",
            os.path.relpath(cfg_path, FACTOR_DIR),
        ]

        print("\n" + "=" * 100, flush=True)
        print("[RUN]", date, flush=True)
        print("cfg:", cfg_path, flush=True)
        print("out:", out_csv, flush=True)
        print("log:", log_path, flush=True)
        print("cmd:", " ".join(cmd), flush=True)

        with open(log_path, "w") as logf:
            ret = subprocess.run(
                cmd,
                cwd=FACTOR_DIR,
                stdout=logf,
                stderr=subprocess.STDOUT,
            )

        if ret.returncode != 0:
            print("[FAILED]", date, "returncode:", ret.returncode, flush=True)
            failed.append(date)
            try:
                with open(log_path, "r") as f:
                    lines = f.readlines()
                print("last 80 log lines:")
                print("".join(lines[-80:]))
            except Exception:
                pass
        else:
            print("[DONE]", date, flush=True)

    print("\n" + "=" * 100)
    print("ALL FINISHED")
    print("failed dates:", failed)

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
