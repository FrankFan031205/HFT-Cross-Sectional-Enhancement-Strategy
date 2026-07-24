import os
import argparse

REQUIRED_SNAPSHOT_PRICE_COLS = (
    [f"bidprice{i}" for i in range(1, 11)]
    + [f"askprice{i}" for i in range(1, 11)]
)

def has_required_snapshot_cols(df):
    return all(c in df.columns for c in REQUIRED_SNAPSHOT_PRICE_COLS)

def missing_snapshot_cols(df):
    return [c for c in REQUIRED_SNAPSHOT_PRICE_COLS if c not in df.columns]
import yaml
import numpy as np
import pandas as pd

from utils import data_loader, get_date_security_info
from formula_factor import function_dict as FactorFormulaDict
from dump_factor_features import calc_one_stock


def read_features(path):
    with open(path, "r") as f:
        obj = yaml.safe_load(f)

    if isinstance(obj, dict) and "features" in obj:
        features = list(obj["features"])
    elif isinstance(obj, list):
        features = list(obj)
    else:
        raise RuntimeError(f"unsupported feature yaml format: {path}")

    features = [str(x) for x in features]
    features = [x for x in features if x in FactorFormulaDict]

    seen = set()
    out = []
    for x in features:
        if x not in seen:
            out.append(x)
            seen.add(x)

    if len(out) == 0:
        raise RuntimeError("no valid features found")

    return out


def read_universe(path):
    df = pd.read_csv(path, dtype={"securityid": str})

    if "securityid" not in df.columns:
        raise RuntimeError(f"universe file must contain securityid column: {path}")

    sids = (
        df["securityid"]
        .dropna()
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(6)
        .drop_duplicates()
        .tolist()
    )

    sids = sorted(sids)

    if len(sids) == 0:
        raise RuntimeError("empty universe")

    return sids


def format_datetime(date, timestamp):
    s = str(timestamp)

    if "." in s:
        s = s.split(".")[0]

    s = "".join([c for c in s if c.isdigit()])
    s = s.zfill(9)

    return f"{int(date)}_{s}"


def clean_output(df, date, securityid, factors, horizons):
    if df is None:
        return None

    if hasattr(df, "to_pandas"):
        df = df.to_pandas()

    if len(df) == 0:
        return None

    df = df.copy()

    df["date"] = int(date)

    if "SecurityID" in df.columns:
        df["securityid"] = (
            df["SecurityID"]
            .astype(str)
            .str.replace(r"\.0$", "", regex=True)
            .str.zfill(6)
        )
    elif "securityid" in df.columns:
        df["securityid"] = (
            df["securityid"]
            .astype(str)
            .str.replace(r"\.0$", "", regex=True)
            .str.zfill(6)
        )
    else:
        df["securityid"] = str(securityid).zfill(6)

    if "timestamp" not in df.columns:
        raise RuntimeError("calc_one_stock output has no timestamp column")

    df["datetime"] = df["timestamp"].map(lambda x: format_datetime(date, x))

    if "f_use_check" in df.columns:
        df = df[df["f_use_check"] == 0].copy()

    label_cols = [f"label_{int(h)}" for h in horizons]

    for c in factors:
        if c not in df.columns:
            df[c] = np.nan

    for c in label_cols:
        if c not in df.columns:
            raise RuntimeError(f"missing label column: {c}")

    for c in factors + label_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df[factors + label_cols] = df[factors + label_cols].replace([np.inf, -np.inf], np.nan)

    df = df.dropna(subset=label_cols)

    for c in factors:
        df[c] = df[c].fillna(0.0)

    keep = ["date", "datetime", "securityid"] + factors + label_cols

    return df[keep]


def append_csv(df, output):
    if df is None or len(df) == 0:
        return 0

    os.makedirs(os.path.dirname(output), exist_ok=True)

    header = not os.path.exists(output)
    df.to_csv(output, mode="a", header=header, index=False)

    return len(df)


def load_done_pairs(output):
    if not os.path.exists(output):
        return set()

    done = set()

    usecols = ["date", "securityid"]

    for chunk in pd.read_csv(output, usecols=usecols, dtype={"securityid": str}, chunksize=1000000):
        chunk["securityid"] = chunk["securityid"].astype(str).str.zfill(6)
        pairs = zip(chunk["date"].astype(int), chunk["securityid"])
        done.update(pairs)

    return done


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--universe_file", required=True)
    parser.add_argument("--feature_yaml", required=True)
    parser.add_argument("--horizons", default="30,60,90,120")
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")

    args = parser.parse_args()

    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]
    factors = read_features(args.feature_yaml)
    universe = read_universe(args.universe_file)

    print("=" * 100, flush=True)
    print("dump fixed universe factor features", flush=True)
    print("start:", args.start, flush=True)
    print("end:", args.end, flush=True)
    print("universe_file:", args.universe_file, flush=True)
    print("num universe:", len(universe), flush=True)
    print("feature_yaml:", args.feature_yaml, flush=True)
    print("num factors:", len(factors), flush=True)
    print("horizons:", horizons, flush=True)
    print("output:", args.output, flush=True)
    print("overwrite:", args.overwrite, flush=True)
    print("resume:", args.resume, flush=True)
    print("=" * 100, flush=True)

    if args.overwrite and os.path.exists(args.output):
        os.remove(args.output)
        print("removed old output:", args.output, flush=True)

    if os.path.exists(args.output) and not args.resume:
        raise RuntimeError(f"output exists: {args.output}. Use --overwrite or --resume.")

    done_pairs = load_done_pairs(args.output) if args.resume else set()

    if done_pairs:
        print("loaded done pairs:", len(done_pairs), flush=True)

    dates = get_date_security_info.get_date_list(args.start, args.end)

    total = 0
    errors = []

    for date in dates:
        print("\n" + "=" * 100, flush=True)
        print("loading date:", date, flush=True)

        data_loader.init_clickhouse_client(date)

        for i, sid in enumerate(universe):
            pair = (int(date), str(sid).zfill(6))

            if pair in done_pairs:
                print(date, i + 1, "/", len(universe), sid, "skip done", flush=True)
                continue

            try:
                print(date, i + 1, "/", len(universe), sid, flush=True)

                df_one = calc_one_stock(
                    str(date),
                    str(sid),
                    factors,
                    horizons,
                )

                out = clean_output(
                    df_one,
                    date=date,
                    securityid=sid,
                    factors=factors,
                    horizons=horizons,
                )

                n = append_csv(out, args.output)
                total += n

                print("  appended rows:", n, "total new:", total, flush=True)

            except Exception as e:
                print("stock error:", date, sid, repr(e), flush=True)
                errors.append({
                    "date": date,
                    "securityid": sid,
                    "error": repr(e),
                })

    print("\n" + "=" * 100, flush=True)
    print("DONE", flush=True)
    print("saved:", args.output, flush=True)
    print("new rows appended:", total, flush=True)

    if errors:
        err_path = args.output.replace(".csv", "_errors.csv")
        pd.DataFrame(errors).to_csv(err_path, index=False)
        print("errors saved:", err_path, flush=True)
        print("num errors:", len(errors), flush=True)


if __name__ == "__main__":
    main()
