# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def pick(df, names, required=True, label="col"):
    for c in names:
        if c in df.columns:
            return c
    low = {str(c).lower(): c for c in df.columns}
    for c in names:
        if str(c).lower() in low:
            return low[str(c).lower()]
    if required:
        raise KeyError(f"cannot find {label}: {names}; cols={list(df.columns)}")
    return None


def read_sig(path, h):
    df = pd.read_parquet(path)
    sid = pick(df, ["securityid", "SecurityID", "sid", "symbol"], label=f"{h} sid")
    date = pick(df, ["date", "execution_date"], label=f"{h} date")
    dt = pick(df, ["datetime", "execution_datetime", "tsminute", "timestamp"], label=f"{h} datetime")
    if h == "h10":
        sig = pick(df, ["pred_ret_h10", "pred_z_h10", "pred_z", "signal", "score", "pred_ret_h20"], label=f"{h} signal")
    elif h == "h20":
        sig = pick(df, ["pred_ret_h20", "pred_z_h20", "pred_z", "signal", "score"], label=f"{h} signal")
    else:
        sig = pick(df, ["pred_ret_h30", "pred_z_h30", "pred_z", "signal", "score", "pred_ret_h20"], label=f"{h} signal")
    return pd.DataFrame({
        "date": df[date].astype(int),
        "datetime": pd.to_datetime(df[dt]),
        "securityid": df[sid].astype(str).str.extract(r"(\d+)")[0].str.zfill(6),
        f"sig_{h}": pd.to_numeric(df[sig], errors="coerce").fillna(0.0),
    }), sig


def zscore(g, col):
    x = pd.to_numeric(g[col], errors="coerce").fillna(0.0)
    mu = x.mean()
    sd = x.std()
    if not np.isfinite(sd) or sd <= 1e-12:
        return pd.Series(0.0, index=g.index)
    return (x - mu) / sd


def parse_mix(s):
    ans = []
    for part in s.split(";"):
        part = part.strip()
        if not part:
            continue
        name, w = part.split(":")
        ws = [float(x) for x in w.split(",")]
        assert len(ws) == 3
        ans.append((name.strip(), ws[0], ws[1], ws[2]))
    return ans


def patch_benchmark(base, weight_path):
    w = pd.read_parquet(weight_path)
    wsid = pick(w, ["securityid", "SecurityID", "sid", "symbol"], label="weight sid")
    ww = pick(w, ["benchmark_weight", "weight"], label="weight")
    w = pd.DataFrame({
        "_sid": w[wsid].astype(str).str.extract(r"(\d+)")[0].str.zfill(6),
        "_w": pd.to_numeric(w[ww], errors="coerce").fillna(0.0),
    }).drop_duplicates("_sid", keep="last")

    sid = pick(base, ["securityid", "SecurityID", "sid", "symbol"], label="base sid")
    date = pick(base, ["date", "execution_date"], label="base date")
    dt = pick(base, ["datetime", "execution_datetime", "tsminute", "timestamp"], label="base datetime")

    base = base.copy()
    base[sid] = base[sid].astype(str).str.extract(r"(\d+)")[0].str.zfill(6)

    for c in ["benchmark_weight", "bench_weight", "index_weight", "ew_benchmark_weight"]:
        if c in base.columns:
            base = base.drop(columns=[c])

    base = base.merge(w, left_on=sid, right_on="_sid", how="left")
    base["_w"] = base["_w"].fillna(0.0)

    s = base.groupby([date, dt])["_w"].transform("sum")
    n = base.groupby([date, dt])[sid].transform("count")
    base["benchmark_weight"] = np.where(s > 1e-12, base["_w"] / s, 1.0 / n)

    stats = {
        "matched_names": int(base.loc[base["_w"] > 0, sid].nunique()),
        "total_names": int(base[sid].nunique()),
        "bench_min": float(base.groupby([date, dt])["benchmark_weight"].sum().min()),
        "bench_max": float(base.groupby([date, dt])["benchmark_weight"].sum().max()),
    }
    base = base.drop(columns=["_sid", "_w"], errors="ignore")
    return base, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h10-dir", required=True)
    ap.add_argument("--h20-dir", required=True)
    ap.add_argument("--h30-dir", required=True)
    ap.add_argument("--weight-file", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--mix-specs", default="h20:0,1,0;mix_020602:0.2,0.6,0.2;mix_030502:0.3,0.5,0.2;mix_000703:0,0.7,0.3;mix_020404:0.2,0.4,0.4")
    args = ap.parse_args()

    h10 = Path(args.h10_dir)
    h20 = Path(args.h20_dir)
    h30 = Path(args.h30_dir)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    mixes = parse_mix(args.mix_specs)
    out_dirs = {}
    for name, *_ in mixes:
        d = out_root / f"zzy_pure_cs_mix_{name}_csi2000_weight_input"
        d.mkdir(parents=True, exist_ok=True)
        out_dirs[name] = d

    h20_files = sorted(h20.glob("*.parquet"))
    if not h20_files:
        raise FileNotFoundError(h20)

    h10_map = {p.name: p for p in h10.glob("*.parquet")} if h10.exists() else {}
    h30_map = {p.name: p for p in h30.glob("*.parquet")} if h30.exists() else {}

    print("[h10 files]", len(h10_map), h10)
    print("[h20 files]", len(h20_files), h20)
    print("[h30 files]", len(h30_map), h30)

    rows = []
    need_h10 = any(abs(w10) > 1e-12 for _, w10, _, _ in mixes)
    need_h30 = any(abs(w30) > 1e-12 for _, _, _, w30 in mixes)

    for k, f20 in enumerate(h20_files, 1):
        name = f20.name
        print(f"\n[{k}/{len(h20_files)}] {name}")
        f10 = h10_map.get(name)
        f30 = h30_map.get(name)

        if need_h10 and f10 is None:
            raise FileNotFoundError(f"missing h10 {name}")
        if need_h30 and f30 is None:
            raise FileNotFoundError(f"missing h30 {name}")

        base = pd.read_parquet(f20)
        base, bench_stats = patch_benchmark(base, args.weight_file)

        sid = pick(base, ["securityid", "SecurityID", "sid", "symbol"], label="base sid")
        date = pick(base, ["date", "execution_date"], label="base date")
        dt = pick(base, ["datetime", "execution_datetime", "tsminute", "timestamp"], label="base datetime")
        sig20 = pick(base, ["pred_ret_h20", "pred_z", "signal", "score"], label="h20 signal")

        base[sid] = base[sid].astype(str).str.extract(r"(\d+)")[0].str.zfill(6)
        base[dt] = pd.to_datetime(base[dt])
        base["sig_h20"] = pd.to_numeric(base[sig20], errors="coerce").fillna(0.0)

        key = [date, dt, sid]

        if f10 is not None:
            d10, sig10 = read_sig(f10, "h10")
            d10 = d10.rename(columns={"date": date, "datetime": dt, "securityid": sid})
            base = base.merge(d10[key + ["sig_h10"]], on=key, how="left")
        else:
            sig10 = ""
            base["sig_h10"] = 0.0

        if f30 is not None:
            d30, sig30 = read_sig(f30, "h30")
            d30 = d30.rename(columns={"date": date, "datetime": dt, "securityid": sid})
            base = base.merge(d30[key + ["sig_h30"]], on=key, how="left")
        else:
            sig30 = ""
            base["sig_h30"] = 0.0

        base[["sig_h10", "sig_h30"]] = base[["sig_h10", "sig_h30"]].fillna(0.0)
        base["z_h10"] = base.groupby([date, dt], group_keys=False).apply(lambda g: zscore(g, "sig_h10"))
        base["z_h20"] = base.groupby([date, dt], group_keys=False).apply(lambda g: zscore(g, "sig_h20"))
        base["z_h30"] = base.groupby([date, dt], group_keys=False).apply(lambda g: zscore(g, "sig_h30"))

        for mix, w10, w20, w30 in mixes:
            out = base.copy()
            out["pred_ret_mix"] = w10 * out["z_h10"] + w20 * out["z_h20"] + w30 * out["z_h30"]
            out["pred_ret_h20"] = out["pred_ret_mix"]  # compatible with existing optimizer
            path = out_dirs[mix] / name
            out.to_parquet(path, index=False)
            print("[write]", mix, path.name, out.shape, bench_stats)
            rows.append({
                "file": name, "mix": mix, "w10": w10, "w20": w20, "w30": w30,
                "rows": len(out), "names": out[sid].nunique(),
                "sig10": sig10, "sig20": sig20, "sig30": sig30,
                **bench_stats,
            })

    summ = pd.DataFrame(rows)
    sp = out_root / "_prepare_summary.csv"
    summ.to_csv(sp, index=False)
    print("\n===== DONE =====")
    print(summ.groupby("mix")[["rows", "names", "matched_names", "total_names"]].mean().to_string())
    print("[summary]", sp)
    for m, d in out_dirs.items():
        print(m, d)


if __name__ == "__main__":
    main()
