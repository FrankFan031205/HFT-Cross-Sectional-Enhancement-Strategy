import argparse
import glob
import os
import subprocess
from pathlib import Path

import pandas as pd
import yaml


DEFAULT_REGISTRY = "config/experiments/factor_registry.yaml"

DEFAULT_REGISTRY_CONTENT = {
    "defaults": {
        "base_config_path": "config/backtest.yaml",
        "fill_model": {
            "mode": "queue_aware_trade",
            "queue_ahead_multiplier": 0.05,
        },
        "policy": {
            "name": "abs_top40",
            "type": "abs_signal_quantile",
            "quantile": 0.60,
        },
        "inventory": {
            "initial_position_per_symbol": 5000,
            "max_position_per_symbol": 15000,
            "sell_floor_position": 3000,
            "buy_block_position": 9000,
            "tplus1": True,
            "allow_short": False,
        },
        "fee": {
            "mode": "single_fill_side_specific",
            "commission_rate": 0.00005,
            "transfer_fee_rate": 0.00001,
            "handling_fee_rate": 0.0000341,
            "regulatory_fee_rate": 0.00002,
            "stamp_duty_rate": 0.0005,
        },
    },
    "experiments": [],
}


def run(cmd, shell=False):
    if shell:
        print("\n$ " + cmd)
        subprocess.run(cmd, shell=True, check=True)
    else:
        print("\n$ " + " ".join(cmd))
        subprocess.run(cmd, check=True)


def norm_sid(s):
    return s.astype(str).str.replace(".0", "", regex=False).str.zfill(6)


def parse_dt(s):
    x = s.astype(str)

    # supports format like 20241022_093000000
    m = x.str.extract(r"^(\d{8})_(\d{9})$")
    if m.notna().all(axis=1).mean() > 0.8:
        z = m[0] + m[1]
        return pd.to_datetime(z, format="%Y%m%d%H%M%S%f", errors="coerce")

    return pd.to_datetime(s, errors="coerce")


def dt_key(s):
    return parse_dt(s).dt.floor("ms").dt.strftime("%Y-%m-%d %H:%M:%S.%f").str[:-3]


def find_col(cols, candidates, name, required=True):
    for c in candidates:
        if c in cols:
            return c
    if required:
        raise RuntimeError(f"cannot find {name}; tried {candidates}")
    return None


def infer_signal_col(cols, signal_col):
    if signal_col:
        if signal_col not in cols:
            cands = [
                c for c in cols
                if "hidden" in c.lower()
                or "factor" in c.lower()
                or "pred" in c.lower()
                or "alpha" in c.lower()
            ]
            raise RuntimeError(f"signal_col not found: {signal_col}; candidates={cands}")
        return signal_col

    cands = [
        c for c in cols
        if "hidden" in c.lower()
        or "factor" in c.lower()
        or "pred" in c.lower()
        or "alpha" in c.lower()
    ]

    if len(cands) == 1:
        print("auto detected signal_col:", cands[0])
        return cands[0]

    raise RuntimeError(f"cannot auto-detect signal_col uniquely; candidates={cands}")


def load_registry(path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    if not p.exists():
        with open(p, "w") as f:
            yaml.safe_dump(DEFAULT_REGISTRY_CONTENT, f, sort_keys=False, allow_unicode=True)

    with open(p, "r") as f:
        data = yaml.safe_load(f) or {}

    data.setdefault("defaults", DEFAULT_REGISTRY_CONTENT["defaults"])
    data.setdefault("experiments", [])
    return data


def save_registry(path, data):
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def update_registry(registry_path, tag, quote_path, signal_col, overwrite):
    data = load_registry(registry_path)
    exps = data["experiments"]

    item = {
        "tag": tag,
        "quote_decision_path": quote_path,
        "signal_col": signal_col,
    }

    idx = None
    for i, x in enumerate(exps):
        if x.get("tag") == tag:
            idx = i
            break

    if idx is None:
        exps.append(item)
        print("[registry] added:", tag)
    else:
        if not overwrite:
            raise RuntimeError(f"tag already exists: {tag}; use --overwrite")
        exps[idx] = item
        print("[registry] updated:", tag)

    save_registry(registry_path, data)
    print("[registry] saved:", registry_path)


def scan_hidden(hidden_csv, dt_col, sec_col, signal_col, chunksize, sample_limit):
    print("\n[1] scanning hidden factor file")

    dates = set()
    securities = set()
    sig_parts = []
    nrows = 0
    nsample = 0

    usecols = [dt_col, sec_col, signal_col]

    for i, chunk in enumerate(pd.read_csv(hidden_csv, usecols=usecols, chunksize=chunksize, low_memory=False)):
        nrows += len(chunk)

        dt = parse_dt(chunk[dt_col])
        dates.update(dt.dt.strftime("%Y-%m-%d").dropna().unique().tolist())

        sid = norm_sid(chunk[sec_col])
        securities.update(sid.dropna().unique().tolist())

        sig = pd.to_numeric(chunk[signal_col], errors="coerce").dropna()

        if nsample < sample_limit:
            need = sample_limit - nsample
            if len(sig) > need:
                sig = sig.sample(need, random_state=1)
            sig_parts.append(sig)
            nsample += len(sig)

        if i % 20 == 0:
            print(f"[hidden] chunk={i}, rows={nrows}, sampled={nsample}")

    if not sig_parts:
        raise RuntimeError("no valid signal values found")

    sig_sample = pd.concat(sig_parts, ignore_index=True)

    stats = {
        "dates": dates,
        "securities": securities,
        "q30": sig_sample.quantile(0.30),
        "q70": sig_sample.quantile(0.70),
        "abs_q50": sig_sample.abs().quantile(0.50),
        "abs_q90": sig_sample.abs().quantile(0.90),
    }

    print("\n===== hidden stats =====")
    print("rows:", nrows)
    print("num dates:", len(dates))
    print("date range:", min(dates), "->", max(dates))
    print("num securities:", len(securities))
    print("signal_col:", signal_col)
    print("q30:", stats["q30"])
    print("q70:", stats["q70"])
    print("abs_q50:", stats["abs_q50"])
    print("abs_q90:", stats["abs_q90"])

    return stats


def load_market(pricing_csv, dates, securities, chunksize, price_scale):
    print("\n[2] loading market state from pricing csv")

    cols = pd.read_csv(pricing_csv, nrows=0).columns.tolist()

    dt_col = find_col(cols, ["datetime", "time", "timestamp"], "pricing datetime")
    sec_col = find_col(cols, ["securityid", "SecurityID", "symbol"], "pricing securityid")

    bid_col = find_col(
        cols,
        ["bid_price", "bid_quote_price", "bid1", "bidprice1", "bidprice1", "bidpx1"],
        "best bid price",
    )
    ask_col = find_col(
        cols,
        ["ask_price", "ask_quote_price", "ask1", "askprice1", "askprice1", "askpx1"],
        "best ask price",
    )
    mid_col = find_col(cols, ["mid_price", "midprice", "mid"], "mid price", required=False)

    usecols = [dt_col, sec_col, bid_col, ask_col]
    if mid_col:
        usecols.append(mid_col)

    parts = []
    scanned = 0
    kept = 0

    for i, chunk in enumerate(pd.read_csv(pricing_csv, usecols=usecols, chunksize=chunksize, low_memory=False)):
        scanned += len(chunk)

        dt = parse_dt(chunk[dt_col])
        chunk["_date"] = dt.dt.strftime("%Y-%m-%d")
        chunk["_dt_key"] = dt.dt.floor("ms").dt.strftime("%Y-%m-%d %H:%M:%S.%f").str[:-3]
        chunk["securityid"] = norm_sid(chunk[sec_col])

        mask = chunk["_date"].isin(dates) & chunk["securityid"].isin(securities)
        sub = chunk.loc[mask].copy()

        if len(sub):
            sub["bid_price"] = pd.to_numeric(sub[bid_col], errors="coerce")
            sub["ask_price"] = pd.to_numeric(sub[ask_col], errors="coerce")

            if mid_col:
                sub["mid_price"] = pd.to_numeric(sub[mid_col], errors="coerce")
            else:
                sub["mid_price"] = (sub["bid_price"] + sub["ask_price"]) / 2.0

            sub = sub[["_dt_key", "securityid", "bid_price", "ask_price", "mid_price"]]
            parts.append(sub)
            kept += len(sub)

        if i % 20 == 0:
            print(f"[pricing] chunk={i}, scanned={scanned}, kept={kept}")

    if not parts:
        raise RuntimeError("no matched market rows found")

    market = pd.concat(parts, ignore_index=True).drop_duplicates(["_dt_key", "securityid"])
    market = market.dropna(subset=["bid_price", "ask_price", "mid_price"])

    if price_scale == "auto":
        med = market[["bid_price", "ask_price"]].stack().median()
        scale = 100.0 if med > 300 else 1.0
    else:
        scale = float(price_scale)

    if scale != 1.0:
        for c in ["bid_price", "ask_price", "mid_price"]:
            market[c] = market[c] / scale

    print("\n===== market state =====")
    print("rows:", len(market))
    print("num securities:", market["securityid"].nunique())
    print("price scale:", scale)
    print(market[["bid_price", "ask_price", "mid_price"]].describe())

    return market


def generate_quote(hidden_csv, quote_out, market, dt_col, sec_col, signal_col, stats, chunksize, quote_size):
    print("\n[3] generating quote decision")
    print("quote_out:", quote_out)

    Path(quote_out).parent.mkdir(parents=True, exist_ok=True)

    market = market.set_index(["_dt_key", "securityid"])

    first = True
    total = 0
    written = 0
    bid_cnt = 0
    ask_cnt = 0
    missing_market = 0

    usecols = [dt_col, sec_col, signal_col]

    for i, chunk in enumerate(pd.read_csv(hidden_csv, usecols=usecols, chunksize=chunksize, low_memory=False)):
        total += len(chunk)

        chunk["_dt_key"] = dt_key(chunk[dt_col])
        chunk["securityid"] = norm_sid(chunk[sec_col])
        chunk[signal_col] = pd.to_numeric(chunk[signal_col], errors="coerce")

        joined = chunk.join(market, on=["_dt_key", "securityid"], how="left")
        missing_market += joined["bid_price"].isna().sum()

        joined = joined.dropna(subset=[signal_col, "bid_price", "ask_price", "mid_price"]).copy()

        sig = joined[signal_col]
        joined["quote_bid"] = sig >= stats["q70"]
        joined["quote_ask"] = sig <= stats["q30"]

        joined["bid_quote_price"] = joined["bid_price"]
        joined["ask_quote_price"] = joined["ask_price"]
        joined["bid1"] = joined["bid_price"]
        joined["ask1"] = joined["ask_price"]

        joined["bid_size"] = quote_size
        joined["ask_size"] = quote_size

        joined["raw_pred"] = sig
        joined["pred_used"] = sig
        joined["fair_price"] = joined["mid_price"] * (1.0 + sig)
        joined["bid_edge"] = joined["fair_price"] - joined["bid_quote_price"]
        joined["ask_edge"] = joined["ask_quote_price"] - joined["fair_price"]

        abs_sig = sig.abs()
        joined["risk_state"] = "normal"
        joined.loc[abs_sig < stats["abs_q50"], "risk_state"] = "weak_alpha"
        joined.loc[abs_sig >= stats["abs_q90"], "risk_state"] = "strong_alpha"

        out = joined[
            [
                dt_col,
                "securityid",
                signal_col,
                "raw_pred",
                "pred_used",
                "quote_bid",
                "quote_ask",
                "bid1",
                "ask1",
                "bid_price",
                "ask_price",
                "bid_quote_price",
                "ask_quote_price",
                "bid_size",
                "ask_size",
                "mid_price",
                "fair_price",
                "bid_edge",
                "ask_edge",
                "risk_state",
            ]
        ].copy()

        out = out.rename(columns={dt_col: "datetime"})

        out.to_csv(quote_out, index=False, mode="w" if first else "a", header=first)
        first = False

        written += len(out)
        bid_cnt += int(out["quote_bid"].sum())
        ask_cnt += int(out["quote_ask"].sum())

        if i % 20 == 0:
            print(
                f"[quote] chunk={i}, hidden_rows={total}, written={written}, "
                f"missing_market={missing_market}, bid={bid_cnt}, ask={ask_cnt}"
            )

    print("\n===== quote generated =====")
    print("hidden rows:", total)
    print("quote rows:", written)
    print("missing market rows:", missing_market)
    print("quote_bid:", bid_cnt)
    print("quote_ask:", ask_cnt)
    print("saved:", quote_out)

    if written == 0:
        raise RuntimeError("quote output is empty")
    if bid_cnt == 0 and ask_cnt == 0:
        raise RuntimeError("quote_bid and quote_ask are both zero")


def cleanup_outputs(tag):
    print("\n[cleanup] removing sample outputs before full run")
    for pat in [
        f"outputs/fills/*{tag}*.csv",
        f"outputs/trades/*{tag}*.csv",
        f"outputs/metrics/*{tag}*.csv",
        f"outputs/portfolio/*{tag}*.csv",
        f"outputs/cache/*{tag}*.csv",
    ]:
        for p in glob.glob(pat):
            try:
                os.remove(p)
                print("removed:", p)
            except FileNotFoundError:
                pass


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--hidden-csv", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--signal-col", default="")
    parser.add_argument("--pricing-csv", default="../PricingModel/data/pricing_dataset_h60_202410_100_full.csv")
    parser.add_argument("--quote-out", default="")

    parser.add_argument("--hidden-datetime-col", default="datetime")
    parser.add_argument("--hidden-security-col", default="securityid")

    parser.add_argument("--chunksize", type=int, default=500000)
    parser.add_argument("--threshold-sample-rows", type=int, default=2000000)
    parser.add_argument("--quote-size", type=int, default=100)
    parser.add_argument("--price-scale", default="auto")

    parser.add_argument("--registry", default=DEFAULT_REGISTRY)
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--build", action="store_true")
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--max-quotes", type=int, default=5000)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--skip-quote-generation", action="store_true")

    args = parser.parse_args()

    Path("logs").mkdir(exist_ok=True)
    Path("config/experiments").mkdir(parents=True, exist_ok=True)

    if not Path(args.hidden_csv).exists():
        raise FileNotFoundError(args.hidden_csv)
    if not Path(args.pricing_csv).exists():
        raise FileNotFoundError(args.pricing_csv)

    hidden_cols = pd.read_csv(args.hidden_csv, nrows=0).columns.tolist()

    if args.hidden_datetime_col not in hidden_cols:
        raise RuntimeError(f"hidden datetime column not found: {args.hidden_datetime_col}")
    if args.hidden_security_col not in hidden_cols:
        raise RuntimeError(f"hidden security column not found: {args.hidden_security_col}")

    signal_col = infer_signal_col(hidden_cols, args.signal_col)

    quote_out = args.quote_out
    if not quote_out:
        quote_out = f"../MarketMakingModel/outputs/quote_decisions/quote_decisions_{args.tag}.csv"

    print("===== hidden csv to backtest =====")
    print("hidden_csv:", args.hidden_csv)
    print("tag:", args.tag)
    print("signal_col:", signal_col)
    print("pricing_csv:", args.pricing_csv)
    print("quote_out:", quote_out)

    if not args.skip_quote_generation:
        stats = scan_hidden(
            args.hidden_csv,
            args.hidden_datetime_col,
            args.hidden_security_col,
            signal_col,
            args.chunksize,
            args.threshold_sample_rows,
        )

        market = load_market(
            args.pricing_csv,
            stats["dates"],
            stats["securities"],
            args.chunksize,
            args.price_scale,
        )

        generate_quote(
            args.hidden_csv,
            quote_out,
            market,
            args.hidden_datetime_col,
            args.hidden_security_col,
            signal_col,
            stats,
            args.chunksize,
            args.quote_size,
        )
    else:
        print("[skip] quote generation")
        if not Path(quote_out).exists():
            raise FileNotFoundError(quote_out)

    update_registry(args.registry, args.tag, quote_out, signal_col, args.overwrite)

    if args.build or args.sample or args.full:
        run(["python", "scripts/build_experiment_configs.py", "--registry", args.registry])

    exp_path = f"config/experiments/generated/{args.tag}.yaml"

    if args.sample:
        run([
            "python",
            "scripts/run_backtest_experiment.py",
            "--exp",
            exp_path,
            "--max-quotes",
            str(args.max_quotes),
        ])

    if args.full:
        if args.sample:
            cleanup_outputs(args.tag)

        log_path = f"logs/{args.tag}_backtest_experiment.log"
        cmd = (
            f"nohup python scripts/run_backtest_experiment.py "
            f"--exp {exp_path} "
            f"> {log_path} 2>&1 &"
        )
        run(cmd, shell=True)
        print("full backtest started in background")
        print("log:", log_path)
        print(f"tail -f {log_path}")

    print("\n===== done =====")
    print("hidden_csv:", args.hidden_csv)
    print("quote_decision:", quote_out)
    print("tag:", args.tag)
    print("signal_col:", signal_col)
    print("generated_config:", exp_path)


if __name__ == "__main__":
    main()
