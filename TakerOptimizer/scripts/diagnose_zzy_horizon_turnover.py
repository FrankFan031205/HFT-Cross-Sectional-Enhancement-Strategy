# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl


def parse_cases(s: str):
    """
    Format:
      3:3,5:5,10:5,10:10,20:10,20:20,30:15
    means:
      horizon_min : rebalance_min
    """
    out = []
    for x in s.split(","):
        x = x.strip()
        if not x:
            continue
        h, rb = x.split(":")
        out.append((int(h), int(rb)))
    return out


def safe_corr(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 5:
        return np.nan
    aa = a[m]
    bb = b[m]
    if aa.std() <= 1e-12 or bb.std() <= 1e-12:
        return np.nan
    return float(np.corrcoef(aa, bb)[0, 1])


def compute_ic_metrics(df: pd.DataFrame):
    keys = ["date", "ts"]

    ic_rows = []
    for _, g in df.groupby(keys, sort=False):
        ic_rows.append((
            safe_corr(g["score"].values, g["y"].values),
            safe_corr(g["rank_score_asc"].values, g["rank_y_asc"].values),
        ))

    ic = pd.DataFrame(ic_rows, columns=["ic", "rank_ic"])

    return {
        "n_snapshots": len(ic),
        "ic_mean": ic["ic"].mean(),
        "ic_std": ic["ic"].std(),
        "rank_ic_mean": ic["rank_ic"].mean(),
        "rank_ic_std": ic["rank_ic"].std(),
    }


def compute_decile_metrics(df: pd.DataFrame):
    keys = ["date", "ts"]

    top = df[df["rank_desc"] <= df["decile_n"]]
    bot = df[df["rank_desc"] > (df["n_universe"] - df["decile_n"])]

    top_by_t = top.groupby(keys)["y"].mean()
    bot_by_t = bot.groupby(keys)["y"].mean()

    both = pd.concat([top_by_t.rename("top"), bot_by_t.rename("bottom")], axis=1).dropna()

    if both.empty:
        return {
            "top_decile_bps": np.nan,
            "bottom_decile_bps": np.nan,
            "top_bottom_bps": np.nan,
        }

    return {
        "top_decile_bps": both["top"].mean() * 1e4,
        "bottom_decile_bps": both["bottom"].mean() * 1e4,
        "top_bottom_bps": (both["top"] - both["bottom"]).mean() * 1e4,
    }


def simulate_turnover(rank_sub: pd.DataFrame, k: int, exit_rank: int):
    """
    rank_sub includes rows with rank_desc <= max_exit_rank.
    Equal-weight topK / stateful topK turnover proxy.

    turnover unit:
      target gross = 1.0
      full turnover = sum(abs(w_t - w_{t-1}))
    Therefore if target gross = 10%, multiply by 0.10.
    """
    full_turnovers = []
    stateful_turnovers = []
    next_topk_retentions = []
    next_exit_retentions = []
    daily_full = []
    daily_stateful = []
    stateful_hold_counts = []

    for d, gd in rank_sub.groupby("date", sort=True):
        infos = []
        for ts, gt in gd.groupby("ts", sort=True):
            gt = gt.sort_values("rank_desc")
            topk = list(gt.loc[gt["rank_desc"] <= k, "sid"].astype(str))
            topk_set = set(topk)
            exit_set = set(gt.loc[gt["rank_desc"] <= exit_rank, "sid"].astype(str))
            sorted_by_rank = list(gt["sid"].astype(str))
            infos.append((ts, topk, topk_set, exit_set, sorted_by_rank))

        if not infos:
            continue

        # Full topK rebalance turnover.
        prev_top = set()
        day_full = 0.0
        for j, (_, topk, topk_set, _, _) in enumerate(infos):
            denom = max(k, 1)
            if j == 0:
                turn = len(topk_set) / denom
            else:
                overlap = len(prev_top & topk_set)
                turn = (len(prev_top - topk_set) + len(topk_set - prev_top)) / denom
                next_topk_retentions.append(overlap / max(len(prev_top), 1))
            full_turnovers.append(turn)
            day_full += turn
            prev_top = topk_set
        daily_full.append(day_full)

        # Retention into next exit band.
        for j in range(len(infos) - 1):
            prev_top = infos[j][2]
            next_exit = infos[j + 1][3]
            if prev_top:
                next_exit_retentions.append(len(prev_top & next_exit) / len(prev_top))

        # Stateful entry/exit band.
        held = set()
        day_state = 0.0
        for j, (_, topk, topk_set, exit_set, sorted_by_rank) in enumerate(infos):
            keep = held & exit_set

            new_held = list(keep)
            for sid in topk:
                if sid not in keep:
                    new_held.append(sid)
                if len(new_held) >= k:
                    break

            # If topK not enough due invalid names, fill from rank list.
            if len(new_held) < k:
                for sid in sorted_by_rank:
                    if sid not in new_held:
                        new_held.append(sid)
                    if len(new_held) >= k:
                        break

            new_held = set(new_held[:k])

            denom = max(k, 1)
            if j == 0:
                turn = len(new_held) / denom
            else:
                turn = (len(held - new_held) + len(new_held - held)) / denom

            stateful_turnovers.append(turn)
            stateful_hold_counts.append(len(new_held))
            day_state += turn
            held = new_held

        daily_stateful.append(day_state)

    def mean_or_nan(x):
        return float(np.nanmean(x)) if len(x) else np.nan

    return {
        "k": k,
        "exit_rank": exit_rank,

        "topk_retention_next_topk": mean_or_nan(next_topk_retentions),
        "topk_retention_next_exit": mean_or_nan(next_exit_retentions),

        "full_topk_turnover_per_reb": mean_or_nan(full_turnovers),
        "full_topk_turnover_per_day": mean_or_nan(daily_full),

        "stateful_turnover_per_reb": mean_or_nan(stateful_turnovers),
        "stateful_turnover_per_day": mean_or_nan(daily_stateful),
        "stateful_avg_n_hold": mean_or_nan(stateful_hold_counts),
    }


def load_case(pred_root: Path, horizon: int, model: str, rebalance_min: int, score_col: str):
    if model == "res":
        fp = pred_root / "pred" / f"{horizon}min" / "test_predictions.parquet"
    elif model == "ts":
        fp = pred_root / "pred_ts" / f"{horizon}min" / "test_predictions.parquet"
    else:
        raise ValueError("model must be res or ts")

    if not fp.exists():
        raise FileNotFoundError(str(fp))

    rb_sec = int(rebalance_min * 60)

    schema = pl.scan_parquet(str(fp)).schema
    if score_col in schema:
        sc = score_col
    elif "pred_z" in schema:
        sc = "pred_z"
    else:
        sc = "pred"

    lf = (
        pl.scan_parquet(str(fp))
        .filter(pl.col("ts").cast(pl.Int64) % rb_sec == 0)
        .select([
            pl.col("date").cast(pl.Int64),
            pl.col("sid").cast(pl.Utf8),
            pl.col("ts").cast(pl.Int64),
            pl.col(sc).cast(pl.Float64).alias("score"),
            pl.col("y_raw").cast(pl.Float64).alias("y"),
        ])
        .filter(
            pl.col("score").is_not_null()
            & pl.col("y").is_not_null()
            & pl.col("score").is_finite()
            & pl.col("y").is_finite()
        )
    )

    df = lf.collect().to_pandas()
    if df.empty:
        raise RuntimeError(f"empty case h{horizon} rb{rebalance_min}")

    return df


def enrich_ranks(df: pd.DataFrame):
    keys = ["date", "ts"]

    df = df.copy()
    df["n_universe"] = df.groupby(keys)["sid"].transform("size")
    df["decile_n"] = np.ceil(df["n_universe"] * 0.10).astype(int).clip(lower=1)

    # rank_desc: 1 = best signal
    df["rank_desc"] = df.groupby(keys)["score"].rank(method="first", ascending=False)
    df["rank_score_asc"] = df.groupby(keys)["score"].rank(method="average", ascending=True)
    df["rank_y_asc"] = df.groupby(keys)["y"].rank(method="average", ascending=True)

    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/mnt/data1/zzy/optimizer_data")
    ap.add_argument("--model", default="res", choices=["res", "ts"])
    ap.add_argument("--score-col", default="pred_z")
    ap.add_argument("--cases", default="3:3,5:5,10:5,10:10,20:10,20:20,30:15")
    ap.add_argument("--top-ks", default="50,100")
    ap.add_argument("--exit-ranks", default="150,300,500")
    ap.add_argument("--fee-bps", type=float, default=0.5)
    ap.add_argument("--gross-for-fee-proxy", type=float, default=0.10)
    ap.add_argument("--out", default="/mnt/data1/fwz/HFT_010-dev_fwz_outputs/HorizonDiagnostics/zzy_horizon_turnover_diagnostics.csv")
    args = ap.parse_args()

    pred_root = Path(args.data_root)
    cases = parse_cases(args.cases)
    top_ks = [int(x) for x in args.top_ks.split(",") if x.strip()]
    exit_ranks = [int(x) for x in args.exit_ranks.split(",") if x.strip()]
    max_exit = max(exit_ranks)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    for horizon, rb_min in cases:
        print(f"\n===== horizon={horizon}min model={args.model} rebalance={rb_min}min =====")

        try:
            df = load_case(
                pred_root=pred_root,
                horizon=horizon,
                model=args.model,
                rebalance_min=rb_min,
                score_col=args.score_col,
            )
        except Exception as e:
            print("[SKIP]", horizon, rb_min, e)
            continue

        print("[loaded]", df.shape)

        df = enrich_ranks(df)

        base = {
            "model": args.model,
            "horizon_min": horizon,
            "rebalance_min": rb_min,
            "rows": len(df),
            "dates": df["date"].nunique(),
            "snapshots": df[["date", "ts"]].drop_duplicates().shape[0],
            "avg_universe": df.groupby(["date", "ts"])["sid"].size().mean(),
        }

        base.update(compute_ic_metrics(df))
        base.update(compute_decile_metrics(df))

        # topK forward return metrics
        for k in top_ks:
            sel = df[df["rank_desc"] <= k]
            if len(sel):
                topk_by_t = sel.groupby(["date", "ts"])["y"].mean()
                base[f"top{k}_target_bps"] = topk_by_t.mean() * 1e4
                base[f"top{k}_snapshots"] = len(topk_by_t)
            else:
                base[f"top{k}_target_bps"] = np.nan
                base[f"top{k}_snapshots"] = 0

        # retention / turnover only needs top max_exit names
        rank_sub = df[df["rank_desc"] <= max_exit][["date", "ts", "sid", "rank_desc"]].copy()

        for k in top_ks:
            for er in exit_ranks:
                if er < k:
                    continue

                m = simulate_turnover(rank_sub, k=k, exit_rank=er)

                row = dict(base)
                row.update(m)

                # fee proxy: bps of capital per day assuming target gross = gross_for_fee_proxy.
                # total_fee_return_bps = turnover_to_capital * fee_bps
                g = float(args.gross_for_fee_proxy)
                fee = float(args.fee_bps)
                row["fee_bps_cap_day_full_topk_gross_proxy"] = (
                    row["full_topk_turnover_per_day"] * g * fee
                )
                row["fee_bps_cap_day_stateful_gross_proxy"] = (
                    row["stateful_turnover_per_day"] * g * fee
                )

                rows.append(row)

        # save incrementally
        tmp = pd.DataFrame(rows)
        tmp.to_csv(out_path, index=False)
        print("[partial saved]", out_path)

        show_cols = [
            "model", "horizon_min", "rebalance_min",
            "rank_ic_mean", "top_bottom_bps",
            "k", "exit_rank",
            "topk_retention_next_topk",
            "topk_retention_next_exit",
            "full_topk_turnover_per_day",
            "stateful_turnover_per_day",
            "fee_bps_cap_day_stateful_gross_proxy",
        ]
        print(tmp[tmp["horizon_min"].eq(horizon) & tmp["rebalance_min"].eq(rb_min)][show_cols].to_string(index=False))

    res = pd.DataFrame(rows)
    if res.empty:
        raise SystemExit("no result")

    res = res.sort_values(
        ["stateful_turnover_per_day", "rank_ic_mean", "top_bottom_bps"],
        ascending=[True, False, False],
    )
    res.to_csv(out_path, index=False)

    print("\n===== FINAL SORTED BY LOW STATEFUL TURNOVER =====")
    show_cols = [
        "model", "horizon_min", "rebalance_min",
        "rank_ic_mean", "top_bottom_bps",
        "top50_target_bps", "top100_target_bps",
        "k", "exit_rank",
        "topk_retention_next_topk",
        "topk_retention_next_exit",
        "full_topk_turnover_per_day",
        "stateful_turnover_per_day",
        "fee_bps_cap_day_stateful_gross_proxy",
    ]
    show_cols = [c for c in show_cols if c in res.columns]
    print(res[show_cols].to_string(index=False))
    print("\n[saved]", out_path)


if __name__ == "__main__":
    main()
