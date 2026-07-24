import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import yaml


def read_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def mkdir_parent(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def log_print(msg, log_path=None):
    print(msg)
    if log_path:
        mkdir_parent(log_path)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(str(msg) + "\n")


def unique_list(xs):
    ans = []
    for x in xs:
        if x is not None and x not in ans:
            ans.append(x)
    return ans


def normalize_date_series(s):
    raw = s.astype("string").str.strip()
    out = pd.to_datetime(raw, errors="coerce")
    mask = out.notna()
    ans = pd.Series(index=s.index, dtype="float64")

    if mask.any():
        ans.loc[mask] = out.loc[mask].dt.strftime("%Y%m%d").astype("int64")

    mask2 = ans.isna()
    if mask2.any():
        num = pd.to_numeric(raw.loc[mask2], errors="coerce")
        ans.loc[mask2] = num

    if ans.isna().any():
        bad = raw.loc[ans.isna()].head(5).tolist()
        raise ValueError(f"failed to parse date values: {bad}")

    return ans.astype("int64")


def parse_datetime_series(s):
    raw = s.astype("string").str.strip()
    dt = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")

    m1 = raw.str.match(r"^\\d{4}-\\d{2}-\\d{2} \\d{2}:\\d{2}:\\d{2}\\.\\d+$", na=False)
    if m1.any():
        dt.loc[m1] = pd.to_datetime(raw.loc[m1], format="%Y-%m-%d %H:%M:%S.%f", errors="coerce")

    m2 = raw.str.match(r"^\\d{4}-\\d{2}-\\d{2} \\d{2}:\\d{2}:\\d{2}$", na=False)
    if m2.any():
        dt.loc[m2] = pd.to_datetime(raw.loc[m2], format="%Y-%m-%d %H:%M:%S", errors="coerce")

    m3 = raw.str.match(r"^\\d{8}_\\d{6,12}$", na=False)
    if m3.any():
        dt.loc[m3] = pd.to_datetime(raw.loc[m3], format="%Y%m%d_%H%M%S%f", errors="coerce")

    m4 = raw.str.match(r"^\\d{14,20}$", na=False)
    if m4.any():
        dt.loc[m4] = pd.to_datetime(raw.loc[m4], format="%Y%m%d%H%M%S%f", errors="coerce")

    remain = dt.isna()
    if remain.any():
        compact = (
            raw.loc[remain]
            .str.replace("_", "", regex=False)
            .str.replace("-", "", regex=False)
            .str.replace(":", "", regex=False)
            .str.replace(" ", "", regex=False)
            .str.replace(".", "", regex=False)
        )

        parsed = pd.to_datetime(compact, format="%Y%m%d%H%M%S%f", errors="coerce")
        mask2 = parsed.isna()
        if mask2.any():
            parsed2 = pd.to_datetime(compact.loc[mask2], format="%Y%m%d%H%M%S", errors="coerce")
            parsed.loc[mask2] = parsed2

        dt.loc[remain] = parsed

    if dt.isna().any():
        bad = raw.loc[dt.isna()].head(5).tolist()
        raise ValueError(f"failed to parse datetime values: {bad}")

    return dt

def normalize_symbol_series(s):
    raw = s.astype("string").str.strip()
    extracted = raw.str.extract(r"(\d+)")[0]
    num = pd.to_numeric(extracted, errors="coerce")
    if num.notna().all():
        return num.astype("int64")
    return raw


def floor_signed_to_lot(notional, price, lot_size):
    if not np.isfinite(notional) or not np.isfinite(price) or price <= 0:
        return 0

    sign = 1 if notional >= 0 else -1
    qty = abs(notional) / price

    if lot_size > 1:
        qty = np.floor(qty / lot_size) * lot_size
    else:
        qty = np.floor(qty)

    return int(sign * qty)


def load_minute_market(cfg):
    data_cfg = cfg["data"]
    c = cfg["columns"]
    ocfg = cfg["optimizer"]

    path = Path(data_cfg["market_data_path"])
    if not path.exists():
        raise FileNotFoundError(f"market_data_path not found: {path}")

    header = pd.read_csv(path, nrows=0)
    cols = header.columns.tolist()

    needed = unique_list([
        c["date_col"],
        c["datetime_col"],
        c["symbol_col"],
        c["price_col"],
        c.get("bid_col"),
        c.get("ask_col"),
        c.get("spread_col"),
        c.get("bid_volume_col"),
        c.get("ask_volume_col"),
        c.get("limit_up_col"),
        c.get("limit_down_col"),
        c.get("alpha_col"),
    ])

    usecols = [x for x in needed if x in cols]

    missing = [x for x in [c["date_col"], c["datetime_col"], c["symbol_col"], c["price_col"]] if x not in cols]
    if missing:
        raise ValueError(f"market file missing required columns: {missing}")

    chunksize = int(ocfg.get("chunksize", 1000000))
    parts = []

    for i, chunk in enumerate(pd.read_csv(path, usecols=usecols, chunksize=chunksize), 1):
        chunk["date"] = normalize_date_series(chunk[c["date_col"]])
        chunk["__dt"] = parse_datetime_series(chunk[c["datetime_col"]])
        chunk["minute"] = chunk["__dt"].dt.floor("min")
        chunk["securityid"] = normalize_symbol_series(chunk[c["symbol_col"]])

        chunk = chunk.sort_values(["date", "minute", "securityid", "__dt"])
        chunk = chunk.groupby(["date", "minute", "securityid"], as_index=False).tail(1)
        parts.append(chunk)

        if i % 10 == 0:
            print(f"market chunks loaded: {i}")

    df = pd.concat(parts, ignore_index=True)
    df = df.sort_values(["date", "minute", "securityid", "__dt"])
    df = df.groupby(["date", "minute", "securityid"], as_index=False).tail(1)
    df = df.reset_index(drop=True)

    return df


def load_minute_prediction(cfg):
    data_cfg = cfg["data"]
    c = cfg["columns"]
    ocfg = cfg["optimizer"]

    path = data_cfg.get("prediction_path")
    alpha_col = c["alpha_col"]

    if not path:
        return None

    path = Path(path)
    if not path.exists():
        print(f"prediction_path not found: {path}, skip external prediction")
        return None

    header = pd.read_csv(path, nrows=0)
    cols = header.columns.tolist()

    if alpha_col not in cols:
        print(f"alpha_col {alpha_col} not found in prediction file, skip external prediction")
        return None

    needed = unique_list([
        c["date_col"],
        c["datetime_col"],
        c["symbol_col"],
        alpha_col,
    ])

    usecols = [x for x in needed if x in cols]

    missing = [x for x in [c["datetime_col"], c["symbol_col"], alpha_col] if x not in cols]
    if missing:
        raise ValueError(f"prediction file missing required columns: {missing}")

    chunksize = int(ocfg.get("chunksize", 1000000))
    parts = []

    for i, chunk in enumerate(pd.read_csv(path, usecols=usecols, chunksize=chunksize), 1):
        if c["date_col"] in chunk.columns:
            chunk["date"] = normalize_date_series(chunk[c["date_col"]])
        else:
            dt0 = parse_datetime_series(chunk[c["datetime_col"]])
            chunk["date"] = dt0.dt.strftime("%Y%m%d").astype("int64")

        chunk["__dt"] = parse_datetime_series(chunk[c["datetime_col"]])
        chunk["minute"] = chunk["__dt"].dt.floor("min")
        chunk["securityid"] = normalize_symbol_series(chunk[c["symbol_col"]])
        chunk[alpha_col] = pd.to_numeric(chunk[alpha_col], errors="coerce")

        chunk = chunk.dropna(subset=[alpha_col])
        chunk = chunk.sort_values(["date", "minute", "securityid", "__dt"])
        chunk = chunk.groupby(["date", "minute", "securityid"], as_index=False).tail(1)

        parts.append(chunk[["date", "minute", "securityid", alpha_col]])

        if i % 10 == 0:
            print(f"prediction chunks loaded: {i}")

    pred = pd.concat(parts, ignore_index=True)
    pred = pred.sort_values(["date", "minute", "securityid"])
    pred = pred.drop_duplicates(["date", "minute", "securityid"], keep="last")

    return pred


def load_barra_proxy(cfg):
    path = Path(cfg["data"]["barra_proxy_path"])
    if not path.exists():
        raise FileNotFoundError(f"barra_proxy_path not found: {path}")

    c = cfg["columns"]
    style_cols = c.get("barra_style_cols", [])
    industry_col = c.get("industry_col")

    header = pd.read_csv(path, nrows=0)
    cols = header.columns.tolist()

    needed = unique_list(["date", "securityid", industry_col] + style_cols)
    usecols = [x for x in needed if x in cols]

    barra = pd.read_csv(path, usecols=usecols)
    barra["date"] = normalize_date_series(barra["date"])
    barra["securityid"] = normalize_symbol_series(barra["securityid"])

    for col in style_cols:
        if col in barra.columns:
            barra[col] = pd.to_numeric(barra[col], errors="coerce").fillna(0.0)

    if industry_col and industry_col in barra.columns:
        barra[industry_col] = barra[industry_col].fillna("UNKNOWN").astype(str)

    barra = barra.drop_duplicates(["date", "securityid"], keep="last")
    return barra


def fallback_alpha_weights(df_t, prev_weight_map, cfg):
    c = cfg["columns"]
    ocfg = cfg["optimizer"]

    symbol_col = "securityid"
    alpha_col = c["alpha_col"]
    mode = ocfg.get("portfolio_mode", "long_only")

    alpha = pd.to_numeric(df_t[alpha_col], errors="coerce").fillna(0.0)
    alpha = alpha - alpha.mean()

    gross_limit = float(ocfg.get("gross_limit", 1.0))
    single = float(ocfg.get("single_name_limit", 0.02))

    if alpha.abs().sum() == 0:
        w = np.zeros(len(df_t))
    else:
        if mode == "long_only":
            a = alpha.clip(lower=0.0)
            if a.sum() == 0:
                w = np.zeros(len(df_t))
            else:
                w = a / a.sum() * gross_limit
        else:
            w = alpha / alpha.abs().sum() * gross_limit

    if mode == "long_only":
        w = np.clip(w, 0.0, single)
    else:
        w = np.clip(w, -single, single)
        w = w - w.mean()

    gross = np.abs(w).sum()
    if gross > gross_limit and gross > 0:
        w = w / gross * gross_limit

    out = df_t[[symbol_col]].copy()
    out["target_weight"] = w
    out["optimizer_status"] = "fallback_alpha_rescale"
    return out


def solve_one_timestamp(df_t, prev_weight_map, cfg):
    c = cfg["columns"]
    ocfg = cfg["optimizer"]

    symbol_col = "securityid"
    alpha_col = c["alpha_col"]
    industry_col = c.get("industry_col", "industryID1")
    style_cols = [x for x in c.get("barra_style_cols", []) if x in df_t.columns]

    df_t = df_t.copy()
    df_t = df_t.dropna(subset=[alpha_col])

    n = len(df_t)
    if n < int(ocfg.get("min_names_per_timestamp", 20)):
        out = df_t[[symbol_col]].copy()
        out["target_weight"] = 0.0
        out["optimizer_status"] = "too_few_names"
        return out

    try:
        import cvxpy as cp
    except Exception:
        return fallback_alpha_weights(df_t, prev_weight_map, cfg)

    symbols = df_t[symbol_col].astype(str).tolist()

    alpha = pd.to_numeric(df_t[alpha_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    alpha = alpha - np.nanmean(alpha)
    alpha_std = np.nanstd(alpha)

    if alpha_std > 0:
        alpha = alpha / alpha_std

    w_prev = np.array([prev_weight_map.get(s, 0.0) for s in symbols], dtype=float)

    mode = ocfg.get("portfolio_mode", "long_only")
    gross_limit = float(ocfg.get("gross_limit", 0.10))
    single_limit = float(ocfg.get("single_name_limit", 0.005))
    turnover_limit = float(ocfg.get("turnover_limit", 0.03))

    risk_aversion = float(ocfg.get("risk_aversion", 1.0))
    barra_risk_aversion = float(ocfg.get("barra_risk_aversion", 0.2))
    industry_risk_aversion = float(ocfg.get("industry_risk_aversion", 0.1))
    turnover_penalty = float(ocfg.get("turnover_penalty", 0.05))

    risk_diag = np.ones(n)

    if "volatility_z" in df_t.columns:
        vol = pd.to_numeric(df_t["volatility_z"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        risk_diag = 1.0 + np.square(vol)

    w = cp.Variable(n)

    objective_terms = [
        alpha @ w,
        -risk_aversion * cp.sum(cp.multiply(risk_diag, cp.square(w))),
        -turnover_penalty * cp.norm1(w - w_prev),
    ]

    X_style = None

    if len(style_cols) > 0:
        xs = []
        for col in style_cols:
            x = pd.to_numeric(df_t[col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            xs.append(x)
        X_style = np.vstack(xs).T
        objective_terms.append(-barra_risk_aversion * cp.sum_squares(X_style.T @ w))

    X_industry = None

    if industry_col in df_t.columns:
        industry = df_t[industry_col].where(df_t[industry_col].notna(), "UNKNOWN").map(str)
        X_industry = pd.get_dummies(industry).to_numpy(dtype=float)
        objective_terms.append(-industry_risk_aversion * cp.sum_squares(X_industry.T @ w))

    objective = cp.Maximize(sum(objective_terms))

    constraints = [
        cp.norm1(w) <= gross_limit,
        cp.norm1(w - w_prev) <= turnover_limit,
    ]

    if mode == "long_only":
        constraints += [
            w >= 0.0,
            w <= single_limit,
            cp.sum(w) <= gross_limit,
        ]
    else:
        constraints += [
            cp.sum(w) == float(ocfg.get("net_target", 0.0)),
            w <= single_limit,
            w >= -single_limit,
        ]

    style_limit = float(ocfg.get("style_limit", 0.03))

    if X_style is not None:
        for j in range(X_style.shape[1]):
            x = X_style[:, j]
            constraints += [
                x @ w <= style_limit,
                x @ w >= -style_limit,
            ]

    industry_limit = float(ocfg.get("industry_limit", 0.03))

    if X_industry is not None:
        constraints += [
            X_industry.T @ w <= industry_limit,
            X_industry.T @ w >= -industry_limit,
        ]

    prob = cp.Problem(objective, constraints)

    try:
        solver = ocfg.get("solver", "OSQP")
        prob.solve(solver=solver, warm_start=True, verbose=False)
        status = str(prob.status)
    except Exception:
        try:
            prob.solve(warm_start=True, verbose=False)
            status = str(prob.status)
        except Exception as e:
            out = df_t[[symbol_col]].copy()
            out["target_weight"] = 0.0
            out["optimizer_status"] = "solve_error:" + str(e)[:100]
            return out

    if w.value is None:
        return fallback_alpha_weights(df_t, prev_weight_map, cfg)

    target_weight = np.asarray(w.value).reshape(-1)
    target_weight[np.abs(target_weight) < 1e-10] = 0.0

    out = df_t[[symbol_col]].copy()
    out["target_weight"] = target_weight
    out["optimizer_status"] = status
    return out


def build_optimizer_input(cfg):
    c = cfg["columns"]
    alpha_col = c["alpha_col"]

    market = load_minute_market(cfg)

    if alpha_col not in market.columns:
        pred = load_minute_prediction(cfg)
        if pred is None:
            raise ValueError(f"alpha_col {alpha_col} not found in market and prediction cannot be loaded")
        market = market.merge(pred, on=["date", "minute", "securityid"], how="left")

    barra = load_barra_proxy(cfg)
    df = market.merge(barra, on=["date", "securityid"], how="left")

    for col in c.get("barra_style_cols", []):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    industry_col = c.get("industry_col")
    if industry_col and industry_col in df.columns:
        df[industry_col] = df[industry_col].fillna("UNKNOWN").astype(str)

    df[alpha_col] = pd.to_numeric(df[alpha_col], errors="coerce")

    df = df.sort_values(["minute", "securityid"]).reset_index(drop=True)
    return df


def apply_optimizer(cfg):
    data_cfg = cfg["data"]
    c = cfg["columns"]
    ocfg = cfg["optimizer"]

    output_path = data_cfg["output_path"]
    log_path = data_cfg.get("log_path")

    mkdir_parent(output_path)
    if log_path:
        mkdir_parent(log_path)
        Path(log_path).write_text("", encoding="utf-8")

    alpha_col = c["alpha_col"]
    price_col = c["price_col"]
    capital = float(ocfg.get("capital", 1.0))
    lot_size = int(ocfg.get("lot_size", 100))

    df = build_optimizer_input(cfg)

    log_print("===== input summary =====", log_path)
    log_print(f"rows: {len(df)}", log_path)
    log_print(f"minute range: {df['minute'].min()} -> {df['minute'].max()}", log_path)
    log_print(f"num minutes: {df['minute'].nunique()}", log_path)
    log_print(f"num symbols: {df['securityid'].nunique()}", log_path)
    log_print(f"alpha non-null rate: {df[alpha_col].notna().mean():.6f}", log_path)

    prev_weight_map = defaultdict(float)
    prev_qty_map = defaultdict(int)

    all_out = []
    minutes = sorted(df["minute"].dropna().unique())

    for i, minute in enumerate(minutes, 1):
        df_t = df[df["minute"] == minute].copy()

        prev_weight_before = df_t["securityid"].astype(str).map(prev_weight_map).fillna(0.0).astype(float)

        opt_out = solve_one_timestamp(df_t, prev_weight_map, cfg)

        out = df_t.merge(opt_out, on="securityid", how="left")
        out["target_weight"] = out["target_weight"].fillna(0.0)
        out["optimizer_status"] = out["optimizer_status"].fillna("missing_optimizer_output")

        price = pd.to_numeric(out[price_col], errors="coerce")
        out["target_notional"] = out["target_weight"] * capital

        target_qty = []
        current_qty = []
        delta_qty = []

        for _, row in out.iterrows():
            sym = str(row["securityid"])
            px = row[price_col]
            tq = floor_signed_to_lot(row["target_notional"], px, lot_size)
            cq = int(prev_qty_map.get(sym, 0))
            dq = tq - cq

            target_qty.append(tq)
            current_qty.append(cq)
            delta_qty.append(dq)

        out["current_qty"] = current_qty
        out["target_qty"] = target_qty
        out["delta_qty"] = delta_qty

        out["side"] = "NONE"
        out.loc[out["delta_qty"] > 0, "side"] = "BUY"
        out.loc[out["delta_qty"] < 0, "side"] = "SELL"

        out["selected"] = out["target_qty"] != 0
        out["gross_weight"] = np.abs(out["target_weight"])
        out["abs_delta_notional"] = np.abs(out["delta_qty"] * price)

        out["alpha_rank"] = pd.to_numeric(out[alpha_col], errors="coerce").rank(method="first", ascending=False)

        minute_turnover_weight = float(np.abs(out["target_weight"].to_numpy() - prev_weight_before.to_numpy()).sum())

        for _, row in out.iterrows():
            sym = str(row["securityid"])
            prev_weight_map[sym] = float(row["target_weight"])
            prev_qty_map[sym] = int(row["target_qty"])

        keep_cols = unique_list([
            "date",
            "minute",
            c["datetime_col"],
            "securityid",
            alpha_col,
            price_col,
            c.get("bid_col"),
            c.get("ask_col"),
            c.get("spread_col"),
            c.get("bid_volume_col"),
            c.get("ask_volume_col"),
            c.get("limit_up_col"),
            c.get("limit_down_col"),
            c.get("industry_col"),
        ] + c.get("barra_style_cols", []) + [
            "alpha_rank",
            "target_weight",
            "gross_weight",
            "target_notional",
            "current_qty",
            "target_qty",
            "delta_qty",
            "side",
            "selected",
            "abs_delta_notional",
            "optimizer_status",
        ])

        keep_cols = [x for x in keep_cols if x in out.columns]
        all_out.append(out[keep_cols])

        if i % 50 == 0 or i == len(minutes):
            log_print(
                f"minute {i}/{len(minutes)} {minute}, "
                f"rows={len(out)}, selected={int(out['selected'].sum())}, "
                f"gross={float(out['gross_weight'].sum()):.6f}, "
                f"turnover_weight={minute_turnover_weight:.6f}, "
                f"status={out['optimizer_status'].iloc[0]}",
                log_path,
            )

    final = pd.concat(all_out, ignore_index=True)
    final.to_csv(output_path, index=False)

    log_print("===== final summary =====", log_path)
    log_print(f"output_path: {output_path}", log_path)
    log_print(f"rows: {len(final)}", log_path)
    log_print(f"minutes: {final['minute'].nunique()}", log_path)
    log_print(f"avg selected per minute: {final.groupby('minute')['selected'].sum().mean():.4f}", log_path)
    log_print(f"avg gross weight: {final.groupby('minute')['gross_weight'].sum().mean():.6f}", log_path)
    log_print(f"avg abs delta notional per minute: {final.groupby('minute')['abs_delta_notional'].sum().mean():.2f}", log_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = read_yaml(args.config)
    apply_optimizer(cfg)


if __name__ == "__main__":
    main()
