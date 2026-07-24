import numpy as np
import pandas as pd


def fallback_alpha_weights(df_t, prev_weight_map, cfg):
    c = cfg["columns"]
    ocfg = cfg["optimizer"]

    symbol_col = c["symbol_col"]
    alpha_col = c["alpha_col"]

    alpha = pd.to_numeric(df_t[alpha_col], errors="coerce").fillna(0.0)
    alpha = alpha - alpha.mean()

    if alpha.abs().sum() == 0:
        w = np.zeros(len(df_t))
    else:
        w = alpha / alpha.abs().sum()
        w = w * float(ocfg.get("gross_limit", 1.0))

    single = float(ocfg.get("single_name_limit", 0.02))
    w = np.clip(w, -single, single)

    gross = np.abs(w).sum()
    gross_limit = float(ocfg.get("gross_limit", 1.0))
    if gross > gross_limit and gross > 0:
        w = w / gross * gross_limit

    # enforce net close to zero
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

    symbol_col = c["symbol_col"]
    alpha_col = c["alpha_col"]
    industry_col = c.get("industry_col", "industryID1")

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

    alpha = pd.to_numeric(df_t[alpha_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    alpha = alpha - np.nanmean(alpha)
    alpha_std = np.nanstd(alpha)
    if alpha_std > 0:
        alpha = alpha / alpha_std

    vol = pd.to_numeric(df_t.get("volatility_z", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    risk_diag = 1.0 + np.square(vol)

    symbols = df_t[symbol_col].astype(str).tolist()
    w_prev = np.array([prev_weight_map.get(s, 0.0) for s in symbols], dtype=float)

    w = cp.Variable(n)

    risk_aversion = float(ocfg.get("risk_aversion", 1.0))
    turnover_penalty = float(ocfg.get("turnover_penalty", 0.05))

    objective = cp.Maximize(
        alpha @ w
        - risk_aversion * cp.sum(cp.multiply(risk_diag, cp.square(w)))
        - turnover_penalty * cp.norm1(w - w_prev)
    )

    constraints = [
        cp.sum(w) == 0,
        cp.norm1(w) <= float(ocfg.get("gross_limit", 1.0)),
        w <= float(ocfg.get("single_name_limit", 0.02)),
        w >= -float(ocfg.get("single_name_limit", 0.02)),
        cp.norm1(w - w_prev) <= float(ocfg.get("turnover_limit", 0.30)),
    ]

    for exposure_col, limit_key in [
        ("size_z", "size_limit"),
        ("liquidity_z", "liquidity_limit"),
        ("volatility_z", "volatility_limit"),
    ]:
        if exposure_col in df_t.columns:
            x = pd.to_numeric(df_t[exposure_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            lim = float(ocfg.get(limit_key, 0.10))
            constraints += [
                x @ w <= lim,
                x @ w >= -lim,
            ]

    if industry_col in df_t.columns:
        industry = df_t[industry_col].where(df_t[industry_col].notna(), "UNKNOWN").map(str)
        X = pd.get_dummies(industry).to_numpy(dtype=float)
        lim = float(ocfg.get("industry_limit", 0.03))
        constraints += [
            X.T @ w <= lim,
            X.T @ w >= -lim,
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

    out = df_t[[symbol_col]].copy()
    out["target_weight"] = np.asarray(w.value).reshape(-1)
    out["optimizer_status"] = status
    return out
