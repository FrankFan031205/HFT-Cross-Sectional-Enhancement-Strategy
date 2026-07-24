import polars as pl

EPS = 1e-12


def _c(x):
    return pl.col(x).cast(pl.Float64)


def _sum_expr(exprs):
    out = exprs[0]
    for e in exprs[1:]:
        out = out + e
    return out


def _p(side, i):
    return _c(f"{side}price{i}")


def _v(side, i):
    return _c(f"{side}volume{i}")


def _vol(side, l, r):
    return _sum_expr([_v(side, i) for i in range(l, r + 1)])


def _val(side, l, r):
    return _sum_expr([_p(side, i) * _v(side, i) for i in range(l, r + 1)])


def _mid():
    return _c("midprice")


def _spread():
    return (_p("ask", 1) - _p("bid", 1)).abs()


def _rel_spread():
    return _spread() / (_mid().abs() + EPS)


def _safe_div(a, b):
    return a / (b.abs() + EPS)


def _imb(b, a):
    return (b - a) / (b + a + EPS)


def _log1p_pos(x):
    return (x.abs() + 1.0).log()


def _close_liq(side, l, r):
    exprs = []
    for i in range(l, r + 1):
        if side == "bid":
            dist = (_mid() - _p("bid", i)).abs()
        else:
            dist = (_p("ask", i) - _mid()).abs()
        exprs.append(_v(side, i) / (dist + _spread() + EPS))
    return _sum_expr(exprs)


def fwz_f1_spread_depth_stress():
    depth_val5 = _val("bid", 1, 5) + _val("ask", 1, 5)
    return _safe_div(_rel_spread(), _log1p_pos(depth_val5)).alias("fwz_f1_spread_depth_stress")


def fwz_f2_top_queue_concentration_imb():
    bid5 = _vol("bid", 1, 5)
    ask5 = _vol("ask", 1, 5)
    fac = _safe_div(_v("bid", 1), bid5) - _safe_div(_v("ask", 1), ask5)
    return fac.alias("fwz_f2_top_queue_concentration_imb")


def fwz_f3_outer_depth_imb():
    fac = _imb(_vol("bid", 6, 10), _vol("ask", 6, 10))
    return fac.alias("fwz_f3_outer_depth_imb")


def fwz_f4_near_outer_depth_skew():
    bid_near = _vol("bid", 1, 3)
    bid_outer = _vol("bid", 4, 10)
    ask_near = _vol("ask", 1, 3)
    ask_outer = _vol("ask", 4, 10)
    fac = _safe_div(bid_near, bid_outer) - _safe_div(ask_near, ask_outer)
    return fac.alias("fwz_f4_near_outer_depth_skew")


def fwz_f5_book_convexity_imb():
    bid_conv = _safe_div(_vol("bid", 1, 2) - _vol("bid", 4, 5), _vol("bid", 1, 5))
    ask_conv = _safe_div(_vol("ask", 1, 2) - _vol("ask", 4, 5), _vol("ask", 1, 5))
    fac = bid_conv - ask_conv
    return fac.alias("fwz_f5_book_convexity_imb")


def fwz_f6_value_weighted_depth_imb():
    fac = _imb(_val("bid", 1, 10), _val("ask", 1, 10))
    return fac.alias("fwz_f6_value_weighted_depth_imb")


def fwz_f7_close_liquidity_imb():
    bid_close = _close_liq("bid", 1, 5)
    ask_close = _close_liq("ask", 1, 5)
    fac = _imb(bid_close, ask_close)
    return fac.alias("fwz_f7_close_liquidity_imb")


def fwz_f8_outer_price_slope_asym():
    bid_slope = _safe_div(_p("bid", 1) - _p("bid", 10), _vol("bid", 1, 10))
    ask_slope = _safe_div(_p("ask", 10) - _p("ask", 1), _vol("ask", 1, 10))
    fac = _safe_div(bid_slope - ask_slope, _mid())
    return fac.alias("fwz_f8_outer_price_slope_asym")


def fwz_f9_microprice_concentration_adj():
    micro = (_p("ask", 1) * _v("bid", 1) + _p("bid", 1) * _v("ask", 1)) / (_v("bid", 1) + _v("ask", 1) + EPS)
    bid_conc = _safe_div(_v("bid", 1), _vol("bid", 1, 5))
    ask_conc = _safe_div(_v("ask", 1), _vol("ask", 1, 5))
    conc = (bid_conc + ask_conc) / 2.0
    fac = _safe_div(micro - _mid(), _spread()) * conc
    return fac.alias("fwz_f9_microprice_concentration_adj")


def fwz_f10_queue_imb_spread_gate():
    fac = _imb(_v("bid", 1), _v("ask", 1)) * _rel_spread()
    return fac.alias("fwz_f10_queue_imb_spread_gate")


def fwz_f11_depth_imb_vol_gate():
    depth_imb = _imb(_vol("bid", 1, 10), _vol("ask", 1, 10))
    vol_ratio = _safe_div(_c("ATR6"), _c("ATR14"))
    fac = depth_imb * vol_ratio
    return fac.alias("fwz_f11_depth_imb_vol_gate")


def fwz_f12_bias5_spread_reversal():
    fac = -_c("BIAS5") * _rel_spread()
    return fac.alias("fwz_f12_bias5_spread_reversal")


def fwz_f13_bias_term_momentum():
    fac = _c("BIAS5") - _c("BIAS10")
    return fac.alias("fwz_f13_bias_term_momentum")


def fwz_f14_vol_term_structure():
    fac = _safe_div(_c("VOL5"), _c("VOL20")) - 1.0
    return fac.alias("fwz_f14_vol_term_structure")


def fwz_f15_atr_term_structure():
    fac = _safe_div(_c("ATR6"), _c("ATR14")) - 1.0
    return fac.alias("fwz_f15_atr_term_structure")


def fwz_f16_turnover_size_pressure():
    fac = _safe_div(_log1p_pos(_c("turnoverRate")), _log1p_pos(_c("negMarketValue")))
    return fac.alias("fwz_f16_turnover_size_pressure")


def fwz_f17_turnover_bias_interaction():
    bias_term = _c("BIAS5") - _c("BIAS10")
    fac = _log1p_pos(_c("turnoverRate")) * _safe_div(bias_term, _c("ATR14"))
    return fac.alias("fwz_f17_turnover_bias_interaction")


def fwz_f18_limit_mid_location():
    rng = _c("limit_up_price") - _c("limit_down_price")
    fac = _safe_div(_mid() - _c("limit_down_price"), rng) - 0.5
    return fac.alias("fwz_f18_limit_mid_location")


def fwz_f19_limit_boundary_stress():
    dist_up = (_c("limit_up_price") - _mid()).abs()
    dist_down = (_mid() - _c("limit_down_price")).abs()
    nearest = pl.when(dist_up < dist_down).then(dist_up).otherwise(dist_down)
    nearest_pct = _safe_div(nearest, _mid())
    fac = _safe_div(_rel_spread(), nearest_pct)
    return fac.alias("fwz_f19_limit_boundary_stress")


def fwz_f20_book_limit_interaction():
    depth_imb = _imb(_vol("bid", 1, 10), _vol("ask", 1, 10))
    rng = _c("limit_up_price") - _c("limit_down_price")
    limit_loc = _safe_div((_mid() - _c("limit_down_price")) - (_c("limit_up_price") - _mid()), rng)
    fac = depth_imb * limit_loc
    return fac.alias("fwz_f20_book_limit_interaction")