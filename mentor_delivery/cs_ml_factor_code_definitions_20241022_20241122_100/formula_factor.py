#%%
import polars as pl  
import inspect

EPS = 1e-12

# =========================================================
# helper exprs
# =========================================================

def _ret_expr(window: int) -> pl.Expr:
    return pl.col("midprice") / pl.col("midprice").shift(window).over("SecurityID") - 1


# =========================================================
# fwz1
# =========================================================

def fwz1_ret_1() -> pl.Expr:
    return _ret_expr(1).alias("fwz1_ret_1")


def fwz1_ret_3() -> pl.Expr:
    return _ret_expr(3).alias("fwz1_ret_3")


def fwz1_ret_5() -> pl.Expr:
    return _ret_expr(5).alias("fwz1_ret_5")


def fwz1_ret_10() -> pl.Expr:
    return _ret_expr(10).alias("fwz1_ret_10")


def fwz1_ret_30() -> pl.Expr:
    return _ret_expr(30).alias("fwz1_ret_30")

def fwz1_ret_45() -> pl.Expr:
    return pl.col('fwz1_ret_1').rolling_sum(window_size=45).over('SecurityID').alias('fwz1_ret_45')

def fwz1_ret_check() -> pl.Expr:
    return pl.col("fwz1_ret_1").rolling_rank(window_size=20).over('SecurityID').alias('fwz1_ret_check')

def cf_fwz1_ret_1_mean() -> pl.Expr:
    factor_name = 'cf_fwz1_ret_1_mean'.split('cf_')[1].split('_mean')[0]
    return pl.col(factor_name).mean().over('timestamp').alias("cf_fwz1_ret_1_mean")

def cf_fwz1_ret_30_mean() -> pl.Expr:
    factor_name = 'cf_fwz1_ret_30_mean'.split('cf_')[1].split('_mean')[0]
    return pl.col(factor_name).mean().over('timestamp').alias("cf_fwz1_ret_30_mean")

def cf_fwz1_ret_check_mean() -> pl.Expr:
    factor_name = 'cf_fwz1_ret_check_mean'.split('cf_')[1].split('_mean')[0]
    return pl.col(factor_name).mean().over('timestamp').alias("cf_fwz1_ret_check_mean")


# =========================================================
# fwz2
# =========================================================
eps = 1e-12

def _eps() -> float:
    return 1e-12


def _depth_expr(side: str, n: int) -> pl.Expr:
    return sum(pl.col(f"{side}volume{i}") for i in range(1, n + 1))


def _obi_expr(n: int) -> pl.Expr:
    eps = _eps()
    bid_depth = _depth_expr("bid", n)
    ask_depth = _depth_expr("ask", n)
    return (bid_depth - ask_depth) / (bid_depth + ask_depth + eps)


def _weighted_depth_expr(side: str, n: int) -> pl.Expr:
    return sum(pl.col(f"{side}volume{i}") / i for i in range(1, n + 1))


def _weighted_obi_expr(n: int) -> pl.Expr:
    eps = _eps()
    bid_depth = _weighted_depth_expr("bid", n)
    ask_depth = _weighted_depth_expr("ask", n)
    return (bid_depth - ask_depth) / (bid_depth + ask_depth + eps)


def _roll(col: str, window: int) -> pl.Expr:
    return pl.col(col).rolling_sum(window_size=window).over("SecurityID")


def _roll_imbalance(buy_col: str, sell_col: str, window: int) -> pl.Expr:
    eps = _eps()
    buy = _roll(buy_col, window)
    sell = _roll(sell_col, window)
    return (buy - sell) / (buy + sell + eps)


def _trade_imbalance_expr(window: int) -> pl.Expr:
    return _roll_imbalance("activate_buy_volume", "activate_sell_volume", window)


def _order_imbalance_expr(window: int) -> pl.Expr:
    return _roll_imbalance("activate_buy_order_volume", "activate_sell_order_volume", window)


def _cancel_pressure_expr(window: int) -> pl.Expr:
    eps = _eps()
    buy_cancel = _roll("cancel_buy_volume", window)
    sell_cancel = _roll("cancel_sell_volume", window)
    return (sell_cancel - buy_cancel) / (sell_cancel + buy_cancel + eps)


def _near_cancel_pressure_expr(window: int) -> pl.Expr:
    eps = _eps()
    buy_cancel = _roll("near_buy_cancel_volume", window)
    sell_cancel = _roll("near_sell_cancel_volume", window)
    return (sell_cancel - buy_cancel) / (sell_cancel + buy_cancel + eps)


def _trade_order_linear_expr(window: int) -> pl.Expr:
    return 0.6 * _trade_imbalance_expr(window) + 0.4 * _order_imbalance_expr(window)

def fwz2_obi_1() -> pl.Expr:
    return _obi_expr(1).alias("fwz2_obi_1")


def fwz2_obi_3() -> pl.Expr:
    return _obi_expr(3).alias("fwz2_obi_3")


def fwz2_obi_5() -> pl.Expr:
    return _obi_expr(5).alias("fwz2_obi_5")


def fwz2_obi_10() -> pl.Expr:
    return _obi_expr(10).alias("fwz2_obi_10")


def fwz2_weighted_obi_5() -> pl.Expr:
    return _weighted_obi_expr(5).alias("fwz2_weighted_obi_5")


def fwz2_microprice_deviation() -> pl.Expr:
    eps = _eps()
    microprice = (
        pl.col("askprice1") * pl.col("bidvolume1")
        + pl.col("bidprice1") * pl.col("askvolume1")
    ) / (pl.col("bidvolume1") + pl.col("askvolume1") + eps)

    return (microprice / (pl.col("midprice") + eps) - 1).alias("fwz2_microprice_deviation")


def fwz2_depth_imbalance_5() -> pl.Expr:
    return _obi_expr(5).alias("fwz2_depth_imbalance_5")


def fwz2_depth_imbalance_10() -> pl.Expr:
    return _obi_expr(10).alias("fwz2_depth_imbalance_10")


def fwz2_total_depth_5() -> pl.Expr:
    return (_depth_expr("bid", 5) + _depth_expr("ask", 5)).alias("fwz2_total_depth_5")


def fwz2_total_depth_10() -> pl.Expr:
    return (_depth_expr("bid", 10) + _depth_expr("ask", 10)).alias("fwz2_total_depth_10")


def fwz2_book_slope_imbalance_5() -> pl.Expr:
    eps = _eps()

    bid_dist = sum((pl.col("midprice") - pl.col(f"bidprice{i}")).abs() for i in range(1, 6))
    ask_dist = sum((pl.col(f"askprice{i}") - pl.col("midprice")).abs() for i in range(1, 6))

    bid_slope = _depth_expr("bid", 5) / (bid_dist + eps)
    ask_slope = _depth_expr("ask", 5) / (ask_dist + eps)

    return ((bid_slope - ask_slope) / (bid_slope + ask_slope + eps)).alias("fwz2_book_slope_imbalance_5")


def fwz2_bid_concentration_5() -> pl.Expr:
    eps = _eps()
    return (pl.col("bidvolume1") / (_depth_expr("bid", 5) + eps)).alias("fwz2_bid_concentration_5")


def fwz2_ask_concentration_5() -> pl.Expr:
    eps = _eps()
    return (pl.col("askvolume1") / (_depth_expr("ask", 5) + eps)).alias("fwz2_ask_concentration_5")


def fwz2_relative_spread() -> pl.Expr:
    eps = _eps()
    return (pl.col("spread") / (pl.col("midprice") + eps)).alias("fwz2_relative_spread")


def fwz2_volatility_20() -> pl.Expr:
    ret = pl.col("midprice") / pl.col("midprice").shift(1).over("SecurityID") - 1
    return ret.rolling_std(window_size=20).over("SecurityID").alias("fwz2_volatility_20")


def fwz2_volatility_60() -> pl.Expr:
    ret = pl.col("midprice") / pl.col("midprice").shift(1).over("SecurityID") - 1
    return ret.rolling_std(window_size=60).over("SecurityID").alias("fwz2_volatility_60")


def fwz2_active_buy_ratio_10() -> pl.Expr:
    eps = _eps()
    return (
        _roll("activate_buy_volume", 10)
        / (_roll("volume", 10) + eps)
    ).alias("fwz2_active_buy_ratio_10")


def fwz2_trade_imbalance_5() -> pl.Expr:
    return _trade_imbalance_expr(5).alias("fwz2_trade_imbalance_5")


def fwz2_trade_imbalance_10() -> pl.Expr:
    return _trade_imbalance_expr(10).alias("fwz2_trade_imbalance_10")


def fwz2_trade_imbalance_20() -> pl.Expr:
    return _trade_imbalance_expr(20).alias("fwz2_trade_imbalance_20")


def fwz2_trade_imbalance_30() -> pl.Expr:
    return _trade_imbalance_expr(30).alias("fwz2_trade_imbalance_30")


def fwz2_amount_imbalance_10() -> pl.Expr:
    return _roll_imbalance("activate_buy_amount", "activate_sell_amount", 10).alias("fwz2_amount_imbalance_10")


def fwz2_vwap_deviation_10() -> pl.Expr:
    eps = _eps()
    amount = _roll("activate_buy_amount", 10) + _roll("activate_sell_amount", 10)
    volume = _roll("volume", 10)
    vwap = amount / (volume + eps)

    return (vwap / (pl.col("midprice") + eps) - 1).alias("fwz2_vwap_deviation_10")


def fwz2_trade_count_imbalance_10() -> pl.Expr:
    return _roll_imbalance("activate_buy_trade_count", "activate_sell_trade_count", 10).alias("fwz2_trade_count_imbalance_10")


def fwz2_trade_intensity_10() -> pl.Expr:
    return _roll("trade_count", 10).alias("fwz2_trade_intensity_10")


def fwz2_large_trade_imbalance_10() -> pl.Expr:
    return _roll_imbalance("large_buy_trade_volume", "large_sell_trade_volume", 10).alias("fwz2_large_trade_imbalance_10")


def fwz2_order_imbalance_5() -> pl.Expr:
    return _order_imbalance_expr(5).alias("fwz2_order_imbalance_5")


def fwz2_order_imbalance_10() -> pl.Expr:
    return _order_imbalance_expr(10).alias("fwz2_order_imbalance_10")


def fwz2_order_imbalance_20() -> pl.Expr:
    return _order_imbalance_expr(20).alias("fwz2_order_imbalance_20")


def fwz2_order_imbalance_30() -> pl.Expr:
    return _order_imbalance_expr(30).alias("fwz2_order_imbalance_30")


def fwz2_order_count_imbalance_10() -> pl.Expr:
    return _roll_imbalance("activate_buy_order_count", "activate_sell_order_count", 10).alias("fwz2_order_count_imbalance_10")


def fwz2_aggressive_order_imbalance_10() -> pl.Expr:
    return _roll_imbalance("aggressive_buy_order_volume", "aggressive_sell_order_volume", 10).alias("fwz2_aggressive_order_imbalance_10")


def fwz2_order_distance_imbalance_10() -> pl.Expr:
    eps = _eps()

    buy_dist = _roll("buy_order_distance_amount", 10) / (_roll("activate_buy_order_volume", 10) + eps)
    sell_dist = _roll("sell_order_distance_amount", 10) / (_roll("activate_sell_order_volume", 10) + eps)

    return ((sell_dist - buy_dist) / (sell_dist + buy_dist + eps)).alias("fwz2_order_distance_imbalance_10")


def fwz2_cancel_pressure_10() -> pl.Expr:
    return _cancel_pressure_expr(10).alias("fwz2_cancel_pressure_10")


def fwz2_cancel_pressure_20() -> pl.Expr:
    return _cancel_pressure_expr(20).alias("fwz2_cancel_pressure_20")


def fwz2_near_cancel_pressure_10() -> pl.Expr:
    return _near_cancel_pressure_expr(10).alias("fwz2_near_cancel_pressure_10")


def fwz2_buy_cancel_ratio_10() -> pl.Expr:
    eps = _eps()
    return (
        _roll("cancel_buy_volume", 10)
        / (_roll("activate_buy_order_volume", 10) + eps)
    ).alias("fwz2_buy_cancel_ratio_10")


def fwz2_sell_cancel_ratio_10() -> pl.Expr:
    eps = _eps()
    return (
        _roll("cancel_sell_volume", 10)
        / (_roll("activate_sell_order_volume", 10) + eps)
    ).alias("fwz2_sell_cancel_ratio_10")


def fwz2_cancel_ratio_imbalance_10() -> pl.Expr:
    eps = _eps()

    buy_ratio = _roll("cancel_buy_volume", 10) / (_roll("activate_buy_order_volume", 10) + eps)
    sell_ratio = _roll("cancel_sell_volume", 10) / (_roll("activate_sell_order_volume", 10) + eps)

    return ((sell_ratio - buy_ratio) / (sell_ratio + buy_ratio + eps)).alias("fwz2_cancel_ratio_imbalance_10")


def fwz2_fake_liquidity_imbalance() -> pl.Expr:
    eps = _eps()

    bid_depth = _depth_expr("bid", 5)
    ask_depth = _depth_expr("ask", 5)

    buy_cancel_ratio = _roll("cancel_buy_volume", 10) / (_roll("activate_buy_order_volume", 10) + eps)
    sell_cancel_ratio = _roll("cancel_sell_volume", 10) / (_roll("activate_sell_order_volume", 10) + eps)

    fake_bid = bid_depth * buy_cancel_ratio
    fake_ask = ask_depth * sell_cancel_ratio

    return ((fake_ask - fake_bid) / (fake_ask + fake_bid + eps)).alias("fwz2_fake_liquidity_imbalance")


def fwz2_trade_order_linear_10() -> pl.Expr:
    return _trade_order_linear_expr(10).alias("fwz2_trade_order_linear_10")


def fwz2_trade_obi_linear_10() -> pl.Expr:
    return (
        0.7 * _trade_imbalance_expr(10)
        + 0.3 * _obi_expr(5)
    ).alias("fwz2_trade_obi_linear_10")


def fwz2_order_obi_linear_10() -> pl.Expr:
    return (
        0.7 * _order_imbalance_expr(10)
        + 0.3 * _obi_expr(5)
    ).alias("fwz2_order_obi_linear_10")


def fwz2_trade_order_obi_linear_10() -> pl.Expr:
    return (
        0.5 * _trade_imbalance_expr(10)
        + 0.3 * _order_imbalance_expr(10)
        + 0.2 * _obi_expr(5)
    ).alias("fwz2_trade_order_obi_linear_10")


def fwz2_trade_cancel_linear_10() -> pl.Expr:
    return (
        0.6 * _trade_imbalance_expr(10)
        + 0.4 * _cancel_pressure_expr(10)
    ).alias("fwz2_trade_cancel_linear_10")


def fwz2_obi_cancel_adjusted_10() -> pl.Expr:
    return (
        0.6 * _obi_expr(5)
        + 0.4 * _cancel_pressure_expr(10)
    ).alias("fwz2_obi_cancel_adjusted_10")


def fwz2_liquidity_adjusted_trade_order_10() -> pl.Expr:
    eps = _eps()

    trade_order = _trade_order_linear_expr(10)
    rel_spread = pl.col("spread") / (pl.col("midprice") + eps)

    return (
        trade_order / (1 + rel_spread * 10000)
    ).alias("fwz2_liquidity_adjusted_trade_order_10")

def fwz2_aggressive_ratio_imbalance_10() -> pl.Expr:
    eps = _eps()

    buy_ratio = _roll("aggressive_buy_order_volume", 10) / (_roll("activate_buy_order_volume", 10) + eps)
    sell_ratio = _roll("aggressive_sell_order_volume", 10) / (_roll("activate_sell_order_volume", 10) + eps)

    return (
        (buy_ratio - sell_ratio) / (buy_ratio + sell_ratio + eps)
    ).alias("fwz2_aggressive_ratio_imbalance_10")


def fwz2_fill_efficiency_imbalance_10() -> pl.Expr:
    eps = _eps()

    buy_eff = _roll("activate_buy_volume", 10) / (_roll("activate_buy_order_volume", 10) + eps)
    sell_eff = _roll("activate_sell_volume", 10) / (_roll("activate_sell_order_volume", 10) + eps)

    return (
        (buy_eff - sell_eff) / (buy_eff + sell_eff + eps)
    ).alias("fwz2_fill_efficiency_imbalance_10")


def fwz2_large_trade_ratio_imbalance_10() -> pl.Expr:
    eps = _eps()

    buy_ratio = _roll("large_buy_trade_volume", 10) / (_roll("activate_buy_volume", 10) + eps)
    sell_ratio = _roll("large_sell_trade_volume", 10) / (_roll("activate_sell_volume", 10) + eps)

    return (
        (buy_ratio - sell_ratio) / (buy_ratio + sell_ratio + eps)
    ).alias("fwz2_large_trade_ratio_imbalance_10")


#=========================================================
# fwz_f1 - fwz_f20
#=========================================================

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


# =========================================================
# dict
# =========================================================
#%%
def create_function_dict():
    function_dict = {}
    current_module = inspect.currentframe().f_back.f_globals  # get a factor_func_dict

    for name, obj in current_module.items():
        if inspect.isfunction(obj) and not name.startswith("_") and name != "create_function_dict":
            function_dict[name] = obj

    return function_dict

function_dict = create_function_dict()