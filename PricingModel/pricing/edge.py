def calc_bid_edge(fair_price, bid_price):
    return fair_price - bid_price


def calc_ask_edge(fair_price, ask_price):
    return ask_price - fair_price


def calc_min_edge(mid_price, tick_size, config):
    pcfg = config["pricing"]

    min_edge_ticks = pcfg.get("min_edge_ticks", 1.0)
    min_edge_abs = pcfg.get("min_edge_abs", 0.0)
    fee_rate = pcfg.get("fee_rate", 0.0)

    tick_edge = min_edge_ticks * tick_size
    fee_edge = fee_rate * mid_price

    return max(min_edge_abs, tick_edge) + fee_edge