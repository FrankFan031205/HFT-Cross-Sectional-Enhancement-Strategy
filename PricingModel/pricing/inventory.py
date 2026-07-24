def calc_inventory_penalty(position, tick_size, config):
    lambda_inv = config["pricing"].get("lambda_inv", 0.0)
    return lambda_inv * position * tick_size


def calc_reservation_price(fair_price, position, tick_size, config):
    penalty = calc_inventory_penalty(position, tick_size, config)
    return fair_price - penalty