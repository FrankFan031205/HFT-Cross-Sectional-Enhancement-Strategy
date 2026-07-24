import numpy as np


def calc_fair_price(mid_price, pred_ret, beta=1.0, return_type="simple"):
    adj_ret = beta * pred_ret

    if return_type == "simple":
        return mid_price * (1 + adj_ret)

    if return_type == "log":
        return mid_price * np.exp(adj_ret)

    raise ValueError(f"Unknown return_type: {return_type}")