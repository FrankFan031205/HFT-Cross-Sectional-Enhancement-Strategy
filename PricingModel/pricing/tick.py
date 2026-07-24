import numpy as np


def floor_to_tick(price, tick_size):
    return np.floor(price / tick_size) * tick_size


def ceil_to_tick(price, tick_size):
    return np.ceil(price / tick_size) * tick_size


def round_to_tick(price, tick_size):
    return np.round(price / tick_size) * tick_size