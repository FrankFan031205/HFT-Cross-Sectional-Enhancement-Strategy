import numpy as np


def add_quote_mapping(df, cfg):
    qcfg = cfg.get("quote_mapping", {})
    max_intensity = float(qcfg.get("max_quote_intensity", 1.0))
    weight_scale = float(qcfg.get("weight_scale", 0.02))
    min_quote_weight = float(qcfg.get("min_quote_weight", 0.002))

    w = df["target_weight"].fillna(0.0).astype(float)
    active = w.abs() >= min_quote_weight

    df["quote_intensity"] = 0.0
    df["bid_aggressiveness"] = 0.0
    df["ask_aggressiveness"] = 0.0

    df.loc[active, "quote_intensity"] = (w[active].abs() / weight_scale).clip(0.0, max_intensity)
    df.loc[active, "bid_aggressiveness"] = (w[active].clip(lower=0.0) / weight_scale).clip(0.0, max_intensity)
    df.loc[active, "ask_aggressiveness"] = ((-w[active]).clip(lower=0.0) / weight_scale).clip(0.0, max_intensity)

    df["optimizer_side"] = "neutral"
    df.loc[active & (w > 0), "optimizer_side"] = "bid_bias"
    df.loc[active & (w < 0), "optimizer_side"] = "ask_bias"

    return df
