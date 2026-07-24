import pandas as pd
import numpy as np


def add_operator_factors(
    df,
    base_factors
):

    df=df.copy()


    for f in base_factors:


        # =================
        # delta
        # =================

        df[f"{f}_delta_10"] = (
            df[f]
            -
            df.groupby("SecurityID")[f]
            .shift(10)
        )


        # =================
        # rolling mean
        # =================

        df[f"{f}_mean_20"] = (
            df.groupby("SecurityID")[f]
            .rolling(
                20,
                min_periods=5
            )
            .mean()
            .reset_index(
                level=0,
                drop=True
            )
        )


        # =================
        # rolling std
        # =================

        df[f"{f}_std_20"] = (
            df.groupby("SecurityID")[f]
            .rolling(
                20,
                min_periods=5
            )
            .std()
            .reset_index(
                level=0,
                drop=True
            )
        )


        # =================
        # nonlinear
        # =================

        df[f"{f}_square"] = (
            df[f] ** 2
        )


        df[f"{f}_sign_sqrt"] = (
            np.sign(df[f])
            *
            np.sqrt(
                np.abs(df[f])
            )
        )


    return df
