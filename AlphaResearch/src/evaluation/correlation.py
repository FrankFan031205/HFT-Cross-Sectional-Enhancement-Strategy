import pandas as pd



def factor_corr(
    df,
    factors
):

    return df[
        factors
    ].corr()
