import numpy as np

EPS=1e-12



def add_midprice(df):

    df["midprice"] = (
        df["bidprice1"]
        +
        df["askprice1"]
    ) / 200

    return df



def add_obi(df,n=5):

    bid=sum(
        df[f"bidvolume{i}"]
        for i in range(1,n+1)
    )

    ask=sum(
        df[f"askvolume{i}"]
        for i in range(1,n+1)
    )

    df["obi"] = (
        (bid-ask)
        /
        (bid+ask+EPS)
    )

    return df





def add_microprice(df):

    bidprice = df["bidprice1"].astype(float)
    askprice = df["askprice1"].astype(float)

    bidvol = df["bidvolume1"].astype(float)
    askvol = df["askvolume1"].astype(float)


    denom = bidvol + askvol


    micro_tick = (
        askprice * bidvol
        +
        bidprice * askvol
    ) / (denom + EPS)


    micro = micro_tick / 100.0


    df["microprice"] = micro


    valid = (
        (bidvol > 0)
        &
        (askvol > 0)
        &
        (df["midprice"] > 0)
    )


    df["microprice_dev"] = np.where(
        valid,
        micro / df["midprice"] - 1,
        np.nan
    )


    return df

def add_derived_factors(df):

    # microprice * order book imbalance
    df["microprice_obi"] = (
        df["microprice_dev"]
        *
        df["obi"]
    )


    # OBI change (500ms tick)
    df["delta_obi_10"] = (
        df["obi"]
        -
        df.groupby("SecurityID")["obi"]
        .shift(10)
    )


    # OBI acceleration
    df["delta_obi_acc"] = (
        df["delta_obi_10"]
        -
        df.groupby("SecurityID")["delta_obi_10"]
        .shift(10)
    )


    return df


def generate_factors(df):

    df=df.sort_values(
        ["SecurityID","datetime"]
    )

    df=add_midprice(df)

    df=add_obi(df)

    df=add_microprice(df)

    df=add_derived_factors(df)

    return df
