def add_future_returns(df, horizons):

    df=df.sort_values(
        ["SecurityID","datetime"]
    )

    for h in horizons:

        future = (
            df.groupby("SecurityID")
            ["midprice"]
            .shift(-h)
        )


        df[f"ret_{h}"] = (
            future /
            df["midprice"]
            -1
        )

    return df
