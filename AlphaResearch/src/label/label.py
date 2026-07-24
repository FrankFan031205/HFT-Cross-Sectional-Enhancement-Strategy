def add_return_label(df,horizon=60):

    future=(
        df.groupby("SecurityID")
        ["midprice"]
        .shift(-horizon)
    )

    df[
        f"label_{horizon}"
    ]=(
        future/
        df["midprice"]
        -1
    )

    return df
