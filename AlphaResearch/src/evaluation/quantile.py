import pandas as pd



def quantile_return(
    df,
    factor,
    label,
    q=5
):

    data=df[
        [factor,label]
    ].dropna()


    data["group"]=pd.qcut(
        data[factor],
        q,
        labels=False
    )


    result=(
        data
        .groupby("group")[label]
        .mean()
        .reset_index()
    )


    result.columns=[
        "group",
        "return"
    ]


    return result
