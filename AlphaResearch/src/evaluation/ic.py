import pandas as pd



def calc_ic(
    df,
    factor,
    label
):

    x=df[
        [factor,label]
    ].dropna()

    if len(x)<10:
        return None

    return x[factor].corr(
        x[label],
        method="pearson"
    )



def calc_rank_ic(
    df,
    factor,
    label
):

    x=df[
        [factor,label]
    ].dropna()

    if len(x)<10:
        return None

    return x[factor].corr(
        x[label],
        method="spearman"
    )



def factor_report(
    df,
    factors,
    horizons
):

    results=[]


    for f in factors:

        for h in horizons:

            label=f"ret_{h}"


            results.append(
                {
                    "factor":f,
                    "horizon":h,
                    "IC":
                    calc_ic(
                        df,
                        f,
                        label
                    ),

                    "RankIC":
                    calc_rank_ic(
                        df,
                        f,
                        label
                    )
                }
            )


    return pd.DataFrame(results)
