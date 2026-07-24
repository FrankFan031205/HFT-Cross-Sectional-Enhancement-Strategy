import pandas as pd


def rolling_ic(
    df,
    factor,
    label,
    window=1000
):

    data=df[
        [factor,label]
    ].dropna()


    result=[]


    for i in range(
        window,
        len(data)
    ):

        sample=data.iloc[
            i-window:i
        ]


        ic=sample[factor].corr(
            sample[label],
            method="pearson"
        )


        rank_ic=sample[factor].corr(
            sample[label],
            method="spearman"
        )


        result.append(
            {
                "index":i,
                "IC":ic,
                "RankIC":rank_ic
            }
        )


    return pd.DataFrame(result)



def calc_icir(ic_series):

    return (
        ic_series.mean()
        /
        (ic_series.std()+1e-12)
    )
