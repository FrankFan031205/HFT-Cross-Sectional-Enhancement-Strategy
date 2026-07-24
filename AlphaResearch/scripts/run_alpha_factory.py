import sys

sys.path.append(
    "/mnt/data1/fwz/HFT_010-dev_fwz"
)


from AlphaResearch.src.data.loader import load_market_data

from AlphaResearch.src.factor.basic_factor import generate_factors

from AlphaResearch.src.factor.factory import generate_alpha_candidates

from AlphaResearch.src.label.future_return import add_future_returns

from AlphaResearch.src.evaluation.ic import calc_ic,calc_rank_ic



def main():


    print("loading")

    df=load_market_data()


    print("basic factors")

    df=generate_factors(df)


    print("generate candidates")

    df=generate_alpha_candidates(df)


    df=add_future_returns(
        df,
        [120]
    )


    label="ret_120"


    factors=[

        c for c in df.columns

        if (
            c.startswith("obi")
            or
            c.startswith("microprice")
            or
            c.startswith("delta_obi")
        )

    ]


    result=[]


    for f in factors:

        result.append(

            {

            "factor":f,

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


    import pandas as pd


    result=pd.DataFrame(result)

    result=result.sort_values(
        "IC",
        ascending=False
    )


    print(result.head(20))


    result.to_csv(
        "AlphaResearch/outputs/alpha_factory_rank.csv",
        index=False
    )



if __name__=="__main__":
    main()
