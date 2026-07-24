import sys

sys.path.append(
    "/mnt/data1/fwz/HFT_010-dev_fwz"
)


from AlphaResearch.src.data.loader import load_market_data

from AlphaResearch.src.factor.basic_factor import generate_factors

from AlphaResearch.src.label.future_return import add_future_returns

from AlphaResearch.src.model.ridge_alpha import RidgeAlphaModel

from AlphaResearch.src.evaluation.ic import (
    calc_ic,
    calc_rank_ic
)



def main():

    print("loading")

    df=load_market_data()


    print("factor")

    df=generate_factors(df)


    df=add_future_returns(
        df,
        [120]
    )


    factors=[
        "obi",
        "microprice_dev",
        "delta_obi_10"
    ]


    label="ret_120"


    print("training ridge")


    model=RidgeAlphaModel(
        alpha=10
    )


    model.fit(
        df,
        factors,
        label
    )


    print("\nweights")

    print(
        model.coefficients(
            factors
        )
    )


    df=model.predict(
        df,
        factors
    )


    print("\nalpha IC")

    print(
        calc_ic(
            df,
            "alpha_score",
            label
        )
    )


    print(
        calc_rank_ic(
            df,
            "alpha_score",
            label
        )
    )


    df.to_csv(
        "AlphaResearch/outputs/ridge_alpha_result.csv",
        index=False
    )


if __name__=="__main__":
    main()
