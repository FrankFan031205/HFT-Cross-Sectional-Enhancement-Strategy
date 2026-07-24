import sys

sys.path.append(
    "/mnt/data1/fwz/HFT_010-dev_fwz"
)


from AlphaResearch.src.data.loader import load_market_data

from AlphaResearch.src.factor.basic_factor import generate_factors

from AlphaResearch.src.label.future_return import add_future_returns

from AlphaResearch.src.evaluation.ic import (
    calc_ic,
    calc_rank_ic
)

from AlphaResearch.src.evaluation.rolling_ic import (
    rolling_ic,
    calc_icir
)

from AlphaResearch.src.evaluation.quantile import (
    quantile_return
)

from AlphaResearch.src.evaluation.correlation import (
    factor_corr
)



def main():


    print("loading")

    df=load_market_data()


    print("factors")

    df=generate_factors(df)


    print("labels")

    df=add_future_returns(
        df,
        [120]
    )


    factors=[
        "obi",
        "microprice_dev",
        "microprice_obi",
        "delta_obi_10",
        "delta_obi_acc"
    ]


    label="ret_120"



    # ====================
    # IC
    # ====================

    print("\n===== IC =====")


    for f in factors:

        print(
            f,
            "IC=",
            calc_ic(
                df,
                f,
                label
            ),

            "RankIC=",
            calc_rank_ic(
                df,
                f,
                label
            )
        )



    # ====================
    # Rolling IC
    # ====================

    print("\n===== Rolling IC =====")


    for f in factors:


        ric=rolling_ic(
            df,
            f,
            label,
            window=1000
        )


        print(
            f,
            "ICIR=",
            calc_icir(
                ric["IC"]
            )
        )


        ric.to_csv(
            f"AlphaResearch/outputs/{f}_rolling_ic.csv",
            index=False
        )



    # ====================
    # Quantile
    # ====================

    print("\n===== Quantile =====")


    for f in factors:

        qret=quantile_return(
            df,
            f,
            label
        )

        print(
            "\n",
            f
        )

        print(qret)



    # ====================
    # Correlation
    # ====================

    print("\n===== Correlation =====")


    print(
        factor_corr(
            df,
            factors
        )
    )



if __name__=="__main__":
    main()
