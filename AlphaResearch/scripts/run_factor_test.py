import sys

sys.path.append(
    "/mnt/data1/fwz/HFT_010-dev_fwz"
)

from AlphaResearch.src.data.loader import load_market_data

from AlphaResearch.src.factor.basic_factor import generate_factors

from AlphaResearch.src.label.future_return import add_future_returns

from AlphaResearch.src.evaluation.ic import factor_report



def main():

    print("loading")

    df=load_market_data()


    print("factor generation")

    df=generate_factors(df)


    horizons=[
        10,   #5s
        20,   #10s
        60,   #30s
        120   #60s
    ]


    print("label")

    df=add_future_returns(
        df,
        horizons
    )


    factors=[
        "obi",
        "microprice_dev",
        "microprice_obi",
        "delta_obi_10",
        "delta_obi_acc"
    ]


    print("evaluation")


    result=factor_report(
        df,
        factors,
        horizons
    )


    print(result)


    result.to_csv(
        "AlphaResearch/outputs/factor_ic_report.csv",
        index=False
    )



if __name__=="__main__":
    main()
