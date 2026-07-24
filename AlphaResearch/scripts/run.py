import sys

sys.path.append(
    "/mnt/data1/fwz/HFT_010-dev_fwz"
)


from AlphaResearch.src.data.loader import load_market_data

from AlphaResearch.src.factor.basic_factor import generate_factors

from AlphaResearch.src.label.label import add_return_label

from AlphaResearch.src.evaluation.ic import report



def main():

    print("loading data")

    df=load_market_data()

    print("raw:",df.shape)


    print("generating factors")

    df=generate_factors(df)


    print("creating label")

    df=add_return_label(
        df,
        60
    )


    print(report(df))


    report(df).to_csv(
        "AlphaResearch/outputs/ic_report.csv",
        index=False
    )


if __name__=="__main__":
    main()
