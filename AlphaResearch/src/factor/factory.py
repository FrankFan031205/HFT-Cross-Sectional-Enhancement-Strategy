from AlphaResearch.src.factor.operators import (
    add_operator_factors
)



def generate_alpha_candidates(df):


    base_factors=[

        "obi",

        "microprice_dev",

        "delta_obi_10"

    ]


    df=add_operator_factors(
        df,
        base_factors
    )


    # interaction factors


    df["obi_microprice"] = (
        df["obi"]
        *
        df["microprice_dev"]
    )


    df["obi_delta_interaction"] = (
        df["obi"]
        *
        df["delta_obi_10"]
    )


    return df
