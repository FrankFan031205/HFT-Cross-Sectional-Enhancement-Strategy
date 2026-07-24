import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


class RidgeAlphaModel:

    def __init__(
        self,
        alpha=1.0
    ):

        self.scaler = StandardScaler()

        self.model = Ridge(
            alpha=alpha
        )


    def fit(
        self,
        df,
        factors,
        label
    ):

        data=df[
            factors+[label]
        ].dropna()


        X=data[factors]

        y=data[label]


        X_scaled=self.scaler.fit_transform(
            X
        )


        self.model.fit(
            X_scaled,
            y
        )


        return self



    def predict(
        self,
        df,
        factors
    ):

        data=df[factors].copy()


        X_scaled=self.scaler.transform(
            data
        )


        pred=self.model.predict(
            X_scaled
        )


        result=df.copy()

        result["alpha_score"]=pred


        return result



    def coefficients(
        self,
        factors
    ):

        return pd.DataFrame(
            {
                "factor":factors,
                "weight":self.model.coef_
            }
        )
