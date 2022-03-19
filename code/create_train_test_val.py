from typing import Callable
import pandas as pd
from scipy.stats import skew
import numpy as np


class CreateData:
    def __init__(
        self,
        df_path: str,
    ) -> None:
        # Read csv
        df = pd.read_csv(df_path)
        df = df.drop(
            columns=[
                "Id",
                "1stFlrSF",
                "GarageArea",
                "GrLivArea",
            ]
        )

        # Feature engg
        df, object_cols = self.feature_engg_v2(df)

        # Train-valid-test
        self.tr = df[~df.YrSold.isin([2009, 2010])]
        self.val = df[df.YrSold == 2009]
        self.te = df[df.YrSold == 2010]
        self.object_cols = object_cols

    def feature_engg(self, df: pd.DataFrame) -> pd.DataFrame:
        """log transform skewed numeric features"""
        numeric_feats = df.dtypes[df.dtypes != "object"].index
        skewed_feats = df[numeric_feats].apply(
            lambda x: skew(x.dropna())
        )  # compute skewness
        skewed_feats = skewed_feats[skewed_feats > 0.75]
        skewed_feats = skewed_feats.index
        object_cols = df.select_dtypes(include=["object"]).columns
        for col in object_cols:
            df[col] = df[col].astype("category")
        df.dtypes[df.dtypes == "object"]
        df = df.fillna(df.mean())
        df[skewed_feats] = np.log1p(df[skewed_feats])
        return df, object_cols

    def feature_engg_v2(self, df: pd.DataFrame) -> pd.DataFrame:
        """log transform skewed numeric features"""
        numeric_feats = list(df.dtypes[df.dtypes != "object"].index)
        numeric_feats.remove("SalePrice")
        skewed_feats = df[numeric_feats].apply(
            lambda x: skew(x.dropna())
        )  # compute skewness
        skewed_feats = skewed_feats[skewed_feats > 0.75]
        skewed_feats = skewed_feats.index
        object_cols = df.select_dtypes(include=["object"]).columns
        df = pd.get_dummies(df)
        df = df.fillna(df.mean())
        df[skewed_feats] = np.log1p(df[skewed_feats])
        return df, object_cols
