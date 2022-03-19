from scipy.stats import skew
import pandas as pd
import numpy as np


def feature_engg(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering function"""
    numeric_feats = df.dtypes[df.dtypes != "object"].index
    skewed_feats = df[numeric_feats].apply(
        lambda x: skew(x.dropna())
    )  # compute skewness
    skewed_feats = skewed_feats[skewed_feats > 0.75]
    skewed_feats = skewed_feats.index
    df = pd.get_dummies(df)
    df = df.fillna(df.mean())
    df[skewed_feats] = np.log1p(df[skewed_feats])
    return df
