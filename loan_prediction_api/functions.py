import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class MeanByState(BaseEstimator, TransformerMixin):
    def __init__(self, column=None):
        self.column = column
        self.dti_ratio_by_state = None
        self.feature_name = "mean_" + column + "_by_state"

    def fit(self, X, y=None):
        self.dti_ratio_by_state = X.groupby("state").agg(mean=(self.column, "mean"))
        return self

    def transform(self, X, y=None):
        #         X[self.feature_name] = X['state'].transform(lambda x: self.dti_ratio_by_state.loc[x])
        X[self.feature_name] = X["state"].transform(lambda x: self.dti_ratio_by_state.loc[x])
        return X

    def get_feature_name(self):
        return self.feature_name


def get_month(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    X["month"] = pd.DatetimeIndex(X["date"]).month
    X.drop("date", axis=1, inplace=True)
    return X


def get_month_cyclic(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    X["month_sin"] = np.sin(X["month"] / 12 * 2 * np.pi)
    X["month_cos"] = np.cos(X["month"] / 12 * 2 * np.pi)
    X.drop("month", axis=1, inplace=True)
    return X


def get_cr_line_year(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    X["cr_line_year"] = pd.DatetimeIndex(X["earliest_cr_line"]).year
    return X


def get_multi_month(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    X["issue_month"] = pd.DatetimeIndex(X["issue_d"]).month
    X["cr_line_month"] = pd.DatetimeIndex(X["earliest_cr_line"]).month
    X.drop(columns=["issue_d", "earliest_cr_line"], inplace=True)
    return X
