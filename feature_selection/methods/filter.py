""" Feature selection methods using filter strategy. """

import time
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from mrmr import mrmr_regression, mrmr_classif #pip install mrmr_selection
from feature_selection.methods.selector_interface import SelectorInterface


class VarianceFilterSelector(SelectorInterface):
    """
    Feature selection method using a variance filter to select all features with variance above
    threshold value.
    """

    def __init__(self, estimator=None) -> None:
        super().__init__(estimator)
        self.threshold = 0.2

    def __call__(self, X: pd.DataFrame, y: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, parameters: dict):
        start_time = time.time()
        var_thresh = (
            parameters["threshold"] if "threshold" in parameters else self.threshold
        )
        X_data = X.drop(columns=["gvkey", "datadate"])
        selector = VarianceThreshold(threshold=var_thresh)
        selector.fit_transform(X_data, y)
        selection = selector.get_support(indices=True)
        self.time = time.time() - start_time
        return X_data.columns[selection].values


class MRMRFilterSelector(SelectorInterface):
    """
    Feature selection method using minimum redundancy maximum relevancy filter.
    """

    def __init__(self, estimator=None, use_regression=True) -> None:
        super().__init__(estimator)
        self.n_features = 20
        self.use_regression = use_regression
        self.relevance = []
        self.redundancy = []

    def get_scores(self):
        return self.relevance, self.redundancy

    def __call__(self, X: pd.DataFrame, y: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, parameters: dict):
        start_time = time.time()
        n_features = (
            parameters["n_features"] if "n_features" in parameters else self.n_features
        )
        # selection, relevance, redundancy = (
        #     mrmr_regression(X, y, n_features, return_scores=True)
        #     if self.use_regression
        #     else mrmr_classif(X, y, n_features)
        # )
        X_data = X.drop(columns=["gvkey", "datadate"])
        selection = (
            mrmr_regression(X_data, y, n_features)
            if self.use_regression
            else mrmr_classif(X_data, y, n_features)
        )
        self.time = time.time() - start_time
        # self.relevance = relevance
        # self.redundancy = redundancy
        return selection
