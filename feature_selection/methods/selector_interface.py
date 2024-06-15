import pandas as pd
from typing import Union
from sklearn.base import BaseEstimator
from niapy.problems import Problem


class SelectorInterface:

    def __init__(self, estimator: Union[BaseEstimator, Problem]) -> None:
        self.time = 0
        self.estimator = estimator

    def __call__(self, X: pd.DataFrame, y: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, parameters: dict) -> list[str]:
        pass

    def get_runtime(self) -> int:
        return self.time
