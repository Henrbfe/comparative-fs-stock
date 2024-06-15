"""Wrapper-based feature selection methods."""

import time
import pandas as pd
import numpy as np
from sklearn.kernel_approximation import Nystroem
from sklearn.metrics import make_scorer, mean_absolute_percentage_error
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.svm import SVR, LinearSVR
from feature_selection.methods.selector_interface import SelectorInterface
from model.hyperparameters import HYPERPARAMS

def sim(pred_df: pd.DataFrame, company_col: str, date_col: str):

    buy_sell_percentage = 0.5

    def _select_top_n(group, n):
        return group.head(n)

    def _select_bottom_n(group, n):
        return group.tail(n)

    sorted_pred = (
        pred_df.sort_values(by=date_col)
        .groupby(date_col, group_keys=True)
        .apply(lambda x: x.sort_values(by=["prediction"], ascending=False))
        .reset_index(drop=True)
    )

    trade_counts = (
        sorted_pred.groupby(date_col)["prediction"]
        .count()
        .apply(lambda x: max(int(x * buy_sell_percentage), 1))
    )

    longs = sorted_pred.groupby(date_col).apply(
        lambda g: _select_top_n(g, trade_counts.loc[g.name])
    )
    shorts = sorted_pred.groupby(date_col).apply(
        lambda g: _select_bottom_n(g, trade_counts.loc[g.name])
    )
    longs = longs[[company_col, "prediction", "actual"]]
    shorts = shorts[[company_col, "prediction", "actual"]]

    long_returns = longs.groupby(date_col).mean()["actual"]
    short_returns = shorts.groupby(date_col).mean()["actual"]
    algo_returns = (long_returns - short_returns) / 2 + 1

    return np.cumprod(algo_returns)[-1]

def custom_eval(y, y_pred, info_df=None):
    pred_df = info_df.loc[y.index.values].copy()
    pred_df["actual"] = y
    pred_df["prediction"] = y_pred
    mape = mean_absolute_percentage_error(y, y_pred)
    ret50 = sim(pred_df, "gvkey", "datadate")
    return 0.1 * mape + 0.9 * (-ret50)

def sim(pred_df: pd.DataFrame, company_col: str, date_col: str):

    buy_sell_percentage = 0.5

    def _select_top_n(group, n):
        return group.head(n)

    def _select_bottom_n(group, n):
        return group.tail(n)

    sorted_pred = (
        pred_df.sort_values(by=date_col)
        .groupby(date_col, group_keys=True)
        .apply(lambda x: x.sort_values(by=["prediction"], ascending=False))
        .reset_index(drop=True)
    )

    trade_counts = (
        sorted_pred.groupby(date_col)["prediction"]
        .count()
        .apply(lambda x: max(int(x * buy_sell_percentage), 1))
    )

    longs = sorted_pred.groupby(date_col).apply(
        lambda g: _select_top_n(g, trade_counts.loc[g.name])
    )
    shorts = sorted_pred.groupby(date_col).apply(
        lambda g: _select_bottom_n(g, trade_counts.loc[g.name])
    )
    longs = longs[[company_col, "prediction", "actual"]]
    shorts = shorts[[company_col, "prediction", "actual"]]

    long_returns = longs.groupby(date_col).mean()["actual"]
    short_returns = shorts.groupby(date_col).mean()["actual"]
    algo_returns = (long_returns - short_returns) / 2 + 1

    return np.cumprod(algo_returns)[-1]


def custom_eval(y, y_pred, info_df=None):
    pred_df = info_df.loc[y.index.values].copy()
    pred_df["actual"] = y
    pred_df["prediction"] = y_pred
    mape = mean_absolute_percentage_error(y, y_pred)
    ret50 = sim(pred_df, "gvkey", "datadate")
    return 0.1 * mape + 0.9 * (-ret50)


class ForwardSelector(SelectorInterface):

    def __init__(self, dataset_name="nasnor", estimator_name="svr", cv=3) -> None:
        super().__init__(estimator=None)
        if estimator_name == "xgboost":
            self.estimator = XGBRegressor(
                device="cuda",
                learning_rate=HYPERPARAMS[dataset_name]["xgboost"]["learning_rate"],
                gamma=HYPERPARAMS[dataset_name]["xgboost"]["gamma"],
                max_depth=HYPERPARAMS[dataset_name]["xgboost"]["max_depth"],
                reg_lambda=HYPERPARAMS[dataset_name]["xgboost"]["reg_lambda"],
                reg_alpha=HYPERPARAMS[dataset_name]["xgboost"]["reg_alpha"],
                tree_method=HYPERPARAMS[dataset_name]["xgboost"]["tree_method"],
            )
        elif estimator_name == "sgd":
            self.estimator = SGDRegressor(
                max_iter=HYPERPARAMS[dataset_name]["sgd"]["max_iter"],
                eta0=HYPERPARAMS[dataset_name]["sgd"]["eta0"],
                n_iter_no_change=HYPERPARAMS[dataset_name]["sgd"]["n_iter_no_change"],
                early_stopping=HYPERPARAMS[dataset_name]["sgd"]["early_stopping"],
            )
        else:
            self.estimator = make_pipeline(
                Nystroem(
                    kernel=HYPERPARAMS[dataset_name]["svr"]["kernel"],
                    gamma=HYPERPARAMS[dataset_name]["svr"]["gamma"],
                    n_jobs=-1,
                ),
                LinearSVR(
                    epsilon=HYPERPARAMS[dataset_name]["svr"]["epsilon"],
                    C=HYPERPARAMS[dataset_name]["svr"]["C"],
                ),
            )
        self.cv = cv
        self.tol = None

    def __call__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        parameters: dict,
    ) -> list[str]:
        start_time = time.time()
        cv = parameters["cv"] if "cv" in parameters else self.cv
        tol = parameters["tol"] if "tol" in parameters else self.tol
        n_features_to_select = (
            parameters["n_features_to_select"]
            if "n_features_to_select" in parameters
            else "auto"
        )
        X_train = pd.concat([X, X_val])
        y_train = pd.concat([y, y_val])
        info_df = X_train[["gvkey", "datadate"]]
        sffs = SequentialFeatureSelector(
            estimator=self.estimator,
            cv=cv,
            n_jobs=-1,
            tol=tol,
            n_features_to_select=n_features_to_select,
            scoring=make_scorer(custom_eval, greater_is_better=False, info_df=info_df),
        )
        sffs.fit(X_train.drop(columns=["gvkey", "datadate"]), y_train)
        self.time = time.time() - start_time
        return X.drop(columns=["gvkey", "datadate"]).columns[sffs.get_support(indices=True)].values


class BackwardSelector(SelectorInterface):

    def __init__(
        self, dataset_name="nasnor", estimator=SVR, estimator_name="svr", cv=3
    ) -> None:
        super().__init__(estimator=None)
        if estimator_name == "xgboost":
            self.estimator = XGBRegressor(
                device="cuda",
                learning_rate=HYPERPARAMS[dataset_name]["xgboost"]["learning_rate"],
                gamma=HYPERPARAMS[dataset_name]["xgboost"]["gamma"],
                max_depth=HYPERPARAMS[dataset_name]["xgboost"]["max_depth"],
                reg_lambda=HYPERPARAMS[dataset_name]["xgboost"]["reg_lambda"],
                reg_alpha=HYPERPARAMS[dataset_name]["xgboost"]["reg_alpha"],
                tree_method=HYPERPARAMS[dataset_name]["xgboost"]["tree_method"],
            )
        elif estimator_name == "sgd":
            self.estimator = SGDRegressor(
                max_iter=HYPERPARAMS[dataset_name]["sgd"]["max_iter"],
                eta0=HYPERPARAMS[dataset_name]["sgd"]["eta0"],
                n_iter_no_change=HYPERPARAMS[dataset_name]["sgd"]["n_iter_no_change"],
                early_stopping=HYPERPARAMS[dataset_name]["sgd"]["early_stopping"],
            )
        else:
            self.estimator = make_pipeline(
                Nystroem(
                    kernel=HYPERPARAMS[dataset_name]["svr"]["kernel"],
                    gamma=HYPERPARAMS[dataset_name]["svr"]["gamma"],
                    n_jobs=-1,
                ),
                LinearSVR(
                    epsilon=HYPERPARAMS[dataset_name]["svr"]["epsilon"],
                    C=HYPERPARAMS[dataset_name]["svr"]["C"],
                ),
            )
        self.cv = cv
        self.tol = None

    def __call__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        parameters: dict,
    ) -> list[str]:
        start_time = time.time()
        cv = parameters["cv"] if "cv" in parameters else self.cv
        tol = parameters["tol"] if "tol" in parameters else self.tol
        n_features_to_select = (
            parameters["n_features_to_select"]
            if "n_features_to_select" in parameters
            else "auto"
        )
        X_train = pd.concat([X, X_val])
        y_train = pd.concat([y, y_val])
        info_df = X_train[["gvkey", "datadate"]]

        sffs = SequentialFeatureSelector(
            estimator=self.estimator,
            cv=cv,
            n_jobs=-1,
            tol=tol,
            n_features_to_select=n_features_to_select,
            scoring=make_scorer(custom_eval, greater_is_better=False, info_df=info_df),
            direction="backward",
        )
        sffs.fit(X_train.drop(columns=["gvkey", "datadate"]), y_train)
        self.time = time.time() - start_time
        return X.columns[sffs.get_support(indices=True)].values
