from model.hyperparameters import HYPERPARAMS
import numpy as np
import pandas as pd
from niapy.problems import Problem
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, mean_absolute_percentage_error
from sklearn.svm import LinearSVR, SVC
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor
from sklearn.linear_model import SGDRegressor

from evaluation.simulation_suite import SimulationSuite


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


class SVRFeatureSelection(Problem):
    def __init__(
        self, X_train, y_train, X_val, y_val, info_df, alpha=0.99, dataset_name="nasnor"
    ):
        super().__init__(dimension=X_train.shape[1], lower=0, upper=1)
        self.X_train = X_train
        self.y_train = y_train
        self.info_df = info_df
        self.X_val = X_val
        self.y_val = y_val
        self.alpha = alpha
        self.beta = 0.1
        self.selection_thresh = 0.5
        self.dataset_name = dataset_name

    def _evaluate(self, x):
        selected = x > self.selection_thresh
        num_selected = selected.sum()
        if num_selected == 0:
            return np.inf
        model = make_pipeline(
            Nystroem(
                kernel=HYPERPARAMS[self.dataset_name]["svr"]["kernel"],
                gamma=HYPERPARAMS[self.dataset_name]["svr"]["gamma"],
            ),
            LinearSVR(
                epsilon=HYPERPARAMS[self.dataset_name]["svr"]["epsilon"],
                C=HYPERPARAMS[self.dataset_name]["svr"]["C"],
            ),
        )
        model.fit(self.X_train[:, selected], self.y_train)
        preds = model.predict(self.X_val[:, selected])
        pred_df = self.info_df
        pred_df["actual"] = self.y_val
        pred_df["prediction"] = preds
        ret50 = sim(pred_df, "gvkey", "datadate")
        mape = mean_absolute_percentage_error(pred_df["actual"], pred_df["prediction"])
        num_features = self.X_train.shape[1]
        # https://niapy.org/en/stable/tutorials/feature_selection.html
        reg = num_selected / num_features
        return self.beta * mape + (1 - self.beta) * (self.alpha * (-ret50) + (1 - self.alpha) * reg)


class SVMFeatureSelection(Problem):
    def __init__(self, X_train, y_train, alpha=0.99):
        super().__init__(dimension=X_train.shape[1], lower=0, upper=1)
        self.X_train = X_train
        self.y_train = y_train
        self.alpha = alpha

    def _evaluate(self, x):
        selected = x > 0.5
        num_selected = selected.sum()
        if num_selected == 0:
            return 1.0
        accuracy = cross_val_score(
            SVC(), self.X_train[:, selected], self.y_train, cv=2, n_jobs=-1
        ).mean()
        score = 1 - accuracy
        num_features = self.X_train.shape[1]
        return self.alpha * score + (1 - self.alpha) * (num_selected / num_features)


class XGBoostFeatureSelection(Problem):
    def __init__(self, X_train, y_train, alpha=0.99, dataset_name="nasnor"):
        super().__init__(dimension=X_train.shape[1], lower=0, upper=1)
        self.X_train = X_train
        self.y_train = y_train
        self.alpha = alpha
        self.dataset_name = dataset_name

    def _evaluate(self, x):
        selected = x > 0.5
        num_selected = selected.sum()
        if num_selected == 0:
            return 1.0
        model = XGBRegressor(
            device="cuda",
            learning_rate=HYPERPARAMS[self.dataset_name]["xgboost"]["learning_rate"],
            gamma=HYPERPARAMS[self.dataset_name]["xgboost"]["gamma"],
            max_depth=HYPERPARAMS[self.dataset_name]["xgboost"]["max_depth"],
            reg_lambda=HYPERPARAMS[self.dataset_name]["xgboost"]["reg_lambda"],
            reg_alpha=HYPERPARAMS[self.dataset_name]["xgboost"]["reg_alpha"],
            tree_method=HYPERPARAMS[self.dataset_name]["xgboost"]["tree_method"],
            n_estimators=HYPERPARAMS[self.dataset_name]["xgboost"]["n_estimators"],
            subsample=HYPERPARAMS[self.dataset_name]["xgboost"]["subsample"],
            colsample_bytree=HYPERPARAMS[self.dataset_name]["xgboost"][
                "colsample_bytree"
            ],
        )
        split_index = int(0.8 * len(self.X_train))
        X_train = self.X_train[:split_index, selected]
        X_test = self.X_train[split_index:, selected]
        y_train = self.y_train[:split_index]
        y_test = self.y_train[split_index:]
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        mape = mean_absolute_percentage_error(y_test, pred)
        num_features = self.X_train.shape[1]
        return self.alpha * mape + (1 - self.alpha) * (
            abs(num_selected - 8) / num_features
        )


class SGDRegressorFeatureSelection(Problem):
    def __init__(self, X_train, y_train, alpha=0.99, dataset_name="nasnor"):
        print("SGD")
        super().__init__(dimension=X_train.shape[1], lower=0, upper=1)
        self.X_train = X_train
        self.y_train = y_train
        self.alpha = alpha
        self.dataset_name = dataset_name

    def _evaluate(self, x):
        selected = x > 0.5
        num_selected = selected.sum()
        if num_selected == 0:
            return 1.0
        
        model = SGDRegressor(
            early_stopping=HYPERPARAMS[self.dataset_name]["sgd"]["early_stopping"],
            n_iter_no_change=HYPERPARAMS[self.dataset_name]["sgd"]["n_iter_no_change"],
            max_iter=HYPERPARAMS[self.dataset_name]["sgd"]["max_iter"],
            eta0=HYPERPARAMS[self.dataset_name]["sgd"]["eta0"],
        )
        split_index = int(0.8 * len(self.X_train))
        X_train = self.X_train[:split_index, selected]
        X_test = self.X_train[split_index:, selected]
        y_train = self.y_train[:split_index]
        y_test = self.y_train[split_index:]
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        mape = mean_absolute_percentage_error(y_test, pred)
        num_features = self.X_train.shape[1]
        return self.alpha * mape + (1 - self.alpha) * (abs(num_selected-10) / num_features)
