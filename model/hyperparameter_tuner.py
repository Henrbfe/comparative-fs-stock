import time
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.svm import LinearSVR, SVR, SVC
# from thundersvm import SVR as ThunderSVR
from sklearn.metrics import make_scorer, mean_absolute_percentage_error
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor
from skopt import BayesSearchCV  # pip install scikit-optimize

estimators = {
    "svr": SVR(),
    "linsvr" : LinearSVR(),
    "svr_nystroem": make_pipeline(Nystroem(), LinearSVR()),
    # "svr_thunder": ThunderSVR(),
    "svc": SVC(),
    "xgboost": XGBRegressor(tree_method="hist", device="cuda"),
    "sgd": SGDRegressor(),
}

def tune_hyperparams_grid_search(
    estimator_name: str,
    params: dict,
    data: pd.DataFrame,
    features: list[str],
    target: str,
):
    """Tuning hyperparameters using a grid search strategy (running all combinations).

    Args:
        estimator_name (str): The name of the estimator to tune for.
        params (dict): key-value pairs for all hyperparameters with their corresponding search space.
        data (pd.DataFrame): A dataframe containing the training data.
        features (list[str]): A list of the features to use for training.
        target (str): The column name of the target variable.

    Returns:
        grid_result: grid_result
    """
    start_time = time.time()
    scorer = make_scorer(mean_absolute_percentage_error, greater_is_better=False)
    test_fold = (
        [0] * int(len(data) / 3) + [1] * int(len(data) / 3) + [2] * int(len(data) / 3)
    )
    ps = PredefinedSplit(test_fold)
    grid = GridSearchCV(
        estimator=estimators[estimator_name],
        param_grid=params,
        cv=ps,
        scoring=scorer,
        verbose=1,
        error_score="raise",
    )
    grid_result = grid.fit(data[features], data[target])
    print("Results", grid_result)
    print("Best parameters:", grid_result.best_params_)
    print("Best score:", grid_result.best_score_)

    print(f"Finished tuning after {time.time() - start_time} seconds")

    return grid_result


def tune_hyperparams_bayes_search(
    estimator_name: str,
    params: dict,
    data: pd.DataFrame,
    features: list[str],
    target: str,
    scorer=None,
    n_iter=50,
):
    """Tuning hyperparameters using a bayesian search strategy.

    Args:
        estimator_name (str): The name of the estimator to tune for.
        params (dict): Params to search
        data (pd.DataFrame): Data to use for tuning
        features (list[str]): Features to be used for search
        target (str): Name of target variable
        scorer (_type_, optional): Scorer to use for evaluation. Defaults to None.
        n_iter (int, optional): Number of iterations. Defaults to 32.

    Returns:
        _type_: _description_
    """
    optimizer = BayesSearchCV(
        estimators[estimator_name],
        params,
        fit_params={"device": "cuda"},
        scoring=scorer,
        n_iter=n_iter,
        cv=3,
        n_jobs=-1, # max: n_cv (default 3) * n_points
        n_points=5,
        verbose=4,
    )
    optimizer = optimizer.fit(data[features], data[target])
    best_params = {}
    for param in params:
        best_params[param] = optimizer.best_params_[f"{param}"]

    return best_params
