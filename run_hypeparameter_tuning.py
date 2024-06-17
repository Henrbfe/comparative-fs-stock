import sys
from skopt.space.space import Categorical, Real
from data.load_data import date_based_split, load_data_from_csv
from feature_selection.feature_sets import FEATURE_SETS
from model.hyperparameter_tuner import tune_hyperparams_bayes_search

param_spaces = {
    "svr_nystroem": {
        "nystroem__kernel": Categorical(["rbf", "sigmoid"]),
        "nystroem__gamma": Real(1e-3, 1.0),
        "linearsvr__epsilon": Real(1e-5, 1e-2),
        "linearsvr__C": Real(1e-2, 1.0)
    }
}

if __name__ == "__main__":
    estimator = sys.argv[1] if len(sys.argv) > 1 else "svr_nystroem"
    dataset_name = sys.argv[2] if len(sys.argv) > 2 else "jpn"
    dataset_filename = sys.argv[3] if len(sys.argv) > 3 else "dataset/JPN_2010-2024.csv"
    feature_set = sys.argv[4] if len(sys.argv) > 4 else "all"
    n_iter = int(sys.argv[5]) if len(sys.argv) > 5 else 10
    nrows = int(sys.argv[6]) if len(sys.argv) > 6 else None
    random_sampling = bool(sys.argv[7]) if len(sys.argv) > 7 else False

    features = FEATURE_SETS[dataset_name][feature_set]

    TARGET = "return_1w"
    data, features = load_data_from_csv(
        dataset_filename, nrows, features, TARGET, random_sample=random_sampling
    )
    x_train, y_train, x_validation, y_validation, x_test, y_test = date_based_split(
        df=data,
        target_column=TARGET,
        test_size=0.2,
        validation_size=0.25,
        show=False,
        day_of_week=2,
    )

    x_train[TARGET] = y_train

    params = tune_hyperparams_bayes_search(estimator, param_spaces[estimator], x_train, features, TARGET, n_iter=n_iter)
    print(params)
