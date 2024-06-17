""" Hyperparameter settings for ML-models tuned on each data set."""

HYPERPARAMS = {
    "fra": {
        "svr": {
            "kernel": "rbf",
            "gamma": 0.0038431468533665324,
            "C": 0.5434412861280024,
            "epsilon": 1e-06,
        }
    },
    "nasnor": {
        "xgboost": {
            "learning_rate": 0.005,
            "gamma": 0.7,
            "max_depth": 6,
            "reg_lambda": 2.3954083094177627,
            "reg_alpha": 2.007179579351735,
            "tree_method": "hist",
            "n_estimators": 500,
            "subsample": 0.8,
            "colsample_bytree": 0.9,
        },
        "svr": {
            "kernel": "rbf",
            "gamma": 0.0038431468533665324,
            "C": 0.5434412861280024,
            "epsilon": 1e-06,
        },
        "tft": {
            "dropout_rate": 0.85,
            "hidden_layer_size": 160,
            "learning_rate": 0.0001,
            "max_gradient_norm": 0.01,
            "minibatch_size": 128,
            "num_heads": 1,
            "stack_size": 1,
        }
    },
    "usa": {
        "xgboost": {
            "learning_rate": 0.003,
            "gamma": 0.7,
            "max_depth": 5,
            "reg_lambda": 2.0,
            "reg_alpha": 1.7,
            "tree_method": "hist",
            "n_estimators": 200,
            "subsample": 0.75,
            "colsample_bytree": 0.75,
        },
        "svr": {
            "kernel": "rbf",
            "gamma": 0.0038431468533665324,
            "C": 0.5434412861280024,
            "epsilon": 1e-06,
        },
        "tft": {
            "dropout_rate": 0.85,
            "hidden_layer_size": 160,
            "learning_rate": 0.0001,
            "max_gradient_norm": 0.01,
            "minibatch_size": 128,
            "num_heads": 1,
            "stack_size": 1,
        }
    },
    "jpn": {
        "xgboost": {
            "learning_rate": 0.04131074133185103,
            "gamma": 0.015167173644731195,
            "max_depth": 4,
            "reg_lambda": 2.2005098832436606,
            "reg_alpha": 0.38798692905014287,
            "tree_method": "hist",
            "n_estimators": 200,# 14,
            "subsample": 0.5765809453808084,
            "colsample_bytree": 0.8889570077602885,
        },
        "svr": {
            "kernel": "rbf",
            "gamma": 0.001,
            "C": 0.001,
            "epsilon": 0.00230604361906557,
        },
        "tft": {
            "dropout_rate": 0.85,
            "hidden_layer_size": 160,
            "learning_rate": 0.0001,
            "max_gradient_norm": 0.01,
            "minibatch_size": 128,
            "num_heads": 1,
            "stack_size": 1,
        }
    }
}
