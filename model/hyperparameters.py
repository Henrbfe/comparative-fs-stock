""" Hyperparameter settings for ML-models tuned on each data set."""

HYPERPARAMS = {
    "nasnor": {
        "xgboost": {
            # 'learning_rate': 0.003, #0.001
            # 'gamma': 0.0, #6.424129273514142
            # 'max_depth': 5,
            # 'reg_lambda': 0, #4.0
            # 'reg_alpha': 0, #3.8004382764387414
            # 'tree_method': 'auto', #exact
            # 'n_estimators': 100, # 200
            # 'subsample': 0.75,
            # 'colsample_bytree': 0.75,
            ## 1000 it:
            # "learning_rate": 0.005, # 0.1
            # "gamma": 0.7, #4.435152267562321,
            # "max_depth": 6,
            # "reg_lambda": 2.3954083094177627,
            # "reg_alpha": 2.007179579351735,
            # "tree_method": "hist",
            # "n_estimators": 500,
            # "subsample": 0.4,
            # "colsample_bytree": 0.9,
            ## 1000 it:
            "learning_rate": 0.005, # 0.1
            "gamma": 0.7, #4.435152267562321,
            "max_depth": 6,
            "reg_lambda": 2.3954083094177627,
            "reg_alpha": 2.007179579351735,
            "tree_method": "hist",
            "n_estimators": 500,
            "subsample": 0.8,
            "colsample_bytree": 0.9,
            # "learning_rate": 0.0001,
            # "gamma": 0.0,
            # "max_depth": 8,
            # "reg_lambda": 2.400911080356045,
            # "reg_alpha": 3.627329820989898e-17,
            # "tree_method": "hist",
            # "n_estimators": 481,
            # "subsample": 0.5182769365158035,
            # "colsample_bytree": 0.41495705632870855,
        },
        "svr": {
            # Updated 30it with nystroem + linearsvr
            "kernel": "rbf",
            "gamma": 0.0038431468533665324,
            "C": 0.5434412861280024,
            "epsilon": 1e-06,
            # "kernel": "sigmoid",
            # "gamma": 0.009290811716863051,
            # "epsilon": 0.040727602250847234,
            # "tol": 1e-3,  # Not tuned
            # "C": 92.28428679675908,
        },
        "tft": {
            # TODO : awaiting results
        },
        "sgd":{
            "n_iter_no_change":1000, 
            "early_stopping":True, 
            "eta0":0.001, 
            "max_iter": 50000,
        }
    },
    "usa": {
        "xgboost": {
            "learning_rate": 0.003,  # 0.001
            "gamma": 0.7,  # 6.424129273514142
            "max_depth": 5,
            "reg_lambda": 2.0,# 4.0,
            "reg_alpha": 1.7, #3.8004382764387414,
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
            # All
            "dropout_rate": 0.85,
            "hidden_layer_size": 160,
            "learning_rate": 0.0001,
            "max_gradient_norm": 0.01,
            "minibatch_size": 128,
            "num_heads": 1,
            "stack_size": 1,
            # mrmr
            # "dropout_rate": 0.85,
            # "hidden_layer_size": 240,
            # "learning_rate": 0.01,
            # "minibatch_size": 128,
            # "max_gradient_norm": 10.0,
            # "num_heads": 1,
            # "stack_size": 1,
            # Fixed
            # 'total_time_steps': 14,
            # 'num_encoder_steps': 12,
            # 'num_epochs': 30,
            # 'early_stopping_patience': 5
        },
        "sgd":{
            "n_iter_no_change":1000, 
            "early_stopping":True, 
            "eta0":0.001, 
            "max_iter": 50000,
        }
    },
    "jpn": {
        "xgboost": {
            # "learning_rate": 0.02343386587701194,
            # "gamma": 0.37781147726561554,
            # "max_depth": 4,
            # "reg_lambda": 0.9086921139215034,
            # "reg_alpha": 0.0001061590848281391,
            # "tree_method": "hist",
            # "n_estimators": 541,
            # "subsample": 0.7274934901213665,
            # "colsample_bytree": 0.5645358707363102,
            ## 1000 iterations
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
            # Updated 30it with nystroem + linearsvr
            "kernel": "rbf",
            "gamma": 0.001,
            "C": 0.001,
            "epsilon": 0.00230604361906557,
        },
        "tft": {},
        "sgd":{
            "n_iter_no_change":1000, 
            "early_stopping":True, 
            "eta0":0.001, 
            "max_iter": 50000,
        }
    },
}

"""

'learning_rate': 0.003, #0.001
'gamma': 0.5, #6.424129273514142
'max_depth': 5,
'reg_lambda': 2.2, #4.0
'reg_alpha': 1.7, #3.8004382764387414
'tree_method': 'auto', #exact
'n_estimators': 200, # 200
'subsample': 0.75,
'colsample_bytree': 0.75

"""
