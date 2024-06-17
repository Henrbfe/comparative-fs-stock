"""Script for running simulation suite.."""

from pathlib import Path
import sys
import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
import tensorflow as tf
from data.constants import DATE_COL
from feature_selection.feature_sets import FEATURE_SETS, categorize_features
from data.load_data import load_data_from_csv, date_based_split
from model.hyperparameters import HYPERPARAMS
from model.svr import StockSVR
from model.tft.data_formatters.stock import StockFormatter, format_inputs
from model.tft.libs.hyperparam_opt import HyperparamOptManager
from model.tft.libs.tft_model import TemporalFusionTransformer
from model.tft.methods import get_date_info
from model.xgb_model import XGBoost
from evaluation.simulation_suite import SimulationSuite

use_gpu = tf.config.list_physical_devices('GPU')

BASE_PATH = Path(__file__).parent.resolve()
PRED_COL = 'return_1w'
COMPANY_COL = 'gvkey'


def createTFTwithHyperParamOpt(x_train, y_train, x_val, y_val, hyperparam_iterations, dataset, features, feature_set_name):
    print("Creating TFT_HPO...")
    traindf = x_train.copy()
    traindf[PRED_COL]=y_train
    valdf = x_val.copy()
    valdf[PRED_COL]=y_val

    technical_features, fundamental_features, macro_features, sector_features = categorize_features(features, dataset)
    if len(sector_features) == 0:
        sector_features = ["region"]
    col_def = format_inputs(
        id=COMPANY_COL,
        time=DATE_COL,
        target=PRED_COL,
        real_valued_observed=technical_features+fundamental_features+macro_features,
        real_valued_known=[],
        categorical_known=["week_of_year", "month"],
        categorical_static=sector_features,
    )
    data_formatter = StockFormatter(column_definition=col_def)
    data_formatter.set_scalers(traindf)
    
    fixed_params = data_formatter.get_experiment_params()
    param_ranges = TemporalFusionTransformer.get_hyperparm_choices()
    folder = f"TFT_HPO_{dataset}_experiments_{feature_set_name}"
    fixed_params["model_folder"] = folder
    opt_manager = HyperparamOptManager(param_ranges, fixed_params, folder)
    
    success = opt_manager.load_results()
    if success:
        print("Loaded results from previous training")
    else:
        print("Creating new hyperparameter optimisation")
    opt_manager.clear()
    train_samples, valid_samples = data_formatter.get_num_samples_for_calibration()
    formatted_train = data_formatter.transform_inputs(traindf)
    formatted_val = data_formatter.transform_inputs(valdf)

    while len(opt_manager.results.columns) < hyperparam_iterations:
        print("# Running hyperparam optimisation {} of {} for {}".format(
            len(opt_manager.results.columns) + 1, hyperparam_iterations, "TFT"))
        params = opt_manager.get_next_parameters()
        tft = TemporalFusionTransformer(params, custom_name=f"TFT_HPO_{dataset}", formatter=data_formatter)

        if not tft.training_data_cached():
            tft.cache_batched_data(formatted_train, "train", num_samples=train_samples)
            tft.cache_batched_data(formatted_val, "valid", num_samples=valid_samples)
        tft.fit()
        val_loss = tft.evaluate()

        if np.allclose(val_loss, 0.) or np.isnan(val_loss):
            print("Skipping bad configuration....")
            val_loss = np.inf

        opt_manager.update_score(params, val_loss, tft)
    best_params = opt_manager.get_best_params()
    print("BEST PARAMETERS", best_params)
    return TemporalFusionTransformer(best_params, custom_name=f"TFT_HPO_{dataset}_{feature_set_name}", formatter=data_formatter)

def createTFT(x_train, y_train, dataset, features, feature_set_name):
    print("Creating TFT...")
    traindf = x_train.copy()
    traindf[PRED_COL]=y_train
    traindf = traindf.append(pd.Series(0, index=traindf.columns), ignore_index=True)
    traindf = traindf.append(pd.Series(1, index=traindf.columns), ignore_index=True)

    technical_features, fundamental_features, macro_features, sector_features = categorize_features(features, dataset)
    if len(sector_features) == 0:
        sector_features = ["region"]
    
    col_def = format_inputs(
        id=COMPANY_COL,
        time=DATE_COL,
        target=PRED_COL,
        real_valued_observed=technical_features+fundamental_features+macro_features,
        real_valued_known=[],
        categorical_known=["month"], #"week_of_year", 
        categorical_static=sector_features,
    )

    data_formatter = StockFormatter(column_definition=col_def)
    data_formatter.set_scalers(traindf)
    fixed_params = data_formatter.get_experiment_params()
    params = data_formatter.get_default_model_params() if len(HYPERPARAMS[dataset]["tft"]) == 0 else HYPERPARAMS[dataset]["tft"]
    print(f"Running params: {params}")
    params["model_folder"] = f"TFT_{dataset}_experiments_{feature_set_name}"
    params.update(fixed_params)

    tft = TemporalFusionTransformer(params, custom_name=f"TFT_{dataset}_{feature_set_name}", formatter=data_formatter)
    return tft

def createSVR(dataset, feature_set_name):
    print("Creating SVR...")
    params = HYPERPARAMS[dataset]["svr"]
    svr = StockSVR(kernel=params["kernel"], gamma=params["gamma"], epsilon=params["epsilon"], C=params["C"], custom_name=f"SVR_model_{dataset}_{feature_set_name}")
    return svr

def createXGB(dataset, feature_set_name):
    print("Creating XGBoost...")
    device = "cuda" if use_gpu else None
    params = HYPERPARAMS[dataset]["xgboost"]
    xgb = XGBoost(
        learning_rate=params["learning_rate"],
        gamma=params["gamma"],
        max_depth=params["max_depth"],
        reg_lambda=params["reg_lambda"],
        reg_alpha=params["reg_alpha"],
        tree_method=params["tree_method"],
        n_estimators=params["n_estimators"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        custom_name=f"XGB_model_{dataset}_{feature_set_name}",
        device=device
    )
    return xgb


if __name__ == "__main__":
    dataset_name = (sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1] else "jpn"
    dataset_filename = (sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] else "dataset/JPN_2010-2024.csv"
    feature_set = (sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] else "all"
    model = (sys.argv[4]) if len(sys.argv) > 4 else "xgb"
    n_itr = int(sys.argv[5]) if len(sys.argv) > 5 else 250
    nrows = int(sys.argv[6]) if len(sys.argv) > 6 else None
    if not os.path.exists(dataset_filename):
        raise ValueError(f"Could not locate dataset file {dataset_filename}.")

    print(f"Found {use_gpu} gpus")
    features = FEATURE_SETS[dataset_name][feature_set]

    df, features = load_data_from_csv(
        dataset_filename=dataset_filename,
        nrows=nrows,
        features=features,
        target=PRED_COL)


    if model == 'tft' or model == 'tft_hpo' or model == 'tfthpo' or model == 'tft-hpo':
        df['region'] = dataset_name # to avoid empty static dataset
        df = get_date_info(df)
    x_train, y_train, x_validation, y_validation, x_test, y_test = date_based_split(df=df, target_column=PRED_COL, test_size=0.2, validation_size=0.25, show=False, day_of_week=3)
    if model == "xgb" or model == "xgboost":
        sim_model = createXGB(dataset_name, feature_set)
    elif model == "tft":
        sim_model = createTFT(x_train, y_train, dataset_name, features, feature_set)
    elif model == "tft_hpo" or model == "tfthpo" or model == "tft-hpo":
        sim_model = createTFTwithHyperParamOpt(x_train, y_train, x_validation, y_validation, n_itr, dataset_name, features, feature_set)
    elif model == "svr" or model == "svm":
        sim_model = createSVR(dataset_name, feature_set)
    else:
        raise ValueError("Did not recognize input. Model: ",model)

    sim_suite = SimulationSuite( #uncomment line 485 to write to file
        model=sim_model,
        name=f"{model}_{dataset_name}_{feature_set}",
        pred_col=PRED_COL,
        date_col=DATE_COL,
        company_col=COMPANY_COL,
    )
    print(f"Running simulation on {dataset_name} with {feature_set}: {features}")
    path = sim_suite.run_full_test(model=sim_model, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, x_val=x_validation, y_val=y_validation)
    print("Finished simulation")

    if model == "xgb" or model == "xgboost":
        feature_importance = sim_model.model.get_score(importance_type='weight')
        keys = list(feature_importance.keys())  # Corrected variable name
        values = list(feature_importance.values())

        data = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by="score", ascending=False)
        axsub = data.nlargest(40, columns="score").plot(kind='barh', figsize=(20, 10))  # plot top 40 features
        text_yticklabels = list(axsub.get_yticklabels())
        dict_features = dict(enumerate([col for col in x_train.columns if col not in [DATE_COL, COMPANY_COL]]))
        lst_yticklabels = [text_yticklabels[i].get_text().lstrip('f') for i in range(len(text_yticklabels))]
        lst_yticklabels = [dict_features[int(float(i))] for i in lst_yticklabels]
        axsub.set_yticklabels(lst_yticklabels)
        plt.savefig(path + "/xgb_feature_importance.png")
        plt.show()
    elif model=="tft" or model=="tft_hpo":
        print("timesteps", sim_model.time_steps)
        print("hidden_layer_size", sim_model.hidden_layer_size)
        print("dropout_rate", sim_model.dropout_rate)
        print("max_gradient_norm", sim_model.max_gradient_norm)
        print("learning_rate", sim_model.learning_rate)
        print("minibatch_size", sim_model.minibatch_size)
        print("num_epochs", sim_model.num_epochs)
        print("early_stopping_patience", sim_model.early_stopping_patience)
        print("num_encoder_steps", sim_model.num_encoder_steps)
        print("num_stacks", sim_model.num_stacks)
        print("num_heads", sim_model.num_heads)
