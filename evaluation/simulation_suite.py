import math
import os
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_absolute_percentage_error,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
import tensorflow as tf
from evaluation import methods
from model.ensemble import Ensemble
from model.tft.libs.tft_model import TemporalFusionTransformer, append_to_sets


class SimulationSuite:
    """Class for simulating and evaluating a model"""

    def __init__(self, model, name, pred_col, date_col, company_col) -> None:
        """Initializer
        Args:
            model: model to use in simulation
            name: name describing model
            pred_col: target column in dataset
            date_col: column with date in dataset
            company_col: company identifier in dataset
        """
        self.model = model
        self.pred_col = pred_col
        self.date_col = date_col
        self.company_col = company_col

        self.path = self.create_path(name, f"./runs/{model._get_name()}")
        print(f"creating model with path {self.path}")

    def train_model(
        self, model, x_train: pd.DataFrame, y_train, x_val: pd.DataFrame, y_val
    ):
        """Trains desired model on given inputs and outputs
        Args:
            x_train: Training inputs
            y_train: Training targets
            epochs: Number of epochs to train with. Defaults to 1000.
        """
        if isinstance(model, TemporalFusionTransformer):
            model.train(x_train, y_train, x_val, y_val, pred_name=self.pred_col)
        elif isinstance(model, Ensemble) and model.contains_TFT():
            for m in model:
                self.train_model(m, x_train, y_train, x_val, y_val)
        else:
            x_train_new = x_train.drop(columns=[self.date_col, self.company_col])
            x_val_new = x_val.drop(columns=[self.date_col, self.company_col])
            model.train(x_train_new.values, y_train, x_val_new.values, y_val)

    def create_path(self, name, folder_path):
        """Creates path where model and images are stored.
        Args:
            name: name of model
        Returns:
           path: location where model is going to be stored.
        """
        if not os.path.exists(folder_path + f"/{name}"):
            print(f"Folder '{folder_path}/{name}' created successfully.")
        else:
            print(f"Folder '{folder_path}/{name}' already exists.")
            latest = 0
            for folder in [
                f
                for f in os.listdir(folder_path)
                if os.path.isdir(os.path.join(folder_path, f))
            ]:
                try:
                    if int(folder.split(sep="_")[-1]) > latest:
                        latest = int(folder.split(sep="_")[-1])
                except:
                    continue
            name = f"{name}_{latest+1}"
        os.makedirs(folder_path + f"/{name}")
        return folder_path + f"/{name}"

    def run_full_test(self, model, x_train, y_train, x_test, y_test, x_val, y_val, version="1"):
        """Run a full test with training, evaluating and saving the model

        Args:
            x_train: dataset with training inputs
            y_train: dataset with training targets
            x_test: dataset with test inputs
            y_test: dataset with test targets
            train_epochs: number of epochs to train with.
        """
        self.train_model(model, x_train, y_train, x_val, y_val)
        model.save(self.path)
        if isinstance(model, Ensemble) and len(model.models) > 1:
            d = {}
            for sub_model in model:
                name, algo_ret5, algo_ret10, algo_ret20, algo_ret50, dates = (
                    self.evaluate_model(
                        model=sub_model,
                        x_test=x_test,
                        y_test=y_test,
                        x_train=x_train,
                        y_train=y_train,
                        x_val=x_val,
                        y_val=y_val,
                        ensemble_part=True,
                    )
                )
                d[name] = {
                    "algo_ret5": algo_ret5,
                    "algo_ret10": algo_ret10,
                    "algo_ret20": algo_ret20,
                    "algo_ret50": algo_ret50,
                    "dates": dates,
                }
            _, algo_ret5, algo_ret10, algo_ret20, algo_ret50, dates = (
                self.evaluate_model(
                    model=model,
                    x_test=x_test,
                    y_test=y_test,
                    x_train=x_train,
                    y_train=y_train,
                    x_val=x_val,
                    y_val=y_val,
                    ensemble_part=False,
                )
            )
            d["Ensemble"] = {
                "algo_ret5": algo_ret5,
                "algo_ret10": algo_ret10,
                "algo_ret20": algo_ret20,
                "algo_ret50": algo_ret50,
                "dates": dates,
            }
            path = self.create_path("comparison", self.path)
            linestyles = ["dashed", "dashdot", "dotted"]
            colors = ["lightgrey", "grey", "darkgrey", "dimgrey"]
            for buy_percentage in (
                "algo_ret5",
                "algo_ret10",
                "algo_ret20",
                "algo_ret50",
            ):
                graph_dict = {}
                for index, (m_name, m_values) in enumerate(d.items()):
                    color = "red" if m_name == "Ensemble" else colors[index % 4]
                    linestyle = (
                        "solid" if m_name == "Ensemble" else linestyles[index % 3]
                    )
                    graph_dict[(m_name, color, linestyle)] = (
                        m_values["dates"],
                        m_values[buy_percentage],
                    )
                methods.plot_simulation(
                    graph_dict,
                    f"Comparison on {buy_percentage}",
                    False,
                    f"{path}/{buy_percentage}",
                )
        else:
            self.evaluate_model(
                model=model,
                x_test=x_test,
                y_test=y_test,
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
                ensemble_part=False,
                version=version
            )
        print(f"Finished running test. Saved in {self.path}")
        return self.path
    
    def get_datasets_for_eval(self,
            model,
            x_test,
            y_test,
            x_train=None,
            y_train=None,
            x_val=None,
            y_val=None):
        """Converts inputs into correct datasets for evaluation

        Args:
            model: ML model
            x_test: Inputs for test
            y_test: targets for test
            x_train: Inputs for train. Defaults to None.
            y_train: targets for train. Defaults to None.
            x_val: Inputs for validation. Defaults to None.
            y_val: targets for validation. Defaults to None.

        Returns:
            train_df, val_df, test_df: converted datasets
        """
        if isinstance(model, TemporalFusionTransformer):
            train_df = self._evaluate_dataset_tft(
                model=model, x=x_train, y=y_train, x2=None, y2=None
            )
            val_df = self._evaluate_dataset_tft(
                model=model, x=x_val, y=y_val, x2=x_train, y2=y_train
            )
            test_df = self._evaluate_dataset_tft(
                model=model, x=x_test, y=y_test, x2=x_val, y2=y_val, save_pred=True
            )

        elif isinstance(model, Ensemble):
            train_df = self._evaluate_dataset_ensemble(
                model=model, x=x_train, y=y_train, x2=None, y2=None
            )
            val_df = self._evaluate_dataset_ensemble(
                model=model, x=x_val, y=y_val, x2=x_train, y2=y_train
            )
            test_df = self._evaluate_dataset_ensemble(
                model=model, x=x_test, y=y_test, x2=x_val, y2=y_val, save_pred=True
            )
        else:
            train_df = self._evaluate_dataset_other(model=model, x=x_train, y=y_train)
            val_df = self._evaluate_dataset_other(model=model, x=x_val, y=y_val)
            test_df = self._evaluate_dataset_other(
                model=model, x=x_test, y=y_test, save_pred=True
            )
        return train_df, val_df, test_df

    def evaluate_model(
        self,
        model,
        x_test,
        y_test,
        x_train=None,
        y_train=None,
        x_val=None,
        y_val=None,
        ensemble_part=False,
        version="1",
        load_testdf_path=False
    ):
        """Calculates metrics and plots to be able to evaluate model. They are found in the path created.
        Args:
            x_test: dataset containing test data
            y_test: dataset containing test labels
            x_train: dataset containing train data
            y_train: dataset containing train labels
            x_val: dataset containing validation data
            y_val: dataset containing validation labels
        """
        if load_testdf_path:
            train_df, val_df, test_df = None, None, pd.read_csv(load_testdf_path)
            test_df[self.date_col] = pd.to_datetime(test_df[self.date_col])
        else:
            train_df, val_df, test_df = self.get_datasets_for_eval(model, x_test, y_test, x_train, y_train, x_val, y_val)
        
        if x_train is not None:
            pred_train, y_train_true = (
                train_df["prediction"].values,
                train_df["actual"].values,
            )
        else:
            pred_train, y_train_true = None, None
        if x_val is not None:
            pred_val, y_val_true = val_df["prediction"].values, val_df["actual"].values
        else:
            pred_val, y_val_true = None, None
        pred_test, y_test_true = test_df["prediction"].values, test_df["actual"].values
        mae = mean_absolute_error(y_test_true, pred_test)
        mse = mean_squared_error(y_test_true, pred_test)
        mape = mean_absolute_percentage_error(y_test_true, pred_test)
        r2 = r2_score(y_test_true, pred_test)
        path = (
            self.create_path(model._get_custom_name(), self.path)
            if ensemble_part
            else self.path
        )

        methods.get_pred_to_return_scatter(
            pred=pred_test,
            actual=y_test_true,
            title="Predicted to return scatter",
            save_name=f"{path}/pred_return_scatter",
            show=False,
        )
        methods.create_histograms(
            pred_array=pred_test,
            actual_array=y_test_true,
            title="Histogram of data distribution",
            bins=100,
            save_name=f"{path}/histogram_scatter",
            show=False,
        )
        methods.get_return_matrix_per_quantile(
            pred_array=pred_test,
            value_array=y_test_true,
            number_of_quantiles=3,
            title="Confusion matrix on a 3-quantile split",
            show=False,
            save_name=f"{path}/confusion_matrix_quantile",
        )
        methods.get_positive_and_negative_return_matrix(
            pred_array=pred_test,
            value_array=y_test_true,
            title="Confusion matrix on positive/negative return",
            show=False,
            save_name=f"{path}/confusion_matrix_pos_neg",
        )
        methods.get_return_quantiles(
            pred_array=pred_test,
            value_array=y_test_true,
            number_of_quantiles=10,
            boxplot=True,
            show=False,
            save_name=f"{path}/deciles",
        )
        if (
            y_train_true is not None
            and y_val_true is not None
            and y_test_true is not None
        ):
            methods.display_train_val_test_difference(
                pred_train_y=pred_train,
                train_y=y_train_true,
                pred_val_y=pred_val,
                val_y=y_val_true,
                pred_test_y=pred_test,
                test_y=y_test_true,
                save_name=f"{path}/dataset_accuracy",
                show=False,
            )
        # litt waste å gjøre dette kallet to ganger
        (
            accuracy,
            (pos_f1, pos_recall, pos_precision),
            (neg_f1, neg_recall, neg_precision),
            mcc,
        ) = methods.get_accuracy_based_metrics(
            pred=pred_test, test_y=y_test_true, threshold=1.0
        )

        index, index_daily, algo_ret5, algo_daily_5, _, _, _, _, dates, num_trades5 = (
            self.simulate_test_data(
                model_name=model._get_custom_name(),
                pred_df=test_df,
                buy_sell_percentage=0.05,
                rebalance=False,
                show_graph=False,
                save_name_for_graph=f"{path}/algo5",
            )
        )
        _, _, algo_ret10, algo_daily_10, _, _, _, _, dates, num_trades10 = self.simulate_test_data(
            model_name=model._get_custom_name(),
            pred_df=test_df,
            buy_sell_percentage=0.1,
            rebalance=False,
            show_graph=False,
            save_name_for_graph=f"{path}/algo10",
        )
        _, _, algo_ret20, algo_daily_20, _, _, _, _, _, num_trades20 = self.simulate_test_data(
            model_name=model._get_custom_name(),
            pred_df=test_df,
            buy_sell_percentage=0.2,
            rebalance=False,
            show_graph=False,
            save_name_for_graph=f"{path}/algo20",
        )
        _, _, algo_ret50, algo_daily_50, _, _, _, _, _, num_trades50 = self.simulate_test_data(
            model_name=model._get_custom_name(),
            pred_df=test_df,
            buy_sell_percentage=0.5,
            rebalance=False,
            show_graph=False,
            save_name_for_graph=f"{path}/algo50",
        )

        (
            longest_drawdown_start5,
            longest_drawdown_end5,
            largest5,
            largest_index5,
            before_index5,
            after_index5,
        ) = methods.get_drawdowns(algo_ret5)
        (
            longest_drawdown_start10,
            longest_drawdown_end10,
            largest10,
            largest_index10,
            before_index10,
            after_index10,
        ) = methods.get_drawdowns(algo_ret10)
        (
            longest_drawdown_start20,
            longest_drawdown_end20,
            largest20,
            largest_index20,
            before_index20,
            after_index20,
        ) = methods.get_drawdowns(algo_ret20)
        (
            longest_drawdown_start50,
            longest_drawdown_end50,
            largest50,
            largest_index50,
            before_index50,
            after_index50,
        ) = methods.get_drawdowns(algo_ret50)

        cal5 = methods.calculate_calmar_ratio(algo_ret5)
        cal10 = methods.calculate_calmar_ratio(algo_ret10)
        cal20 = methods.calculate_calmar_ratio(algo_ret20)
        cal50 = methods.calculate_calmar_ratio(algo_ret50)
        sharpe5 = methods.calculate_sharpe_ratio(algo_ret5)
        sharpe10 = methods.calculate_sharpe_ratio(algo_ret10)
        sharpe20 = methods.calculate_sharpe_ratio(algo_ret20)
        sharpe50 = methods.calculate_sharpe_ratio(algo_ret50)
        d_ratio5, d_return5, d_var5 = methods.get_d_ratio(
            np.log(algo_daily_5), np.log(index_daily), 0.05
        )
        d_ratio10, d_return10, d_var10 = methods.get_d_ratio(
            np.log(algo_daily_10), np.log(index_daily), 0.05
        )
        d_ratio20, d_return20, d_var20 = methods.get_d_ratio(
            np.log(algo_daily_20), np.log(index_daily), 0.05
        )
        d_ratio50, d_return50, d_var50 = methods.get_d_ratio(
            np.log(algo_daily_50), np.log(index_daily), 0.05
        )

        new_row = {
            "type": model._get_name(),
            "path": path,
            "accuracy": accuracy,
            "pos_f1": pos_f1,
            "pos_recall": pos_recall,
            "pos_precision": pos_precision,
            "neg_f1": neg_f1,
            "neg_recall": neg_recall,
            "neg_precision": neg_precision,
            "long_short5": algo_ret5[-1],
            "long_short10": algo_ret10[-1],
            "long_short20": algo_ret20[-1],
            "long_short50": algo_ret50[-1],
            "cal5": cal5,
            "cal10": cal10,
            "cal20": cal20,
            "cal50": cal50,
            "sharpe5": sharpe5,
            "sharpe10": sharpe10,
            "sharpe20": sharpe20,
            "sharpe50": sharpe50,
            "d-ratio5": d_ratio5,
            "d-ratio10": d_ratio10,
            "d-ratio20": d_ratio20,
            "d-ratio50": d_ratio50,
            "d-return5": d_return5,
            "d-return10": d_return10,
            "d-return20": d_return20,
            "d-return50": d_return50,
            "d-variance5": d_var5,
            "d-variance10": d_var10,
            "d-variance20": d_var20,
            "d-variance50": d_var50,
            "profit_per_trade5": 0 if num_trades5 == 0 else algo_ret5[-1] / num_trades5,
            "profit_per_trade10": (
                0 if num_trades10 == 0 else algo_ret10[-1] / num_trades10
            ),
            "profit_per_trade20": (
                0 if num_trades20 == 0 else algo_ret20[-1] / num_trades20
            ),
            "profit_per_trade50": (
                0 if num_trades50 == 0 else algo_ret50[-1] / num_trades50
            ),
            "max_drawdown5": largest5,
            "max_drawdown10": largest10,
            "max_drawdown20": largest20,
            "max_drawdown50": largest50,
            "longest_drawdown_period5": longest_drawdown_end5 - longest_drawdown_start5,
            "longest_drawdown_period10": longest_drawdown_end10 - longest_drawdown_start10,
            "longest_drawdown_period20": longest_drawdown_end20 - longest_drawdown_start20,
            "longest_drawdown_period50": longest_drawdown_end50 - longest_drawdown_start50,
            "largest_drawdown_ratio5": (
                0
                if (largest_index5 - before_index5) == 0
                else (after_index5 - largest_index5) / (largest_index5 - before_index5)
            ),
            "largest_drawdown_ratio10": (
                0
                if (largest_index10 - before_index10) == 0
                else (after_index10 - largest_index10)
                / (largest_index10 - before_index10)
            ),
            "largest_drawdown_ratio20": (
                0
                if (largest_index20 - before_index20) == 0
                else (after_index20 - largest_index20)
                / (largest_index20 - before_index20)
            ),
            "largest_drawdown_ratio50": (
                0
                if (largest_index50 - before_index50) == 0
                else (after_index50 - largest_index50)
                / (largest_index50 - before_index50)
            ),
            "mse": mse,
            "mae": mae,
            "mape": mape,
            "r2_score": r2,
            "MCC": mcc,
        }
        self.insert_row(path=f"./results//model_results_v{version}.csv", row=new_row)

        self.insert_row(path=f'./results/model_store_all_returns_v{version}.csv', row={
            'type': model._get_name(),
            'custom_name': model._get_custom_name(),
            'path': path,
            'dates': dates,
            'index': index.tolist(),
            'long_short5': algo_ret5.tolist(),
            'long_short10': algo_ret10.tolist(),
            'long_short20': algo_ret20.tolist(),
            'long_short50': algo_ret50.tolist(),
        })
        
        ## path,accuracy,pos_f1,pos_recall,pos_precision,neg_f1,neg_recall,neg_precision,MAPE,long_short5,long_short10,long_short20,cal5,cal10,cal20,sharpe5,sharpe10,sharpe20,d-ratio5,d-ratio10,d-ratio20,profit_per_trade5,profit_per_trade10,profit_per_trade20
        return path.split("/")[-1], algo_ret5, algo_ret10, algo_ret20, algo_ret50, dates

    def insert_row(self, row, path):
        """Helper method for inserting row into csv from path

        Args:
            row (dict): dict of row to insert
            path: path of csv
        """
        if os.path.isfile(path):
            df = pd.read_csv(path)
        else:
            df = pd.DataFrame()
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df.to_csv(path, index=False)

    def _evaluate_dataset_ensemble(
        self, model: Ensemble, x, y, x2=None, y2=None, save_pred=False, convert_price_to_return=True
    ):
        """Method for creating dataframe with predictions and targets. Used with 
            an Ensemble model type.

        Args:
            model (Ensemble): model ensemble
            x: Inputs for dataset to predict
            y: Targets for dataset to predict
            x2: Inputs for supporting dataset. Used in case of TFT, where the last 
                inputs of the validation sets, should be used in the test set. Defaults to None.
            y2: Targets for supporting dataset.Used in case of TFT, where the last 
                inputs of the validation sets, should be used in the test set. Defaults to None.
            save_pred: boolean to save predictions. Defaults to False.
            convert_price_to_return: To convert price to return, required by simulation. 
                Used in combination with prediction of prices, rahter than return. Defaults to True.

        Returns:
            total_pred: DF used for evaluation
        """
        if x is None:
            return None
        total_pred = x[[self.company_col, self.date_col]].copy()
        total_pred["actual"] = y
        for i, sub_model in enumerate(model):
            if isinstance(sub_model, TemporalFusionTransformer):
                pred_map = self._evaluate_dataset_tft(sub_model, x, y=y, x2=x2, y2=y2, convert_price_to_return=False)
            elif isinstance(sub_model, Ensemble):
                pred_map = self._evaluate_dataset_ensemble(
                    sub_model, x, y=y, x2=x2, y2=y2, convert_price_to_return=False
                )
            else:
                pred_map = self._evaluate_dataset_other(sub_model, x, y=None, convert_price_to_return=False)
            total_pred = total_pred.merge(
                right=pred_map[[self.company_col, self.date_col, "prediction"]],
                on=[self.date_col, self.company_col],
                how="outer",
            )
            total_pred.rename(columns={"prediction": f"pred_{i}"}, inplace=True)
            print(sub_model._get_name(), "pred_map shape:", pred_map.shape)
        total_pred["prediction"] = total_pred[[f"pred_{n}" for n in range(i + 1)]].mean(
            axis=1, skipna=True
        )
        total_pred.dropna(subset=["prediction"], axis=0, inplace=True)
        total_pred = total_pred[
            [self.company_col, self.date_col, "prediction", "actual"]
        ]
        if self.pred_col == "prccd_1w" and convert_price_to_return:
            total_pred["prediction"] = total_pred["prediction"]/x["prccd"]
            total_pred["actual"] = total_pred["actual"]/x["prccd"]
        if save_pred:
            total_pred.to_csv(f"{self.path}/predictions_{model._get_custom_name()}.csv")
        return total_pred

    def _evaluate_dataset_tft(
        self, model: TemporalFusionTransformer, x, y, x2=None, y2=None, save_pred=False, convert_price_to_return=True
    ):
        """Method for creating dataframe with predictions and targets. Used with 
            an TFT model type.

        Args:
            model (Ensemble): model ensemble
            x: Inputs for dataset to predict
            y: Targets for dataset to predict
            x2: Inputs for supporting dataset. Used in case of TFT, where the last 
                inputs of the validation sets, should be used in the test set. Defaults to None.
            y2: Targets for supporting dataset.Used in case of TFT, where the last 
                inputs of the validation sets, should be used in the test set. Defaults to None.
            save_pred: boolean to save predictions. Defaults to False.
            convert_price_to_return: To convert price to return, required by simulation. 
                Used in combination with prediction of prices, rahter than return. Defaults to True.

        Returns:
            pred: DF used for evaluation
        """
        if x is None:
            return None
        df = x.copy()
        df[self.pred_col] = y
        if x2 is not None and y2 is not None:
            df2 = x2.copy()
            df2[self.pred_col] = y2
            df = append_to_sets(
                test_df=df,
                train_df=df2,
                number_of_dates_to_append=model.get_input_size() - 1,
                date_col=self.date_col,
            )
        pred, targets = model.forward(df)
        targets.rename(columns={"t+0": "actual"}, inplace=True)
        pred = pred.merge(
            targets[["actual", "forecast_time", "identifier"]],
            on=["forecast_time", "identifier"],
            how="left",
        )
        pred.rename(
            columns={
                "t+0": "prediction",
                "forecast_time": self.date_col,
                "identifier": self.company_col,
            },
            inplace=True,
        )
        pred = pred[[self.company_col, self.date_col, "prediction", "actual"]]
        if self.pred_col == "prccd_1w" and convert_price_to_return:
            df["prediction"] = df["prediction"]/x["prccd"]
            df["actual"] = df["actual"]/x["prccd"]
        if save_pred:
            pred.to_csv(f"{self.path}/predictions_{model._get_custom_name()}.csv")
        return pred

    def _evaluate_dataset_other(self, model, x, y, save_pred=False, convert_price_to_return=True):
        """Method for creating dataframe with predictions and targets. Used with 
            non-temporal ML models. 

        Args:
            model (Ensemble): model ensemble
            x: Inputs for dataset to predict
            y: Targets for dataset to predict
            save_pred: boolean to save predictions. Defaults to False.
            convert_price_to_return: To convert price to return, required by simulation. 
                Used in combination with prediction of prices, rahter than return. Defaults to True.

        Returns:
            pred: DF used for evaluation
        """
        if x is None:
            return None
        df = x[[self.company_col, self.date_col]].copy()
        df["actual"] = y
        pred = model.forward(
            x.drop([self.date_col, self.company_col], axis=1).astype("float64").values
        )
        if isinstance(pred, tf.Tensor):
            pred = pred.detach().numpy()
        df["prediction"] = pred
        df = df[[self.company_col, self.date_col, "prediction", "actual"]]
        if self.pred_col == "prccd_1w" and convert_price_to_return:
            df["prediction"] = df["prediction"]/x["prccd"]
            df["actual"] = df["actual"]/x["prccd"]
        if save_pred:
            df.to_csv(f"{self.path}/predictions_{model._get_custom_name()}.csv")
        return df

    def simulate_test_data(
        self,
        model_name,
        pred_df,
        buy_sell_percentage=0.05,
        rebalance=False,
        show_graph=True,
        save_name_for_graph=False,
    ):
        """Method for generating the simulation

        Args:
            model_name: name of model used
            pred_df: prediction dataframe
            buy_sell_percentage: Deciding chosen risk-strategy. Defaults to 0.05.
            rebalance: Whether or not to rebalance the portfolio each week. Defaults to False.
            show_graph: Boolean to show graph. Defaults to True.
            save_name_for_graph: Save name for graph. Defaults to False.

        Returns:
            cumsum_mean: numpy array of cumulative index returns
            mean_returns: numpy array of index returns per date
            cumsum_algo: numpy array of cumulative long-short returns
            algo_returns: numpy array of long-short returns per date
            cumsum_long: numpy array of cumulative long returns
            long_returns: numpy array of long returns per date
            cumsum_short: numpy array of cumulative short returns
            short_returns: numpy array of short returns per date
            dates: Dates used for simulation.
            total_num_trades: total number of trades
        """
        (
            mean_returns,
            algo_returns,
            long_returns,
            short_returns,
            dates,
            total_num_trades,
        ) = self._simulate_from_dataframe(pred_df, buy_sell_percentage)

        if rebalance:
            cumsum_algo = np.cumsum(algo_returns - 1) + 1
            cumsum_mean = np.cumsum(mean_returns - 1) + 1
            cumsum_long = np.cumsum(long_returns - 1) + 1
            cumsum_short = np.cumsum(short_returns - 1) + 1
        else:
            cumsum_algo = np.cumprod(algo_returns)
            cumsum_mean = np.cumprod(mean_returns)
            cumsum_long = np.cumprod(long_returns)
            cumsum_short = np.cumprod(short_returns)

        methods.plot_simulation(  # NB: The plot shows the returns one time period ahead. I.e. the
            # date in the plot is based on the input data, but the y is the achieved
            # return from that input, which is in reality achieved one time period later
            {
                ("Long/Short", "blue", "solid"): (dates, cumsum_algo),
                ("Long", "lightgreen", "solid"): (dates, cumsum_long),
                ("Short", "lightcoral", "solid"): (dates, cumsum_short),
                ("Index", "orange", "solid"): (dates, cumsum_mean),
            },
            f"{model_name}: Returns on buy/sell {buy_sell_percentage*100}%",
            show_graph,
            save_name_for_graph,
        )
        return (
            cumsum_mean,
            mean_returns,
            cumsum_algo,
            algo_returns,
            cumsum_long,
            long_returns,
            cumsum_short,
            short_returns,
            dates,
            total_num_trades,
        )

    def _simulate_test_data_tft(
        self, tft: TemporalFusionTransformer, x_test, y_test, buy_sell_percentage=0.05
    ):
        pred: pd.DataFrame = self._evaluate_dataset_tft(tft, x=x_test, y=y_test)
        return self._simulate_from_dataframe(pred, buy_sell_percentage)

    def _simulate_test_data_other(
        self, model, x_test, y_test, buy_sell_percentage=0.05
    ):
        pred = self._evaluate_dataset_other(model, x_test, y_test)
        return self._simulate_from_dataframe(pred, buy_sell_percentage)

    def _simulate_test_data_ensemble(
        self, model, x_test, y_test, buy_sell_percentage=0.05
    ):
        pred = self._evaluate_dataset_ensemble(model, x_test, y_test)
        return self._simulate_from_dataframe(pred, buy_sell_percentage)

    def _simulate_from_dataframe(
        self, prediction_df: pd.DataFrame, buy_sell_percentage=0.05
    ):
        """ Helper method used in simulations.
        Generates simulations from prediction datagrame. 

        Args:
            prediction_df (pd.DataFrame): Dataframe containing predictions
            buy_sell_percentage: strategy used in simulation. Defaults to 0.05.

        Returns:
            mean_returns: numpy array of index returns per date
            algo_returns: numpy array of long-short returns per date
            long_returns: numpy array of long returns per date
            short_returns: numpy array of short returns per date
            dates: dates used for simulation
            total_num_trades: total number of trades
        """
        if prediction_df.empty:
            raise ValueError("Empty DF")

        def _select_top_n(group, n):
            return group.head(n)

        def _select_bottom_n(group, n):
            return group.tail(n)

        sorted_pred = (
            prediction_df.sort_values(by=self.date_col) # Er ikke denne unødvendig??
            .groupby(self.date_col, group_keys=True)
            .apply(lambda x: x.sort_values(by=["prediction"], ascending=False))
            .reset_index(drop=True)
        )
        trade_counts = (
            sorted_pred.groupby(self.date_col)["prediction"]
            .count()
            .apply(lambda x: max(int(x * buy_sell_percentage), 1))
        )

        longs = sorted_pred.groupby(self.date_col).apply(
            lambda g: _select_top_n(g, trade_counts.loc[g.name])
        )
        shorts = sorted_pred.groupby(self.date_col).apply(
            lambda g: _select_bottom_n(g, trade_counts.loc[g.name])
        )
        longs = longs[[self.company_col, "prediction", "actual"]]
        shorts = shorts[[self.company_col, "prediction", "actual"]]

        long_returns = longs.groupby(self.date_col).mean()["actual"]
        short_returns = shorts.groupby(self.date_col).mean()["actual"]
        algo_returns = (long_returns - short_returns) / 2 + 1
        mean_returns = prediction_df.groupby(self.date_col).mean()["actual"]
        total_num_trades = trade_counts.sum()
        dates = algo_returns.index.tolist()
        return (
            mean_returns.values,
            algo_returns.values,
            long_returns.values,
            short_returns.values,
            dates,
            total_num_trades,
        )
