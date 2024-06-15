from sklearn.metrics import mean_absolute_percentage_error
import xgboost as xgb
import matplotlib.pyplot as plt
import numpy as np
from model import model_interface


class XGBoost(model_interface.ModelInterface):
    """XGBoost regression class"""

    def __init__(
        self,
        learning_rate=0.1,
        gamma=0,
        max_depth=7,
        reg_lambda=1.0,
        reg_alpha=1.0,
        tree_method="hist",
        n_estimators=50,
        subsample=0.7,
        colsample_bytree=0.8,
        custom_name=None,
        train_x=None,
        train_y=None,
        val_x=None,
        val_y=None,
        device="cuda"
    ):
        """_summary_

        Args:
            learning_rate (float, optional): Learning_rate hyperparameter for XGBoost. Defaults to 0.1.
            gamma (int, optional): Gamma hyperparameter for XGBoost. Defaults to 0.
            max_depth (int, optional): Max_depth hyperparameter for XGBoost. Defaults to 7.
            reg_lambda (float, optional): Reg_lambda hyperparameter for XGBoost. Defaults to 1.0.
            reg_alpha (float, optional): Reg_alpha _hyperparameter for XGBoost Defaults to 1.0.
            tree_method (str, optional): Tree_method hyperparameter for XGBoost. Defaults to "hist".
            n_estimators (int, optional): N_estimators hyperparameter for XGBoost. Defaults to 50.
            subsample (float, optional): Subsample hyperparameter for XGBoost. Defaults to 0.7.
            colsample_bytree (float, optional): Colsample_bytree hyperparameter for XGBoost. Defaults to 0.8.
            custom_name: Custom name to differentiate save locations. Defaults to None.
            train_x: training inputs. Defaults to None.
            train_y: training targets. Defaults to None.
            val_x: validation.inputs Defaults to None.
            val_y: validation targets. Defaults to None.
            device (str, optional): Device to run on. Defaults to "cuda".
        """
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.max_depth = max_depth
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.tree_method = tree_method
        self.n_estimators = n_estimators
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.model = None
        self.custom_name = custom_name
        self.train_x = train_x
        self.train_y = train_y
        self.val_x = val_x
        self.val_y = val_y
        self.device = device

    def _get_custom_name(self):
        if self.custom_name is None:
            return self._get_name()
        else:
            return self.custom_name

    def _get_name(self):
        return "XGBoost"

    def forward(self, x):
        """Predict input x
        Args:
            x: input to predict
        Returns:
            prediction
        """
        d = xgb.DMatrix(x)
        return self.model.predict(d)

    def train(self, train_x, train_y, val_x, val_y):
        """Method to train weights of XGB model
        
            Args:
                train_x: inputs dataset for training data
                train_y: targets for training data
                val_x: inputs dataset for validation data
                val_y: targets for validation data
        """
        if self.train_x is not None:
            train_x = self.train_x
            train_y = self.train_y
            val_x = self.val_x
            val_y = self.val_y
        
        # Initialize XGBoost parameters
        params = {
            "device": self.device,
            "objective": "reg:squarederror",
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "max_depth": self.max_depth,
            "reg_lambda": self.reg_lambda,
            "reg_alpha":self.reg_alpha,
            "tree_method": self.tree_method,
            "n_estimators": self.n_estimators,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            # Add other parameters as needed
        }
        # Convert data to XGBoost DMatrix format
        dtrain = xgb.DMatrix(train_x, label=train_y)
        dvalid = xgb.DMatrix(val_x, label=val_y)

        # Train the XGBoost model with early stopping
        evals = [(dtrain, "train"), (dvalid, "valid")]
        evals_result = {}

        if self.model is None:
            self.model = xgb.train(
                params,
                dtrain,
                num_boost_round=self.n_estimators,
                evals=evals,
                evals_result=evals_result,
                verbose_eval=True, 
                early_stopping_rounds=100,
                feval=lambda preds, dtrain: (
                    "mape",
                    mean_absolute_percentage_error(dtrain.get_label(), preds),
                ),
            )

        else:
            xgb.train(
                params,
                dtrain,
                num_boost_round=self.n_estimators,
                evals=evals,
                evals_result=evals_result,
                verbose_eval=True,
                xgb_model=self.model,
                feval=lambda preds, dtrain: (
                    "mape",
                    mean_absolute_percentage_error(dtrain.get_label(), preds),
                ),
            )

    def save(self, path):
        """Saves the current model

        Args:
            path: location to save at

        Returns:
            boolean describing if save was successful
        """
        try:
            self.model.save_model(path + "/xgb_model.json")
            return True
        except Exception as e:
            print("could not save", e)
            return False

    def plot_feature_importance(self, x_train):
        feature_importance = self.model.get_score(importance_type="gain")

        # Sorting feature importance dictionary by importance value
        sorted_feature_importance = sorted(
            feature_importance.items(), key=lambda x: x[1], reverse=True
        )

        # Extracting feature names and importance scores separately
        sorted_feature_names = [
            list(x_train.columns)[:-1][int(x[0][1:])] for x in sorted_feature_importance
        ]
        sorted_importance_scores = [x[1] for x in sorted_feature_importance]

        # Plotting feature importance with actual feature names
        plt.figure(figsize=(8, 6))
        plt.barh(sorted_feature_names[:20], sorted_importance_scores[:20])
        plt.xlabel("Feature Importance")
        plt.title("Feature Importance")
        plt.gca().invert_yaxis()
        plt.show()
        
    def evaluate(self, X, y):
        pass


def load_xgboost(path):
    """Loads linreg model

    Args:
        path: location where model is stored
    """
    model1 = xgb.Booster()
    model1.load_model(path)
    xgb_model = XGBoost()
    xgb_model.model = model1
    return xgb_model


def mape_eval(preds, dtrain):
    labels = dtrain.get_label()
    error = np.mean(np.abs((labels - preds) / labels))
    return "MAPE", error

