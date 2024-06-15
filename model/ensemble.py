import numpy as np
import tensorflow as tf
from model import model_interface
from model.tft.libs.tft_model import TemporalFusionTransformer


class Ensemble(model_interface.ModelInterface):
    """Class for representing a ensemble of machine learning models
    """
    def __init__(self, models, custom_name=None, train_x=None, train_y=None, val_x = None, val_y=None) -> None:
        """Initialisation of class

        Args:
            models (list): list of ML models used
            custom_name: Custom name to differentiate save locations. Defaults to None.
            train_x: training inputs. Defaults to None.
            train_y: training targets. Defaults to None.
            val_x: validation.inputs Defaults to None.
            val_y: validation targets. Defaults to None.
        """
        self.models = models
        self.custom_name=custom_name
        self.train_x = train_x
        self.train_y = train_y
        self.val_x = val_x
        self.val_y = val_y
    
    def _get_custom_name(self):
        if self.custom_name is None:
            return self._get_name()
        else:
            return self.custom_name
        
    def __iter__(self):
        return iter(self.models)

    def _get_name(self):
        return "Ensemble"
    
    def forward(self, x, comb_type="geometric"):
        """Method for predicting X

        Args:
            x: Input data to be predicted
            comb_type: Type of combination, either geometric or arithmetic. Defaults to "geometric".

        Returns:
            np.array: predictions
        """
        d = {}
        for model in self.models:
            d[model._get_custom_name] = model.forward(x).flatten()
        return self._combine(x, d, comb_type)

    def _combine(self, x, model_dict: dict, comb_type="geometric"):
        """Helper method for combining predictions

        Args:
            x: Input data to be predicted
            model_dict (dict): dict of models with predictions
            comb_type: Type of combination, either geometric or arithmetic. Defaults to "geometric".

        Returns:
            pred: total prediction
        """
        if comb_type == "geometric":
            pred = np.ones((x.shape[0],))
            for value in model_dict.values():
                pred *=np.clip(value, a_min=0.5, a_max=2)
            pred = pred**(1/len(model_dict.keys()))
        else:
            pred = np.zeros((x.shape[0],))
            for value in model_dict.items():
                pred +=np.clip(value, a_min=0.5, a_max=2)
            pred /=len(model_dict.keys())
        if np.NaN in pred:
            print("NAN",pred)
        return pred
    
    def train(self, train_x, train_y, val_x, val_y):
        """Method to train weights of models in the ensemble

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
        for model in self.models:
            if isinstance(model, TemporalFusionTransformer):
                # Does not have the required inputs to train
                raise ValueError("Ensemble.train is called with a TFT")
            else:
                model.train(train_x, train_y, val_x, val_y)

    def save(self,path):
        """Method for saving models in ensemble

        Args:
            path: Path to save models
        """
        if len(self.models)>1:
            for m in self.models:
                m.save(path+f"/{m._get_custom_name()}")
        else:
            self.models[0].save()
    
    def evaluate(self, X, y):
        return np.mean([model.evaluate(X,y) for model in self.models])
    
    def contains_TFT(self):
        for model in self:
            if isinstance(model, TemporalFusionTransformer):
                return True
        return False
            