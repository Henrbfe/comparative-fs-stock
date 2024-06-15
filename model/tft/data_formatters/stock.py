# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Custom formatting functions for stock dataset.

Defines dataset specific column definitions and data transformations.
"""

# from data.load_data import date_based_split
import numpy as np
import model.tft.data_formatters.base
import model.tft.libs.utils as utils
import sklearn.preprocessing

GenericDataFormatter = model.tft.data_formatters.base.GenericDataFormatter
DataTypes = model.tft.data_formatters.base.DataTypes
InputTypes = model.tft.data_formatters.base.InputTypes

class StockFormatter(GenericDataFormatter):
  """Defines and formats data for the stock dataset.

  Attributes:
    column_definition: Defines input and data type of column used in the
      experiment.
    identifiers: Entity identifiers used in experiments.
  """
  
  # ['gvkey','datadate',"sharpe_1w","momentum_1d","momentum_1w","momentum_2w","momentum_4w","rsi","d_oscillator_4w","d_oscillator_6w",'return_7d']
  _column_definition = [
      ('gvkey', DataTypes.CATEGORICAL, InputTypes.ID),
      ('datadate', DataTypes.DATE, InputTypes.TIME),
      ('return_1d', DataTypes.REAL_VALUED, InputTypes.TARGET),
      
      ('sharpe_1w', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('momentum_1d', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('momentum_1w', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('momentum_2w', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('momentum_4w', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('rsi', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('d_oscillator_4w', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('d_oscillator_6w', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('day_of_week', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
      ('day_of_month', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
      ('week_of_year', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
      ('month', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
      #('year', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
      ('region', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
  ]

  def __init__(self, column_definition=None):
    """Initialises formatter."""

    self.identifiers = None
    self._real_scalers = None
    self._cat_scalers = None
    self._target_scaler = None
    self._num_classes_per_cat_input = None
    if column_definition is not None:
      self._column_definition = column_definition
    


  def split_data(self, df, valid_boundary=2021, test_boundary=2022):
    """Splits data frame into training-validation-test data frames.

    This also calibrates scaling object, and transforms data for each split.

    Args:
      df: Source data frame to split.
      valid_boundary: Starting year for validation data
      test_boundary: Starting year for test data

    Returns:
      Tuple of transformed (train, valid, test) data.
    """
    # x_train, y_train, x_validation, y_validation, x_test, y_test = date_based_split()

    print('Formatting train-valid-test splits.')

    index = df['year']
    train = df.loc[index < valid_boundary]
    valid = df.loc[(index >= valid_boundary) & (index < test_boundary)]
    test = df.loc[(index >= test_boundary)]
    
    self.set_scalers(train)
  
    return (self.transform_inputs(data) for data in [train, valid, test]), [train,valid,test]

  def set_scalers(self, df):
    """Calibrates scalers using the data supplied.

    Args:
      df: Data to use to calibrate scalers.
    """
    print('Setting scalers with training data...')
    class ExtendedLabelEncoder(sklearn.preprocessing.LabelEncoder):
      """
      Class for handling unknown
      Created using ChatGPT
      """
      def __init__(self):
        super().__init__()
        
      def fit(self, y):
          super().fit(y)
          self.classes_ = np.append(self.classes_, '<unknown>')  # Add an unknown label
          return self
      
      def transform(self, y):
          unknown_label = '<unknown>'
          y = np.array(y, copy=True)
          mask = ~np.in1d(y, self.classes_)
          y[mask] = unknown_label
          return super().transform(y)

      def inverse_transform(self, y):
          unknown_label = '<unknown>'
          y = np.array(y, copy=True)
          mask = y == len(self.classes_) - 1
          y[mask] = unknown_label
          return super().inverse_transform(y)

    column_definitions = self.get_column_definition()
    id_column = utils.get_single_col_by_input_type(InputTypes.ID,
                                                   column_definitions)
    target_column = utils.get_single_col_by_input_type(InputTypes.TARGET,
                                                       column_definitions)

    # Extract identifiers in case required
    self.identifiers = list(df[id_column].unique())

    # Format real scalers
    real_inputs = utils.extract_cols_from_data_type(
        DataTypes.REAL_VALUED, column_definitions,
        {InputTypes.ID, InputTypes.TIME})

    data = df[real_inputs].values
    self._real_scalers = sklearn.preprocessing.StandardScaler().fit(data)
    self._target_scaler = sklearn.preprocessing.StandardScaler().fit(
        df[[target_column]].values)  # used for predictions

    # Format categorical scalers
    categorical_inputs = utils.extract_cols_from_data_type(
        DataTypes.CATEGORICAL, column_definitions,
        {InputTypes.ID, InputTypes.TIME})

    categorical_scalers = {}
    num_classes = []
    for col in categorical_inputs:
      # Set all to str so that we don't have mixed integer/string columns
      srs = df[col].apply(str)
      # categorical_scalers[col] = ExtendedLabelEncoder().fit(
      categorical_scalers[col] = sklearn.preprocessing.LabelEncoder().fit(
          srs.values)
      num_classes.append(srs.nunique())

    # Set categorical scaler outputs
    self._cat_scalers = categorical_scalers
    self._num_classes_per_cat_input = num_classes

  def transform_inputs(self, df):
    """Performs feature transformations.

    This includes both feature engineering, preprocessing and normalisation.

    Args:
      df: Data frame to transform.

    Returns:
      Transformed data frame.

    """
    output = df.copy()

    if self._real_scalers is None and self._cat_scalers is None:
      raise ValueError('Scalers have not been set!')

    column_definitions = self.get_column_definition()

    real_inputs = utils.extract_cols_from_data_type(
        DataTypes.REAL_VALUED, column_definitions,
        {InputTypes.ID, InputTypes.TIME})
    categorical_inputs = utils.extract_cols_from_data_type(
        DataTypes.CATEGORICAL, column_definitions,
        {InputTypes.ID, InputTypes.TIME})

    # Format real inputs
    output[real_inputs] = self._real_scalers.transform(df[real_inputs].values)

    # Format categorical inputs
    for col in categorical_inputs:
      string_df = df[col].apply(str)
      try: 
        output[col] = self._cat_scalers[col].transform(string_df)
      except ValueError as e:
        print("Problem Col123", col)
        raise ValueError(e)

    return output

  def format_predictions(self, predictions):
    """Reverts any normalisation to give predictions in original scale.

    Args:
      predictions: Dataframe of model predictions.

    Returns:
      Data frame of unnormalised predictions.
    """
    output = predictions.copy()

    column_names = predictions.columns

    for col in column_names:
      if col not in {'forecast_time', 'identifier'}:
        output[col] = self._target_scaler.inverse_transform(predictions[col].values.reshape(1, -1)).flatten()
    return output

  # Default params
  def get_fixed_params(self):
    """Returns fixed model parameters for experiments."""

    fixed_params = {
        'total_time_steps': 12 + 2,
        'num_encoder_steps': 12,
        'num_epochs': 25,
        'early_stopping_patience': 5,
        'multiprocessing_workers': 5,
    }

    return fixed_params

  def get_default_model_params(self):
    """Returns default optimised model parameters."""

    model_params = {
        'dropout_rate': 0.6,
        'hidden_layer_size': 250,
        'learning_rate': 0.0002,
        'minibatch_size': 256,
        'max_gradient_norm': 0.1,
        'num_heads': 3,
        'stack_size': 1
    }

    return model_params
  
  def get_num_samples_for_calibration(self):
    return 300000, 50000

def format_inputs(id:str, time:str, target:str, real_valued_observed: list[str]=[], real_valued_known: list[str]=[], categorical_known: list[str]=[], categorical_static: list[str]=[])->list:
  """_summary_

  Args:
      id (string): id column
      time (string): time column
      target (string): target column
      real_valued_observed (list): list of real_valued_observed values. Defaults to []. Example: observed trading volume
      real_valued_known (list): list of real_valued_known values. Defaults to []. Example: day of week
      categorical_known (list):list of categorical_known values. Defaults to []. Example: day of week
      categorical_static (list): list of categorical_static values. Defaults to []. Example: region
  """
  l = [(id, DataTypes.CATEGORICAL, InputTypes.ID),
      (time, DataTypes.DATE, InputTypes.TIME),
      (target, DataTypes.REAL_VALUED, InputTypes.TARGET)]
  
  for col in real_valued_observed:
    l.append((col, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT))
  for col in real_valued_known:
    l.append((col, DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT))
  for col in categorical_known:
    l.append((col, DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT))
  for col in categorical_static:
    l.append((col, DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT))
  return l
