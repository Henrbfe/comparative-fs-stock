
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from model.tft.libs.tft_model import TemporalFusionTransformer
from model.tft.script_hyperparam_opt import HyperparamOptManager


def get_date_info(df: pd.DataFrame, date_col:str = 'datadate'):
    """Extracts date information from DataFrame date column

    Args:
        df (_type_): _description_
        date_col (str, optional): _description_. Defaults to 'datadate'.

    Returns:
        _type_: _description_
    """
    df[date_col] = pd.to_datetime(df[date_col])
    df['day_of_week'] = df[date_col].dt.dayofweek
    df['day_of_month'] = df[date_col].dt.day
    df['week_of_year'] = df[date_col].dt.isocalendar().week
    df['month'] = df[date_col].dt.month
    df['year'] = df[date_col].dt.year
    return df

def optimize_params(fixed_params: dict, opt_manager: HyperparamOptManager, train: pd.DataFrame, valid: pd.DataFrame, num_repeats=1):
    """ Initilized parameter optimization

    Args:
        fixed_params (dict): Fixed parameters
        opt_manager (HyperparamOptManager): Optimization manager
        train (pd.DataFrame): Train set
        valid (pd.DataFrame): Validation set

    Returns:
        TemporalFusionTransformer: model with best parameters
    """
    best_loss = np.Inf
    for i in range(num_repeats):
        if num_repeats > 1: print(f"Optimizing epoch: {i+1}/{num_repeats+1}")
        new_params = opt_manager.get_next_parameters()
        new_params.update(fixed_params)
        new_params.update({ 'total_time_steps': 75 + 5,
                'num_encoder_steps': 75,
                'num_epochs': 1})
        model = TemporalFusionTransformer(new_params)
        if not model.training_data_cached():
            model.cache_batched_data(train, "train", num_samples=100)
            model.cache_batched_data(valid, "valid", num_samples=10)
        model.fit()
        val_loss = model.evaluate()

        if val_loss < best_loss:
            opt_manager.update_score(new_params, val_loss, model)
            best_loss = val_loss
    best_params = opt_manager.get_best_params()
    print("Best parameters: ", best_params)
    model = TemporalFusionTransformer(best_params)
    model.load(opt_manager.hyperparam_folder, use_keras_loadings=False)
    return model, best_params

def create_sub_plots(train:pd.DataFrame, valid:pd.DataFrame, test:pd.DataFrame, pred:pd.DataFrame, p10_forecast:pd.DataFrame,p90_forecast:pd.DataFrame,gvkey_list:list[int], pred_col:str,num_encoder_step_size:int,cumsum=False,plots_per_graph=12):
    """Create one plot per gvkey 

    Args:
        train (pd.DataFrame): unformatted training set
        valid (pd.DataFrame): unformatted validation set
        test (pd.DataFrame): unformatted test set
        pred (pd.DataFrame): prediction set
        p10_forecast (pd.DataFrame): p10 forecast
        p90_forecast (pd.DataFrame): p90 forecast
        cumsum (bool): Whether or not to cumulatively add. Defaults to False.
    """
    num_plots = len(gvkey_list) // plots_per_graph + (len(gvkey_list) % plots_per_graph>0)
    num_cols = 3 # Number of columns in the subplot grid
    num_rows = plots_per_graph // num_cols + (plots_per_graph % num_cols > 0)  # Number of rows in the subplot grid
    
    for num in range(num_plots):
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 6*num_rows))
        for i, key in enumerate(gvkey_list[plots_per_graph*num:min(plots_per_graph*(num+1), len(gvkey_list))]):
            try:
                row = i // num_cols
                col = i % num_cols
                
                temp_train = train.loc[train['gvkey']==key].copy()
                temp_valid = valid.loc[valid['gvkey']==key].copy()
                temp_test = test.loc[test['gvkey']==key].copy()
                temp_pred = pred.loc[pred['identifier']==key].copy()
                temp_p10_forecast = p10_forecast.loc[p10_forecast['identifier']==key].copy()
                temp_p90_forecast = p90_forecast.loc[p90_forecast['identifier']==key].copy()
                
                if cumsum:
                    temp_train[pred_col] = temp_train[pred_col].cumprod() 
                    temp_valid[pred_col] = temp_valid[pred_col].cumprod() *temp_train[pred_col].iloc[-1]
                    temp_test[pred_col] = temp_test[pred_col].cumprod() *temp_valid[pred_col].iloc[-1]
                    temp_pred['t+0'] = temp_pred['t+0'].cumprod() * temp_test[pred_col].iloc[num_encoder_step_size]#params['num_encoder_steps']
        
                ax =  axes[i] if num_rows < 2 else axes[row, col]
                ax.plot(temp_train['datadate'], temp_train[pred_col], color='blue', label='train')
                ax.plot(temp_valid['datadate'], temp_valid[pred_col], color='red', label='validation')
                ax.plot(temp_test['datadate'], temp_test[pred_col], color='orange', label='test')
                
                ax.plot(temp_p10_forecast['forecast_time'], temp_p10_forecast['t+0']*temp_pred['t+0'] if cumsum else temp_p10_forecast['t+0'], color='grey', label='test p10') # TODO: how to solve cumsum?
                ax.plot(temp_p90_forecast['forecast_time'], temp_p90_forecast['t+0']*temp_pred['t+0'] if cumsum else temp_p90_forecast['t+0'], color='darkgrey', label='test p90') # TODO: how to solve cumsum?
                ax.plot(temp_pred['forecast_time'], temp_pred['t+0'], color='green', label='test prediction')
                # ax.fill_between(x = temp_p10_forecast['forecast_time'], y1= temp_p10_forecast, y2=temp_p90_forecast, color='green', alpha=0.2)

                ax.set_title('Transformer Prediction')
                ax.set_xlabel('Date')
                ax.set_ylabel('Return')
                ax.legend()
                ax.grid(True)
            except Exception as e:
                print("Failed: ", e)
        plt.tight_layout()
        #plt.show()
        plt.savefig(f'subplot{num+1}.png')
        plt.clf()

    
def create_index_plot(train:pd.DataFrame, validation:pd.DataFrame, test:pd.DataFrame, pred:pd.DataFrame,pred_col:str,num_encoder_step_size:int, cumsum = True):
    train_avg_return = train.groupby(by='datadate').mean(numeric_only=True).reset_index()
    validation_avg_return = validation.groupby(by='datadate').mean(numeric_only=True).reset_index()
    test_avg_return = test.groupby(by='datadate').mean(numeric_only=True).reset_index()
    pred_avg_return = pred.groupby(by='forecast_time').mean(numeric_only=True).reset_index()
    if cumsum:
        train_avg_return[pred_col] = train_avg_return[pred_col].cumprod() 
        validation_avg_return[pred_col] = validation_avg_return[pred_col].cumprod() *train_avg_return[pred_col].iloc[-1]
        test_avg_return[pred_col] = test_avg_return[pred_col].cumprod() *validation_avg_return[pred_col].iloc[-1]
        pred_avg_return['t+0'] = pred_avg_return['t+0'].cumprod() * test_avg_return[pred_col].iloc[num_encoder_step_size]
    plt.plot(train_avg_return['datadate'], train_avg_return[pred_col], color='blue', label='train')
    plt.plot(validation_avg_return['datadate'], validation_avg_return[pred_col], color='red', label='validation')
    plt.plot(test_avg_return['datadate'], test_avg_return[pred_col], color='orange', label='test')
    plt.plot(pred_avg_return['forecast_time'], pred_avg_return['t+0'], color='green', label='test prediction')
    plt.legend()
    plt.grid()
    plt.savefig('index_plot.png')
    plt.clf()
    #plt.show()