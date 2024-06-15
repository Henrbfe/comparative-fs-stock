#from tft import libs
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
from data_formatters.stock import StockFormatter
import pandas as pd
import time
# from libs.tft_model import TemporalFusionTransformer
from libs.hyperparam_opt import HyperparamOptManager
from libs import utils
from model.tft.libs.tft_model import TemporalFusionTransformer


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


def optimize_params(fixed_params: dict, opt_manager: HyperparamOptManager, train: pd.DataFrame, valid: pd.DataFrame):
    """ Initilized parameter optimization

    Args:
        fixed_params (dict): Fixed parameters
        opt_manager (HyperparamOptManager): Optimization manager
        train (pd.DataFrame): Train set
        valid (pd.DataFrame): Validation set

    Returns:
        TemporalFusionTransformer: model with best parameters
    """
    num_repeats = 1
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

model_folder = './tft_test/'    
data_csv_path = "./data/datasets/DEU_2019-01-01_2023-01-01_dataset_v1.5.csv"
pred_col = 'return_1d'
cols=['gvkey','datadate',"sharpe_1w","momentum_1d","momentum_1w","momentum_2w","momentum_4w","rsi","d_oscillator_4w","d_oscillator_6w",pred_col]
raw_data = pd.read_csv(data_csv_path, usecols=cols)
raw_data.dropna(inplace=True)
df = get_date_info(raw_data)
print(df['gvkey'].value_counts(dropna=False).nlargest(20))
#gvkey_list = df['gvkey'].value_counts()[df['gvkey'].value_counts() > 500].index.tolist()
gvkey_list=[100764,318659,203162,279220,223089,284944,282580,204820,295079,15576]
df = df.loc[df['gvkey'].isin(gvkey_list)]
df[pred_col] = df[pred_col]+1
df['region']='GER'

data_formatter = StockFormatter() 
(formatted_train, formatted_valid, formatted_test), (train,valid,test) = data_formatter.split_data(df)
fixed_params = data_formatter.get_experiment_params()

params = data_formatter.get_default_model_params()
params['model_folder']=model_folder


opt_manager = HyperparamOptManager({k: [params[k]] for k in params},
                                     fixed_params, model_folder, override_w_fixed_params=False)

now = time.time()
# params.update(fixed_params)
# params.update({ 'total_time_steps': 75 + 5,
#                 'num_encoder_steps': 75,
#                 'num_epochs': 2})
# model = TemporalFusionTransformer(fixed_params).load(model_folder)

# model = TemporalFusionTransformer(params)

model,params = optimize_params(fixed_params, opt_manager, formatted_train, formatted_valid)
print("Finished hyperparameter tuning after {} seconds".format(round(time.time()-now)))


now = time.time()
model.fit(formatted_train, formatted_valid)
print("Finished fitting after {} seconds".format(round(time.time()-now)))
output_map = model.predict(formatted_test, return_targets=True)

targets = data_formatter.format_predictions(output_map["targets"])
p10_forecast = data_formatter.format_predictions(output_map["p10"])
p50_forecast = data_formatter.format_predictions(output_map["p50"])
p90_forecast = data_formatter.format_predictions(output_map["p90"])

def create_sub_plots(train:pd.DataFrame, valid:pd.DataFrame, test:pd.DataFrame, pred:pd.DataFrame, p10_forecast:pd.DataFrame,p90_forecast:pd.DataFrame,cumsum=False,plots_per_graph=12):
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
    num_rows = plots_per_graph // plots_per_graph + (plots_per_graph % num_cols > 0)  # Number of rows in the subplot grid
    col_name = pred_col
    
    for num in range(num_plots):
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 6*num_rows))
        for i, key in enumerate(gvkey_list):
            row = i // num_cols
            col = i % num_cols
            
            temp_train = train.loc[train['gvkey']==key].copy()
            temp_valid = valid.loc[valid['gvkey']==key].copy()
            temp_test = test.loc[test['gvkey']==key].copy()
            temp_pred = pred.loc[pred['identifier']==key].copy()
            temp_p10_forecast = p10_forecast.loc[p10_forecast['identifier']==key].copy()
            temp_p90_forecast = p90_forecast.loc[p90_forecast['identifier']==key].copy()
            
            if cumsum:
                temp_train[col_name] = temp_train[col_name].cumprod() 
                temp_valid[col_name] = temp_valid[col_name].cumprod() *temp_train[col_name].iloc[-1]
                temp_test[col_name] = temp_test[col_name].cumprod() *temp_valid[col_name].iloc[-1]
                temp_pred['t+0'] = temp_pred['t+0'].cumprod() * temp_test[col_name].iloc[params['num_encoder_steps']]

            ax =  axes[i] if num_rows < 2 else axes[row, col]
            ax.plot(temp_train['datadate'], temp_train[col_name], color='blue', label='train')
            ax.plot(temp_valid['datadate'], temp_valid[col_name], color='red', label='validation')
            ax.plot(temp_test['datadate'], temp_test[col_name], color='orange', label='test')
            
            ax.plot(temp_p10_forecast['forecast_time'], temp_p10_forecast['t+0']*temp_pred['t+0'] if cumsum else temp_p10_forecast['t+0'], color='grey', label='test p10') # TODO: how to solve cumsum?
            ax.plot(temp_p90_forecast['forecast_time'], temp_p90_forecast['t+0']*temp_pred['t+0'] if cumsum else temp_p90_forecast['t+0'], color='darkgrey', label='test p90') # TODO: how to solve cumsum?
            ax.plot(temp_pred['forecast_time'], temp_pred['t+0'], color='green', label='test prediction')
            # ax.fill_between(x = temp_p10_forecast['forecast_time'], y1= temp_p10_forecast, y2=temp_p90_forecast, color='green', alpha=0.2)

            ax.set_title('Transformer Prediction')
            ax.set_xlabel('Date')
            ax.set_ylabel('Return')
            ax.legend()
            ax.grid(True)
        
        plt.tight_layout()
        #plt.show()
        plt.savefig(f'subplot{num+1}.png')
        plt.clf()

    
def create_index_plot(train:pd.DataFrame, validation:pd.DataFrame, test:pd.DataFrame, pred:pd.DataFrame, cumsum = True):
    col_name = pred_col
    train_avg_return = train.groupby(by='datadate').mean(numeric_only=True).reset_index()
    validation_avg_return = validation.groupby(by='datadate').mean(numeric_only=True).reset_index()
    test_avg_return = test.groupby(by='datadate').mean(numeric_only=True).reset_index()
    pred_avg_return = pred.groupby(by='forecast_time').mean(numeric_only=True).reset_index()
    if cumsum:
        train_avg_return[col_name] = train_avg_return[col_name].cumprod() 
        validation_avg_return[col_name] = validation_avg_return[col_name].cumprod() *train_avg_return[col_name].iloc[-1]
        test_avg_return[col_name] = test_avg_return[col_name].cumprod() *validation_avg_return[col_name].iloc[-1]
        pred_avg_return['t+0'] = pred_avg_return['t+0'].cumprod() * test_avg_return[col_name].iloc[params['num_encoder_steps']]
    plt.plot(train_avg_return['datadate'], train_avg_return[col_name], color='blue', label='train')
    plt.plot(validation_avg_return['datadate'], validation_avg_return[col_name], color='red', label='validation')
    plt.plot(test_avg_return['datadate'], test_avg_return[col_name], color='orange', label='test')
    plt.plot(pred_avg_return['forecast_time'], pred_avg_return['t+0'], color='green', label='test prediction')
    plt.legend()
    plt.grid()
    plt.savefig('index_plot.png')
    plt.clf()
    #plt.show()
    
def plot_simulation(dates, graphs, title, show, save_name):
        """Method for plotting simulation

        Args:
            dates: list of dates
            graphs: dictionary with (name, color): list of data,
            buy_sell_percentage: What percentage that is bought and sold. Used in title
            show_graph: Whether or not to show graph.
            save_name_for_graph: Name to store the graph as. False to not store the graph.
        """
        for label, values in graphs.items():
            plt.plot(dates, values, label=label[0], color=label[1], linestyle=label[2])
        plt.legend()
        plt.ylabel('Returns')
        plt.xlabel('Dates')
        plt.title(title)
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        if save_name: 
            plt.savefig(save_name)
        if show:
            plt.show()
        plt.clf()


def simulate_test_data_tft(tft: TemporalFusionTransformer, x_test=None, y_test=None, buy_sell_percentage=0.05, rebalance=False, show_graph=True, save_name_for_graph=False):
    def _select_top_n(group, n):
        return group.head(n)
    def _select_bottom_n(group, n):
        return group.tail(n)
    pred = p50_forecast
    dates = pred['forecast_time'].unique()
    sorted_pred = pred.sort_values(by=['forecast_time', 't+0'], ascending=[True, False]).groupby('forecast_time')
    trade_counts = sorted_pred['t+0'].count().apply(lambda x: max(int(x * buy_sell_percentage), 1))    
    longs = sorted_pred.apply(lambda g: _select_top_n(g, trade_counts.loc[g.name]))
    shorts = sorted_pred.apply(lambda g: _select_bottom_n(g, trade_counts.loc[g.name]))
    targets.rename(columns={'t+0': "actual"},inplace=True)
    longs.drop(columns=["forecast_time",'t+1','t+2','t+3','t+4'], inplace=True)
    shorts.drop(columns=["forecast_time",'t+1','t+2','t+3','t+4'], inplace=True)
    longs = longs.merge(targets[['actual', "forecast_time", 'identifier']], on=["forecast_time", 'identifier'], how='left')
    shorts = shorts.merge(targets[['actual', "forecast_time", 'identifier']], on=["forecast_time", 'identifier'], how='left')
    long_returns = longs.groupby("forecast_time").mean()['actual']
    short_returns = shorts.groupby("forecast_time").mean()['actual']

    algo_returns = long_returns - short_returns+1
    mean_returns = targets.groupby('forecast_time').mean()['actual']
    
    if rebalance: 
        cumsum_algo = np.cumsum(algo_returns-1)+1
        cumsum_mean = np.cumsum(mean_returns-1)+1
        cumsum_long = np.cumsum(long_returns-1)+1
        cumsum_short = np.cumsum(short_returns-1)+1
    else:
        cumsum_algo = np.cumprod(algo_returns)
        cumsum_mean = np.cumprod(mean_returns)
        cumsum_long = np.cumprod(long_returns)
        cumsum_short = np.cumprod(short_returns)
        
    total_num_trades = trade_counts.sum()
    
    plot_simulation(dates, # NB: The plot shows the returns one time period ahead. I.e. the
                                # date in the plot is based on the input data, but the y is the achieved 
                                # return from that input, which is in reality achieved one time period later
        {
            ("Long/Short", "blue", "solid"): cumsum_algo,
            ("Long", "lightgreen", "solid"): cumsum_long,
            ("Short", "lightcoral", "solid"): cumsum_short,
            ("Index", "orange", "solid"): cumsum_mean,

         },
        f'{model._get_custom_name()}: Returns on buy/sell {buy_sell_percentage*100}%',
        show=True, save_name=None)
    
    
    return cumsum_mean, mean_returns, cumsum_algo, algo_returns, cumsum_long, long_returns, cumsum_short, short_returns, dates, total_num_trades


simulate_test_data_tft(model, buy_sell_percentage=0.1)

# create_sub_plots(train, valid, test, p50_forecast, p10_forecast, p90_forecast, cumsum = True)
# create_index_plot(train, valid, test, p50_forecast)

# TODO:
"""
# WARNING:tensorflow:`evaluate()` received a value for `sample_weight`, but `weighted_metrics` were not provided.  Did you mean to pass metrics to `weighted_metrics` in `compile()`?  If this is intentional you can pass `weighted_metrics=[]` to `compile()` in order to silence this warning."""
# https://discuss.tensorflow.org/t/metric-on-tensor-flow/12547