"""Contains method for training the tft with the given data set."""

import time
import pandas as pd
from model.tft.data_formatters.stock import StockFormatter, format_inputs
from model.tft.libs.hyperparam_opt import HyperparamOptManager
from model.tft.methods import (
    create_index_plot,
    create_sub_plots,
    get_date_info,
    optimize_params,
)


def run_tft_training(
    pred_col: str, model_folder: str, data_csv_path: str, cols: list[str], nrows : int
):
    raw_data = pd.read_csv(data_csv_path, usecols=cols, nrows=nrows)
    raw_data['month'] = pd.to_datetime(raw_data["datadate"]).dt.month
    raw_data.dropna(inplace=True)
    if pred_col == "return_1w":
        raw_data = raw_data.loc[raw_data["datadate"].dt.day_of_week == 3]

    df = get_date_info(raw_data, date_col="datadate")

    gvkey_list = (
        df["gvkey"].value_counts()[df["gvkey"].value_counts() > 500].index.tolist()
    )

    df = df.loc[df["gvkey"].isin(gvkey_list)]
    df[pred_col] = df[pred_col] + 1
    df["region"] = "GER" # TODO: Change

    # ----------- #TODO: endre her ----------
    col_def = format_inputs(
        id="gvkey",
        time="datadate",
        target=pred_col,
        real_valued_observed=[
            'prccd',
            'momentum_1d',
            'momentum_1w',
            'momentum_2w',
            'momentum_4w',
            'd_oscillator_4w',
            'd_oscillator_6w',
            'rsi',
            'cshtrd',
            'cshoc',

            'PBANSOP',
            'PDAP',
            'PVANPENT',
            'PLOGORE',
            'PHIDE',
            'PMILK',
            'PSHRI',
            'PTEAINDIA',
            'PSOFT',
            'PCOCO',
            'PSAWMAL',
            'PCOBA',
            'PSOIL',
            'PWHEAMT',
            'PWOOLC',
            'POLVOIL',
            'PTOMATO',
            'PGASO',
            'PCHANA',
            'PTIMB',
            'PTEAMOM',
            'PCOPP',
            'PGNUTS',
            'PHARD',
            'PENTM',
            'PSEAF',
            'PSAWORE',
            'PRUBB',
            'PTEASL',
            'PVOIL',
            'PAPPLE',
            'PSILVER',
            'PCOALAU',
            'PGOLD',
            'PPROPANE',
            'POILWTI',
            'PWOOL',
            'PTIN',
            'PORANG',
            'PROIL',
            'PNICK',
            'PNGASUS',
            'PFERT',
            'PCHROM',
            'PBEEF',
            'PCOIL',
            'PLEAD',
            'PCOALSA',
            'LMICS',
            'PLAMB',
            'PNGAS',
            'PLMMODY',
            'PSORG',
            'PCOTTIND',
            'PNGASEU',
            'PSILLUMP',
            'PPORK',
            'PSMEA',
            'PRICENPQ',
            'PFSHMEAL',
            'PBEVE',
            'PALUM',
            'PPOULT',
            'PMAIZMT',
            'PLOGSK',
            'PCOAL',
            'PSOYB',
            'POILDUB',
            'PCERE',
            'PFOOD',
            'PMEAT',
            'PRAWM',
            'PWOOLF',
            'PCOFFOTM',
            'PBARL',
            'PAGRI',
            'PPOTASH',
            'PPALLA',
            'PFANDB',
            'PCOFFROB',
            'PREODOM',
            'PPOIL',
            'PNGASJP',
            'PUREA',
            'PCOFF',
            'PEXGALL',
            'PSUGAUSA',
            'PTEA',
            'PALLFNF',
            'PMANGELE',
            'PHEATOIL',
            'POILBRE',
            'PINDU',
            'PSUNO',
            'PPLAT',
            'PSUGA',
            'PURAN',
            'PPMETA',
            'PZINC',
            'PSALM',
            'PNRG',
            'PLITH',
            'PMETA',
            'POATS',
            'PNFUEL',
            'PALLMETA',
            'PEXGMETA',
            'PIORECR',
            'PSUGAISA',
            'POILAPSP',
            'PCPI_IX',
            'PCPI_PC_CP_A_PT',
            'PCPI_PC_PP_PT',
            'f_score',
            'opex_gr1',
            'opex_gr3',
            'intrinsic_value',
        ],
        real_valued_known=[],
        categorical_known=[
            'month' # Had to include at least one known real of cat value to avoid tf.stack index out of range error
        ],
        categorical_static=[
            'gsector_10',
            'gsector_35',
            'gsector_50',
            'gsubind_30',
            'gsubind_20',
            'gsubind_50',
            'gind_30',
            'ggroup_10',
            'gind_20',
            'ggroup_20',
            'gsector_25',
            'gsubind_40',
            'gsubind_15',
            'gsubind_80',
            'gsector_40',
            'gsector_55',
            'gsector_20',
            'gsector_45',
            'ggroup_30',
            'gind_10',
            'region'
        ],
    )
    # --------------------------------------

    data_formatter = StockFormatter(column_definition=col_def)
    (formatted_train, formatted_valid, formatted_test), (train, valid, test) = (
        data_formatter.split_data(df)
    )
    
    fixed_params = data_formatter.get_experiment_params()

    params = data_formatter.get_default_model_params()
    params["model_folder"] = model_folder

    opt_manager = HyperparamOptManager(
        {k: [params[k]] for k in params},
        fixed_params,
        model_folder,
        override_w_fixed_params=False,
    )

    now = time.time()

    model, params = optimize_params(
        fixed_params, opt_manager, formatted_train, formatted_valid
    )
    print(
        "Finished hyperparameter tuning after {} seconds".format(
            round(time.time() - now)
        )
    )

    now = time.time()
    model.fit(formatted_train, formatted_valid)
    print("Finished fitting after {} seconds".format(round(time.time() - now)))
    output_map = model.predict(formatted_test, return_targets=True)

    targets = data_formatter.format_predictions(output_map["targets"])
    p10_forecast = data_formatter.format_predictions(output_map["p10"])
    p50_forecast = data_formatter.format_predictions(output_map["p50"])
    p90_forecast = data_formatter.format_predictions(output_map["p90"])

    create_sub_plots(
        train=train,
        valid=valid,
        test=test,
        pred=p50_forecast,
        p10_forecast=p10_forecast,
        p90_forecast=p90_forecast,
        gvkey_list=gvkey_list,
        pred_col=pred_col,
        num_encoder_step_size=params["num_encoder_steps"],
        cumsum=True,
    )
    create_index_plot(
        train=train,
        validation=valid,
        test=test,
        pred=p50_forecast,
        pred_col=pred_col,
        num_encoder_step_size=params["num_encoder_steps"],
        cumsum=True,
    )