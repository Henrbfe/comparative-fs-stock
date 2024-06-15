from typing import Optional
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from data.constants import DATE_COL


def load_data_from_csv(
    dataset_filename: str,
    nrows: Optional[int],
    features: Optional[list[str]],
    target: str,
    random_sample=False,
) -> tuple[pd.DataFrame, list[str]]:
    """Load a stock market dataset from a csv-file with the given filename.

    Args:
        dataset_filename (str): Filename of csv to read data from.
        nrows (Optional[int]): The number of rows to load, None if all rows should be read.
        features (Optional[list[str]]): List of features to include, or None if using all features.
        target (str): The name of the target variable.
        random_sample (bool, optional): If nrows is not None, whether to choose top rows or a random sample. Defaults to False.

    Returns:
        tuple[pd.DataFrame, list[str]]: The loaded dataframe and a list of the features included.
    """

    read_nrows = None if random_sample else nrows

    if features:
        cols = ["gvkey", "datadate", target] + features
        df = pd.read_csv(
            dataset_filename, nrows=nrows, usecols=cols, header=0
        )
    else:
        df = pd.read_csv(dataset_filename, nrows=read_nrows, header=0)
        features = [
            col
            for col in df.columns.values
            if col
            not in ["gvkey", "datadate", "return_1d", "return_1w", "conm"]
        ]

    if random_sample:
        df = df.sample(nrows)

    df[features] = df[features].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(axis=1, thresh=int(len(df) * 0.7))
    df = df.dropna()
    features = [feature for feature in features if feature in df.columns.values]

    df = df[(df[target] < 1.3) & (df[target] > 0.7)]


    # Sort on date to spread companies between train and test set
    df["datadate"] = pd.to_datetime(df["datadate"])
    df = df.sort_values(by="datadate")
    df['gvkey'] = df['gvkey']+df['datadate'].dt.day_of_week # To get weekly data for TFT

    print("loaded rows: ", len(df))

    return df[["gvkey", "datadate"] + features + [target]], features


def date_based_split(
    df,
    target_column,
    test_size=0.2,
    validation_size=0.2,
    show=False,
    save_name=None,
    day_of_week=-1,
):
    """Splits dataframe into training, validation and test based on test and validation size.
    Args:
        df: dataset
        target_column: target to predict
        test_size: size of test set. Defaults to 0.2.
        validation_size: size of validation set. If 0, no validation set. Defaults to 0.2.
    Returns:
        x_train, y_train, x_validation, y_validation, x_test, y_test
    """
    df = df.sort_values(by=[DATE_COL], ascending=False)
    test_split_index = int(len(df) * test_size)
    df_test = df.head(test_split_index)
    df_train_validation = df.tail(len(df) - test_split_index)
    if day_of_week >= 0 and day_of_week < 5:
        df_test = df_test.loc[df_test["datadate"].dt.day_of_week == day_of_week]

    if validation_size > 0.0:
        validation_split_index = int(len(df_train_validation) * validation_size)
        df_validation = df_train_validation.head(validation_split_index)
        df_train = df_train_validation.tail(
            len(df_train_validation) - validation_split_index
        )
    else:
        df_validation = None
        df_train = df_train_validation

    x_train, y_train = df_train.drop(target_column, axis=1), df_train[target_column]
    x_test, y_test = df_test.drop(target_column, axis=1), df_test[target_column]

    if validation_size > 0.0:
        x_validation, y_validation = (
            df_validation.drop(target_column, axis=1),
            df_validation[target_column],
        )
    else:
        x_validation, y_validation = None, None

    if save_name or show:
        train_mean = (
            df_train[[DATE_COL, target_column]]
            .groupby(by=[DATE_COL], as_index=False)
            .mean()
        )
        train_mean["data set"] = "Train"
        l = [train_mean]
        if y_validation is not None:
            val_mean = (
                df_validation[[DATE_COL, target_column]]
                .groupby(by=[DATE_COL], as_index=False)
                .mean()
            )
            val_mean["data set"] = "Validation"
            l.append(val_mean)
        test_mean = (
            df_test[[DATE_COL, target_column]]
            .groupby(by=[DATE_COL], as_index=False)
            .mean()
        )
        test_mean["data set"] = "Test"
        l.append(test_mean)
        plot_df = pd.concat(l, ignore_index=True)
        if day_of_week >= 0 and day_of_week < 5:
            plot_df = plot_df.loc[plot_df["datadate"].dt.day_of_week == day_of_week]
        plot_df[target_column] = plot_df[target_column].cumprod()
        sns.lineplot(data=plot_df, x=DATE_COL, y=target_column, hue="data set")
        plt.ylabel("Cumulative return")
        plt.xticks(rotation=30)
        plt.title("Train test split")
        if save_name:
            plt.savefig(save_name)
        if show:
            plt.show()
        plt.clf()

    return x_train, y_train, x_validation, y_validation, x_test, y_test
