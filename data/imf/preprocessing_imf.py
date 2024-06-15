""" Preprocessing of macro-economic factors."""
import pandas as pd


def clean_macro(df : pd.DataFrame, macro_cols : list[str]) -> tuple[pd.DataFrame, list[str]]:
    df["datadate"] = pd.to_datetime(df["time"])
    df = df.drop(columns=["time"])
    df[macro_cols] = df[macro_cols].ffill()
    df = df.dropna(axis=1)
    macro_cols = df.drop(columns=["datadate"]).columns.values.tolist()
    return df[["datadate"] + macro_cols], macro_cols


def macro_rolling_standardization(df : pd.DataFrame, cols : list[str], window_size=6) -> pd.DataFrame:
    """ Standardization features using a rolling window to avoid future values
    'leaking' into past values. """

    rolling_means = (df[cols]
        .rolling(window=window_size, min_periods=1)
        .mean()
        .reset_index(drop=True))
    rolling_std = (df[cols]
        .rolling(window=window_size, min_periods=1)
        .std()
        .reset_index(drop=True))
    rolling_std[rolling_std == 0] = 1

    df[cols] = (df[cols] - rolling_means) / rolling_std

    df[cols] = df[cols].ffill()

    return df
