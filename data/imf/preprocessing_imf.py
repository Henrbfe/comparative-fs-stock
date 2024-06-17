""" Preprocessing of macro-economic factors."""
import pandas as pd


def process_macro_df(df : pd.DataFrame, macro_cols : list[str]) -> tuple[pd.DataFrame, list[str]]:
    df["datadate"] = pd.to_datetime(df["time"])
    df = df.drop(columns=["time"])
    for col in macro_cols:
        df[col] = df[col].apply(float)
    # df[macro_cols] = df[macro_cols].apply(float)
    df[macro_cols] = df[macro_cols].ffill()
    df = df.dropna(axis=1)
    macro_cols = df.drop(columns=["datadate"]).columns.values.tolist()
    df = macro_rolling_standardization(df, macro_cols)
    return df[["datadate"] + macro_cols], macro_cols


def macro_rolling_standardization(df : pd.DataFrame, cols : list[str]) -> pd.DataFrame:
    """ Standardization features using a rolling window to avoid future values
    'leaking' into past values. """

    rolling_means = (
        df[cols]
        .expanding(min_periods=1)
        .mean()
        .reset_index(drop=True)
    )
    rolling_std = (
        df[cols]
        .expanding(min_periods=1)
        .std()
        .fillna(0) # The first occurence for each company (gvkey)
        .reset_index(drop=True)
    )
    rolling_std[rolling_std == 0] = 1

    df[cols] = (df[cols] - rolling_means) / rolling_std

    df[cols] = df[cols].ffill()

    return df
