"""
Utility functions used for calculating fundamental ratios and variables.
"""

import pandas as pd


def periodic_percentage_growth(df : pd.DataFrame, col : str, window: int):
    return (
        df
        .groupby("gvkey")
        .rolling(window=window)
        [col]
        .apply(lambda vals: vals.iloc[-1] / vals.iloc[0])
        .reset_index(drop=True)
    ) - 1


def periodic_diff(df : pd.DataFrame, col : str, window: int):
    return (
        df
        .groupby("gvkey")
        .rolling(window=window)
        [col]
        .apply(lambda vals : vals.iloc[-1] - vals.iloc[0])
        .reset_index(drop=True)
    )


def periodic_start_end_sum(df : pd.DataFrame, col : str, window: int):
    return (
        df
        .groupby("gvkey")
        [col]
        .rolling(window=window)
        .apply(lambda val: val.iloc[-1] + val.iloc[0])
        .reset_index(drop=True)
    )


def calculate_turnover(df : pd.DataFrame, col : str, scale_col : str):
    return df[scale_col] / (periodic_start_end_sum(df, col, 5) / 2)


def calculate_ap_turnover(df : pd.DataFrame):
    return (
        df["cogs"] +
        periodic_diff(df, "inv", window=5) /
        (periodic_start_end_sum(df, "ap", window=5) / 2)
    )


def calculate_days_outstanding(df : pd.DataFrame, col : str, scale_col : str):
    return 365 / calculate_turnover(df, col, scale_col)


def calculate_rolling_std(df : pd.DataFrame, col : str, window: int):
    return (
        df
        .groupby("gvkey")
        .rolling(window=window)
        [col]
        .std()
        .reset_index(drop=True)
    )


def calculate_time_shifted_values(df : pd.DataFrame, col : str, shift: int):
    return df.groupby("gvkey")[col].shift(shift)


def remove_insufficent_cols(df: pd.DataFrame, cols: set[str], thresh=0.05) -> set[str]:
    """Remove columns that have more than 5% nan values."""
    nan_counts = df[list(cols)].isna().sum()
    insufficient_cols = nan_counts[nan_counts >= int(thresh * df.shape[0])].index.values
    df = df.drop(columns=insufficient_cols)
    return cols.difference(insufficient_cols)
