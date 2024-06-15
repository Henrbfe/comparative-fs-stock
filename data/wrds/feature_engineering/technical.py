""" Contains functions to derive statistical indicators such as
momentum, RSI, sharpe ratio, and more. """

import os
from pathlib import Path
import pandas as pd


BASE_PATH = Path(__file__).parent.resolve()


def derive_rolling_return(df: pd.DataFrame, n_periods: int):
    """Determines return for each datadate based on the price change
    since the earliest datadate in the given time period."""

    return (
        df.groupby("gvkey")["prccd"]
        .pct_change(periods=n_periods)
        .shift(-n_periods)
        .reset_index(drop=True)
        .fillna(0)
    )


def derive_sharpe_ratio(
    df: pd.DataFrame, time_period: int, risk_free_return: float = 0
):
    """Determine sharpe ratio for each datadate given price change within time period
    and the risk free return."""
    if not "momentum_1w" in df.columns:
        df["momentum_1w"] = derive_momentum(df, 5)

    df["momentum_1w_a"] = df["momentum_1w"] - risk_free_return

    return (
        df["momentum_1w_a"]
        / df.groupby("gvkey")["momentum_1w_a"]
        .rolling(window=time_period, min_periods=1)
        .std()
        .reset_index(drop=True)
    ).fillna(0) / 100


def derive_momentum(df: pd.DataFrame, periods: int):
    """Determines rate of change of the stock price momentum."""
    return (
        df.groupby("gvkey")["prccd"]
        .pct_change(periods=periods)
        .reset_index(drop=True)
        .fillna(0)
    )


def derive_relative_strength_index(df: pd.DataFrame, alpha=1 / 14):
    """Calculates RSI based on exponentially moving average gain, without considering loss,
    and the ex.mov. average loss, without considering gain."""

    if "momentum_1d" not in df.columns:
        df["momentum_1d"] = (
            df.groupby("gvkey")["prccd"]
            .pct_change(periods=1)
            .reset_index(drop=True)
            .fillna(0)
        )

    df["gain"] = df["momentum_1d"].apply(lambda x: x if x > 0 else 0)
    df["loss"] = df["momentum_1d"].apply(lambda x: abs(x) if x < 0 else 0)

    df["ewm_gain"] = (
        df.groupby("gvkey")["gain"].ewm(alpha=alpha).mean().reset_index(drop=True)
    )
    df["ewm_loss"] = (
        df.groupby("gvkey")["loss"].ewm(alpha=alpha).mean().reset_index(drop=True)
    )

    rsi = 100 - 100 / (1 + df["ewm_gain"] / df["ewm_loss"])

    df.drop(columns=["gain", "loss", "ewm_gain", "ewm_loss"], inplace=True)

    return rsi.fillna(100) / 100


def stochastic_oscillator_k(df: pd.DataFrame, time_period: int):
    """Calculates the %K stochastic oscillator in the given time period."""
    return 100 * (
        df.groupby("gvkey")
        .rolling(window=time_period, min_periods=2)["prccd"]
        .apply(lambda x: (x[-1] - min(x)) / (max(x) - min(x)), raw=True)
        .reset_index(drop=True)
        .fillna(0)
    )


def stochastic_oscillator_d(
    df: pd.DataFrame, time_period: int, k_osc="k_oscillator_1w"
):
    """Calculates the %D stochastic oscillator as a moving average of stochastic %K."""
    return (
        df.groupby("gvkey")
        .rolling(time_period, min_periods=2)[k_osc]
        .mean()
        .reset_index(drop=True)
        .fillna(0)
    ) / 100


def calculate_all_technical_indicators(
    df: pd.DataFrame, filename: str, use_saved_result=True
) -> pd.DataFrame:
    """Performs all calculations for technical indicators given daily price data."""

    save_path_df = f"{BASE_PATH}/saves/technical_" + f"{filename}.csv"

    # Load previous results if available
    if os.path.exists(save_path_df) and use_saved_result:
        df = pd.read_csv(save_path_df, dtype={"gvkey": "O"}, index_col=0)
        df["datadate"] = pd.to_datetime(df["datadate"])
        return df

    df = df.sort_values(by=["gvkey", "datadate"])

    df["return_1d"] = derive_rolling_return(df, 1)
    df["return_1w"] = derive_rolling_return(df, 5)
    df["sharpe_1w"] = derive_sharpe_ratio(df, 5)
    df["momentum_1d"] = derive_momentum(df, 1)
    df["momentum_1w"] = derive_momentum(df, 5)
    df["momentum_2w"] = derive_momentum(df, 10)
    df["momentum_4w"] = derive_momentum(df, 20)
    df["rsi"] = derive_relative_strength_index(df)
    df["k_oscillator_1w"] = stochastic_oscillator_k(df, 5)
    df["d_oscillator_4w"] = stochastic_oscillator_d(df, 20)
    df["k_oscillator_2w"] = stochastic_oscillator_k(df, 10)
    df["d_oscillator_6w"] = stochastic_oscillator_d(df, 30, "k_oscillator_2w")

    sec_cols = [
        "prccd",
        "cshoc",
        "cshtrd",
        "return_1d",
        "return_1w",
        "momentum_1d",
        "momentum_1w",
        "momentum_2w",
        "momentum_4w",
        "rsi",
        "d_oscillator_4w",
        "d_oscillator_6w",
    ]

    df.to_csv(save_path_df)

    return df, sec_cols


