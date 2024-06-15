import pandas as pd

COMPANY_INFO_COLS = [
    "conm",
    "prirow",
    "priusa",
    "prican",
    "gsector",
    "ggroup",
    "gind",
    "gsubind",
]

def process_info_df(info_df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess data for the company information dataset."""
    info_df = info_df[["gvkey"] + COMPANY_INFO_COLS]
    info_df.loc[:, "ggroup"] = info_df["ggroup"].str[2:]
    info_df.loc[:, "gind"] = info_df["gind"].str[4:]
    info_df.loc[:, "gsubind"] = info_df["gsubind"].str[6:]
    info_df = pd.get_dummies(
        info_df, columns=["gsector", "ggroup", "gind", "gsubind"], dtype="float32"
    )
    return info_df

def rolling_normalization(
    df: pd.DataFrame, cols: list[str]
) -> pd.DataFrame:
    """Standardization features using a rolling window to avoid future values
    'leaking' into past values."""

    rolling_means = (
        df.groupby("gvkey")[cols]
        .expanding(min_periods=1)
        .mean()
        .reset_index(drop=True)
    )
    rolling_std = (
        df.groupby("gvkey")[cols]
        .expanding(min_periods=1)
        .std()
        .fillna(0) # The first occurence for each company (gvkey)
        .reset_index(drop=True)
    )

    rolling_std[rolling_std == 0] = 1

    df[cols] = (df[cols] - rolling_means) / rolling_std

    return df
