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

def process_technical_df(
    tech_df: pd.DataFrame,
    info_df: pd.DataFrame,
    global_region: bool,
) -> pd.DataFrame:
    """Clean security data from wrds."""

    tech_df["datadate"] = pd.to_datetime(tech_df["datadate"])
    tech_df = tech_df.sort_values(by=["gvkey", "datadate"])

    # Use only rows with the primary issue of the security to assure consistent pricing.
    primary_issue_colname = "prirow" if global_region else "priusa"  # prican for Canada
    primary_issues = info_df[["gvkey", primary_issue_colname]]
    tech_df = tech_df.merge(primary_issues, how="left", on="gvkey")
    tech_df = tech_df[tech_df["iid"] == tech_df[primary_issue_colname]].drop(
        columns=[primary_issue_colname]
    )

    # Drop rows without price on primary issue.
    tech_df = tech_df.dropna(subset=["prccd"])

    return tech_df


def process_fund_df(fund_df: pd.DataFrame, fund_cols: list[str]) -> pd.DataFrame:
    fund_df["datadate"] = pd.to_datetime(fund_df["datadate"])
    fund_df = fund_df.sort_values(by=["gvkey", "datadate"])
    fund_df = rolling_normalization(fund_df, fund_cols)
    return fund_df


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
