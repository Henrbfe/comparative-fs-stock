"""
Create a dataset for a given country or list of stock exchanges.    
"""
import pandas as pd
from data.imf.preprocessing_imf import process_macro_df
from data.wrds.feature_engineering.fundamental import calculate_all_fundamental_factors
from data.wrds.feature_engineering.technical import calculate_all_technical_indicators
from data.wrds.fetch_wrds import (
    get_securities_selected_country,
    get_company_fundamentals_selected_country,
    get_company_information_selected_country,
)
from data.imf.fetch_imf import get_primary_commodity_prices, get_cpi_selected_country
from data.wrds.preprocessing_wrds import process_fund_df, process_info_df, process_technical_df

FUND_PRESENCE_THRESHOLD = 0.9


def create_dataset_given_country(country_code: str, time_period: tuple[str, str]) -> pd.DataFrame:
    """Fetch each category of data for companies in the given country within the given time period.

    Args:
        country_code (str): 3-alpha code of the country.
        time_period (tuple[str, str]): tuple of start and end date.

    Returns:
        pd.DataFrame: merged dataframe with all feature categories.
    """
    global_region = country_code.lower() not in ["usa", "can"]

    # Fundamental data
    fund_df = get_company_fundamentals_selected_country(country_code, global_region, time_period)
    fund_df, fund_features, companies = calculate_all_fundamental_factors(fund_df)
    fund_df = process_fund_df(fund_df, fund_features)

    # Descriptive data
    desc_df = get_company_information_selected_country(country_code, global_region)
    desc_df = process_info_df(desc_df)

    # Technical data
    tech_df = get_securities_selected_country(country_code, global_region, time_period, companies)
    tech_df, tech_features = calculate_all_technical_indicators(tech_df, f"{country_code}_{time_period[0]}_{time_period[1]}")
    tech_df = process_technical_df(tech_df, desc_df, global_region)

    # Macro-economic data
    commodities = get_primary_commodity_prices("M", time_period[0], time_period[1])
    cpis = get_cpi_selected_country(country_code, "M", time_period[0], time_period[1])
    macro_df = commodities.merge(cpis, on="time")
    macro_features = macro_df.drop(columns=["time"]).columns.values.tolist()
    macro_df, macro_features = process_macro_df(macro_df, macro_features)

    # Merge sets
    merged_df = merge_datasets(fund_df, tech_df, desc_df, macro_df, fund_features, tech_features, [], macro_features)
    merged_df = merged_df.dropna()

    return merged_df


def merge_datasets(
    fund_df: pd.DataFrame,
    sec_df: pd.DataFrame,
    info_df: pd.DataFrame,
    macro_df: pd.DataFrame,
    fund_cols: list[str],
    tech_cols: list[str],
    info_cols: list[str],
    macro_cols: list[str],
) -> pd.DataFrame:
    """Merging together datasets with fundamental, security and general information data.
    Quarterly fundamental data is forward filled to use for daily data points."""

    info_df = info_df[["gvkey"] + info_cols]
    fund_df = fund_df[["gvkey", "datadate"] + fund_cols]
    sec_df = sec_df[["gvkey", "datadate"] + tech_cols]
    macro_df = macro_df[["datadate"] + macro_cols]

    # Merge securiy (technical) data with company information and fill company information for all rows
    merged_df = sec_df.merge(info_df, how="left", on="gvkey")
    merged_df[info_cols] = (
        merged_df.groupby("gvkey")[info_cols].ffill().bfill().reset_index(drop=True)
    )

    # Merge security (technical) data and information with fundamental and fill in fundamental data
    # for days in-between quarterly updates
    merged_df = merged_df.merge(
        fund_df, how="left", on=["gvkey", "datadate"]
    ).sort_values(by=["gvkey", "datadate"])
    merged_df[fund_cols] = (
        merged_df.groupby("gvkey")[fund_cols].ffill().reset_index(drop=True)
    )

    # Merge in macro data and fill for days between monthly updates
    merged_df = merged_df.merge(macro_df, how="left", on="datadate")
    merged_df[macro_cols] = merged_df[macro_cols].ffill()

    return merged_df


def create_dataset(
        country_code: str,
        time_period: tuple[str, str] = ("2023-01-01", "2024-01-01"),
        filename: str = None
    ):
    """Function for creating a stock dataset.

    Args:
        country_code (str, optional): Fetch data from a given country, set to None to use list of exchanges. Defaults to None.
        exchange_codes (list[str], optional): Fetch data from a given list of exchanges, only used if country code is None. Defaults to None.
        time_period (tuple[str, str], optional): Select the desired time period for the data. Defaults to ("2023-01-01", "2024-01-01").
        filename (str, optional): The filename in which to store the created dataset. Defaults to None.

    Raises:
        ValueError: If both country code and exchange codes are set to None.

    Returns:
        pd.DataFrame: A pandas dataframe of the created dataset.
    """
    dataset = create_dataset_given_country(country_code, time_period)
    if not filename:
        filename = f"stock_data_{country_code}_{time_period[0]}-{time_period[1]}"
    print(f"Dataset created successfully, saving to {filename}...")
    dataset.to_csv(f"{filename}.csv")
    print("Dataset saved.")

    return dataset
