"""
Create a dataset for a given country or list of stock exchanges.    
"""
import pandas as pd
from data.wrds.fetch_wrds import (
    get_securities_selected_country,
    get_securities_selected_exchanges,
    get_company_fundamentals_selected_country,
    get_company_fundamentals_selected_exchanges,
    get_company_information_selected_country,
    get_company_information_selected_companies
)
from data.imf.fetch_imf import get_primary_commodity_prices, get_cpi_selected_country
from data.imf.constants import ALL_COMMODITIES

FUND_PRESENCE_THRESHOLD = 0.9


def create_dataset_given_country(country_code: str, time_period: tuple[str, str]) -> pd.DataFrame:
    # Fetch WRDS fundamental and info data
    global_region = country_code.lower() not in ["usa", "can"]
    fund = get_company_fundamentals_selected_country(country_code, global_region, time_period)
    tech = get_securities_selected_country(country_code, global_region, time_period)
    # Remove columns that are not present
    # Select companies
    # Fetch WRDS securities with list of companies
    # Engineer fundamental features
    # Engineer technical features
    # Fetch macro data for given country
    commodities = get_primary_commodity_prices(ALL_COMMODITIES, "M", time_period[0], time_period[1])
    cpis = get_cpi_selected_country(country_code, "M", time_period[0], time_period[1])
    # Merge frames
    # Remove nan rows
    pass


def create_dataset_given_exchanges(exchange_codes: list[str], time_period: tuple[str, str]) -> pd.DataFrame:
    # Fetch WRDS fundamental and info data

    # Remove columns that are not present
    # Select companies
    # Fetch WRDS securities with list of companies
    # Engineer fundamental features
    # Engineer technical features
    # Fetch macro data for countries of given exchanges
    # Merge frames
    # Remove nan rows
    pass


def create_dataset(
        country_code: str = None,
        exchange_codes: list[str] = None,
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
    if country_code:
        dataset = create_dataset_given_country(country_code, time_period)
        if not filename:
            filename = f"stock_data_{country_code}_{time_period[0]}-{time_period[1]}"
    elif exchange_codes:
        dataset = create_dataset_given_exchanges(exchange_codes, time_period)
        if not filename:
            filename = f"stock_data_{"-".join(exchange_codes)}_{time_period[0]}-{time_period[1]}"
    else:
        raise ValueError("Either country code or a list of exchanges must be provided.")

    print(f"Dataset created successfully, saving to {filename}...")
    dataset.to_csv(f"{filename}.csv")
    print("Dataset saved.")

    return dataset
