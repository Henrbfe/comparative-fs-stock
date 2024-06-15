"""
Code for fetching macro-economic data from IMF. Contains functions for fetching commodity
prices and consumer price indices.
"""

import os
import requests
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

BASE_URL = "http://dataservices.imf.org/REST/SDMX_JSON.svc/"
DATA_METHOD = "CompactData/"
SERIES_INFORMATION = "DataStructure/"
METADATA = "GenericMetadata"

def get_dataset_list() -> list[str]:
    url = f"{BASE_URL}Dataflow"
    res = requests.get(
        url,
        timeout=100,
    )
    data = res.json()
    res.close()
    datasets = []
    for dataset in data["Structure"]["Dataflows"]["Dataflow"]:
        datasets.append(dataset["KeyFamilyRef"]["KeyFamilyID"])
    return datasets

def get_codelist(series : str):
    url = f"{BASE_URL}DataStructure/{series}"
    res = requests.get(
        url,
        timeout=100,
    )
    data = res.json()
    res.close()
    return data["Structure"]["CodeLists"]["CodeList"]

def get_request_help():
    url = f"{BASE_URL}help"
    res = requests.get(
        url,
        timeout=100,
    )
    data = res.json()
    res.close()
    return data


def send_imf_request(
    series: str, selector : str, category_name : str, start_period='2017-01-01', end_period='2021-01-01'
) -> pd.DataFrame:
    """Perform a HTTP-request for the requested selection of data within the given time period.

    Args:
        series (str): The IMF data series to fetch from.
        selector (str): The selector describing the desired data.
        category_name (str): The name of the value category.
        start_period (str, optional): The start date of the time period. Defaults to '2017-01-01'.
        end_period (str, optional): The end date of the time period. Defaults to '2021-01-01'.

    Returns:
        pd.DataFrame: The requested data.
    """
    url = (f"{BASE_URL}{DATA_METHOD}{series}/" +
        f"{selector}?startPeriod={start_period}&endPeriod={end_period}")
    res = requests.get(
        url,
        timeout=100,
    )

    request_patience = 5

    while request_patience >= 0 and res.status_code != 200:
        print(f"Failed imf-request with code {res.status_code}, patience left : {request_patience}")
        request_patience -= 1
        res = requests.get(
            url,
            timeout=100,
        )

    indicators = res.json()["CompactData"]["DataSet"]["Series"]
    res.close()

    # entries = [
    #     {"time" : f"{obs['@TIME_PERIOD']}-01", indicators[0][category_name] : obs["@OBS_VALUE"]}
    #     for obs in indicators[0]["Obs"]]
    entries = []

    for ind in indicators:
        if "Obs" not in ind.keys():
            continue
        observations = ind["Obs"]
        indicator = ind[category_name]
        if not isinstance(observations, list):
            observations = [observations] # When only a single entry is present the format is not list
        for i, obs in enumerate(observations):
            val = None if "@OBS_VALUE" not in obs.keys() else obs["@OBS_VALUE"]
            if i >= len(entries):
                entries.append({"time" : f"{obs['@TIME_PERIOD']}-01", indicator : val})
            else:
                entries[i][indicator] = val

    return pd.DataFrame(entries)


load_dotenv()

BASE_PATH = Path(__file__).parent.resolve()

CONSUMER_PRICE_INDEX_SERIES = "CPI"
COMMODITY_PRICE_SERIES = "PCPS"
FINANCIAL_INDICATORS_SERIES = "IFS"


def get_cpi_selected_country(country_code : str, freq : str, start_period : str, end_period : str):
    selector = f"{freq}.{country_code}"

    save_path = f"{BASE_PATH}/saves/{CONSUMER_PRICE_INDEX_SERIES}_{selector}_{start_period}_{end_period}.csv"

    if os.path.exists(save_path):
        return pd.read_csv(save_path, index_col=0)

    df = send_imf_request(
        CONSUMER_PRICE_INDEX_SERIES,
        selector,
        "@INDICATOR",
        start_period,
        end_period)

    df.to_csv(save_path)
    return df

def get_primary_commodity_prices(commodities : list[str], freq : str, start_period : str, end_period : str):
    measure = "IX" # index, can use USD
    selector = f"{freq}..{'+'.join(commodities)}.{measure}"

    save_path = f"{BASE_PATH}/saves/{COMMODITY_PRICE_SERIES}_{selector}_{start_period}_{end_period}"

    if os.path.exists(save_path):
        return pd.read_csv(save_path, index_col=0)

    df = send_imf_request(COMMODITY_PRICE_SERIES, selector, "@COMMODITY", start_period, end_period)
    df.to_csv(save_path)
    return df
