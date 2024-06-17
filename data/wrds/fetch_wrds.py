"""
Fetch and structure technical, fundamental, and descriptive data from WRDS.
"""

import os
from pathlib import Path
import wrds
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
# Set base path for save files to be working directory of this file
BASE_PATH = Path(__file__).parent.resolve()

db = wrds.Connection(wrds_username=os.getenv("wrds_username"))

# WRDS tables
SECD = "comp_global_daily.g_secd"
FUNDQ = "comp_global_daily.g_fundq"
COMP = "comp_global.g_company"
GICCD = "comp_global.r_giccd"
FACTOR = "contrib_global_factor.global_factor"
EXCHG = "comp_global.r_ex_codes"

SECD_NA = "comp_na_daily_all.secd"
FUNDQ_NA = "comp_na_daily_all.fundq"
COMP_NA = "comp_na_daily_all.company"
EXCHG_NA = "comp_na_daily_all.r_ex_codes"


def sql_to_df(sql: str, convert_numerics=True):
    """Run sql query on wrds database and convert the result to pandas dataframe.

    Args:
        sql (str): The query string to run.

    Returns:
        pd.DataFrame: A dataframe with the query results.
    """
    with db.engine.begin() as conn:
        result = conn.execute(sql)
    df = pd.DataFrame(result)
    if convert_numerics:
        numeric_cols = df.select_dtypes("number").columns.values
        for col in numeric_cols:
            df[col] = df[col].apply(float)
    return df


def fetch_data_selected_country(
        country_code : str, table : str, date_range : tuple[str, str], save_file : str
    ) -> pd.DataFrame:
    """Fetch wrds data for given table and country code using sql."""

    save_path = f"{BASE_PATH}/saves/{save_file}"

    if not os.path.exists(save_path):
        result =  sql_to_df(
            "select * " +
            f"from {table} " +
            f"where (fic = '{country_code}') "
            f"and datadate between '{date_range[0]}' and '{date_range[1]}';"
        )
        if not os.path.exists(f"{BASE_PATH}/saves/"):
            os.mkdir(f"{BASE_PATH}/saves/")
        result.to_csv(save_path)

    return pd.read_csv(save_path, dtype={"gvkey" : "O"}, index_col=0)


def fetch_data_selected_exchanges(
        exchange_codes : tuple[int], table : str, date_range : tuple[str, str], save_file : str
) -> pd.DataFrame:
    """Fetch wrds data for companies offered on the given exchange code using sql."""

    save_path = f"{BASE_PATH}/saves/{save_file}"

    if not os.path.exists(save_path):
        result = sql_to_df(
            "select * " +
            f"from {table} " +
            f"where (exchg in {exchange_codes} " +
            f"and datadate between '{date_range[0]}' and '{date_range[1]}';"
        )
        if not os.path.exists(f"{BASE_PATH}/saves/"):
            os.mkdir(f"{BASE_PATH}/saves/")
        result.to_csv(save_path)
    return pd.read_csv(save_path, dtype={"gvkey" : "O"}, index_col=0)


# ------ Technical ------

def get_securities_selected_country(
        country_code : str, global_region: bool, date_range: tuple[str, str], companies: list[str]
    ) -> pd.DataFrame:
    """ Get quarterly fundamental data for all companies belonging to
    the given country code. """

    save_file = f"sec_{country_code}_{date_range[0]}_{date_range[1]}.csv"
    table = SECD if global_region else SECD_NA

    df = fetch_data_selected_country(
        country_code, table, date_range, save_file
    )
    return df[df["gvkey"].isin(companies)]

def get_securities_selected_exchanges(
    exchange_codes : list[str], filename : str, global_region : bool, date_range : tuple[str, str]
) -> pd.DataFrame:
    """ Get quarterly fundamental data for all companies belonging to
    the given country code. """

    save_file = f"sec_{filename}_{date_range[0]}_{date_range[1]}.csv"
    table = SECD if global_region else SECD_NA

    return fetch_data_selected_exchanges(
        exchange_codes, table, date_range, save_file
    )


# ------ Fundamental ------

def get_company_fundamentals_selected_country(
        country_code : str, global_region, date_range
    ) -> pd.DataFrame:
    """ Get quarterly fundamental data for all companies belonging to
    the given country code."""

    save_file = f"fund_{country_code}_{date_range[0]}_{date_range[1]}.csv"
    table = FUNDQ if global_region else FUNDQ_NA

    return fetch_data_selected_country(
        country_code, table, date_range, save_file
    )

def get_company_fundamentals_selected_exchanges(
        exchange_codes : list[str], filename : str, global_region : bool, date_range : tuple[str, str]
    ) -> pd.DataFrame:
    """ Get quarterly fundamental data for all companies belonging to
    the given country code. """

    save_file = f"fund_{filename}_{date_range[0]}_{date_range[1]}.csv"
    table = FUNDQ if global_region else FUNDQ_NA

    return fetch_data_selected_exchanges(
        exchange_codes, table, date_range, save_file
    )


# ------ Descriptive ------

def get_company_information_selected_country(
        country_code : str, global_region
    ) -> pd.DataFrame:
    """ Get quarterly company information for all companies belonging to
    the given country code. """

    table = COMP if global_region else COMP_NA

    return sql_to_df(
        "select * " +
        f"from {table} " +
        f"where (fic = '{country_code}');",
        False
    )

def get_company_information_selected_companies(
        companies : tuple[str], global_region
    ) -> pd.DataFrame:
    """ Get quarterly company information for all companies belonging to
    the given country code. """

    table = COMP if global_region else COMP_NA

    return sql_to_df(
        "select * " +
        f"from {table} " +
        f"where (gvkey in {companies});"
    )


# ------ Misc ------

def get_giccd_code_descriptions():
    """ Maps Global Industry Classification (GIC) codes to their
    respective descriptions."""
    return sql_to_df(
        f"select * from {GICCD};"
    ).set_index("giccd")


def get_available_exchanges():
    """Get all available exchanges from compustat."""

    global_exchanges = sql_to_df(
        f"select * from {EXCHG};"
    )
    na_exchanges = sql_to_df(
        f"select * from {EXCHG_NA};"
    )

    return pd.concat([global_exchanges, na_exchanges])


def get_company_information_all_countries() -> pd.DataFrame:
    """ Get company information for all companies in the database, both global and north-american. """

    global_companies = sql_to_df(
        f"select * from {COMP};"
    )
    na_companies = sql_to_df(
        f"select * from {COMP_NA};"
    )
    return pd.concat([global_companies, na_companies])
