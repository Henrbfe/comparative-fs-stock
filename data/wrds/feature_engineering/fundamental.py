import pandas as pd
import numpy as np
from data.wrds.feature_engineering.fundamental_base_vars import set_fundamental_base_variables
from data.wrds.feature_engineering.constants import RATIO_FACTORS, GROWTH_FACTORS, CHANGE_AT_SCALED_FACTORS, PERIODIC_CHANGE_FACTORS
from data.wrds.feature_engineering.utils import (
    calculate_days_outstanding,
    calculate_turnover,
    calculate_ap_turnover,
    calculate_time_shifted_values,
    calculate_rolling_std,
    periodic_diff,
    periodic_percentage_growth,
    remove_insufficent_cols
)
from warnings import simplefilter

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


def calculate_ratios(df: pd.DataFrame) -> set[str]:
    """Calculates ratios between different factors."""

    new_cols = set()

    for key, factors in RATIO_FACTORS.items():
        for factor in factors:
            colname = f"{factor}_{key}"
            if key == "nix":
                df[colname] = df[factor] / df[key].abs()
            else:
                df[colname] = df[factor] / df[key]
            new_cols.add(colname)

    # Special calculations

    #df["inv_days"] = calculate_days_outstanding(df, "inv", "cogs")
    #new_cols.add("inv_days")
    df["rec_days"] = calculate_days_outstanding(df, "rec", "sale")
    new_cols.add("rec_days")
    df["ap_days"] = calculate_days_outstanding(df, "ap", "cogs")
    new_cols.add("ap_days")

    df["cash_conversion"] = df["inv_days"] + df["rec_days"] - df["ap_days"]
    new_cols.add("cash_conversion")
    #df["inv_turnover"] = calculate_turnover(df, "inv", "cogs")
    #new_cols.add("inv_turnover")
    df["at_turnover"] = calculate_turnover(df, "at", "sale")
    new_cols.add("at_turnover")
    df["rec_turnover"] = calculate_turnover(df, "rec", "sale")
    new_cols.add("rec_turnover")
    df["ap_turnover"] = calculate_ap_turnover(df)
    new_cols.add("ap_turnover")

    df["niq_be"] = df["ni_qtr"] / calculate_time_shifted_values(df, "be", shift=1)
    new_cols.add("niq_be")
    df["niq_at"] = df["ni_qtr"] / calculate_time_shifted_values(df, "at", shift=1)
    new_cols.add("niq_at")

    df["capex_abn"] = df["capx_sale"] / (
        (
            calculate_time_shifted_values(df, "capx_sale", shift=5)
            + calculate_time_shifted_values(df, "capx_sale", shift=9)
            + calculate_time_shifted_values(df, "capx_sale", shift=13)
        )
        / 3
    )
    new_cols.add("capex_abn")
    df["op_atl1"] = df["op"] / calculate_time_shifted_values(df, "at", shift=5)
    new_cols.add("op_atl1")
    df["ope_bel1"] = df["ope"] / calculate_time_shifted_values(df, "be", shift=5)
    new_cols.add("ope_bel1")
    df["gp_atl1"] = df["gp"] / calculate_time_shifted_values(df, "at", shift=5)
    new_cols.add("gp_atl1")
    df["cop_atl1"] = df["cop"] / calculate_time_shifted_values(df, "at", shift=5)
    new_cols.add("cop_atl1")
    df["aliq_at"] = df["aliq"] / calculate_time_shifted_values(df, "at", shift=5)
    new_cols.add("aliq_at")

    print(new_cols)

    new_cols = remove_insufficent_cols(df, new_cols)
    print(new_cols)

    print(
        f"Nan rows after ratios: {df.shape[0] - df.dropna(subset=list(new_cols)).shape[0]}"
    )

    return new_cols

def calculate_growth_factors(df: pd.DataFrame) -> set[str]:
    """Calculate growth factors as the percentage change
    between the start and end of a given period. The periods used
    are 1 year and 3 years (365 days and 1095 days)."""

    new_cols = set()

    for years, factors in GROWTH_FACTORS.items():
        window = 4 * years + 1
        for factor in factors:
            colname = f"{factor}_gr{years}"
            df[colname] = periodic_percentage_growth(df, factor, window)
            new_cols.add(colname)

    new_cols = remove_insufficent_cols(df, new_cols)

    print(
        f"Nan rows after growth: {df.shape[0] - df.dropna(subset=list(new_cols)).shape[0]}"
    )

    return new_cols


def calculate_periodic_change_at_scaled(df: pd.DataFrame) -> set[str]:
    """Calculates the change in factor value from start to end of given period
    divided (scaled by) total assets (at)."""

    new_cols = set()

    for years, factors in CHANGE_AT_SCALED_FACTORS.items():
        window = 4 * years + 1
        for factor in factors:
            # Paper uses _gr1a ending, but _ch1a seems more consistent with naming scheme
            colname = f"{factor}_ch{years}a"
            df[colname] = periodic_diff(df, factor, window) / df["at"]
            new_cols.add(colname)

    df["lnoa_gr1a"] = periodic_diff(df, "lnoa", window=5) / periodic_diff(
        df, "at", window=5
    )
    new_cols.add("lnoa_gr1a")

    new_cols = remove_insufficent_cols(df, new_cols)

    print(
        f"Nan rows after change_at_scaled: {df.shape[0] - df.dropna(subset=list(new_cols)).shape[0]}"
    )

    return new_cols

def calculate_periodic_change(df: pd.DataFrame) -> set[str]:
    """Calculates the difference in factor value from start and end of a given period."""

    new_cols = set()

    # Key is period in years, and values are a list of factors to calculate for
    factors = {
        1: ["ocf_at", "niq_be", "niq_at"],
        5: ["gp_at", "ni_be", "ni_at", "ocf_at", "gp_sale"],
    }

    for years, factors in PERIODIC_CHANGE_FACTORS.items():
        # These are named differently in the paper, but end with ch5 or chg1
        window = 4 * years + 1
        for factor in factors:
            colname = f"{factor}_ch{years}"
            df[colname] = periodic_diff(df, factor, window)
            new_cols.add(colname)

    new_cols = remove_insufficent_cols(df, new_cols)

    print(
        f"Nan rows after change: {df.shape[0] - df.dropna(subset=list(new_cols)).shape[0]}"
    )

    return new_cols


def calculate_volatility_factors(df: pd.DataFrame) -> set[str]:
    """Calculates volatility using a rolling standard deviation."""

    new_cols = set()

    df["niq_saleq_std"] = calculate_rolling_std(df, "ni_qtr_sale_qtr", window=9)
    new_cols.add("niq_saleq_std")
    df["roeq_be_std"] = calculate_rolling_std(df, "ni_qtr_be", window=17)
    new_cols.add("roeq_be_std")
    df["roe_be_std"] = calculate_rolling_std(df, "ni_be", window=21)
    new_cols.add("roe_be_std")
    df["ocfq_saleq_std"] = calculate_rolling_std(df, "ocf_qtr_sale_qtr", window=17)
    new_cols.add("ocfq_saleq_std")

    new_cols = remove_insufficent_cols(df, new_cols)

    print(
        f"Nan rows after volatility factors: {df.shape[0] - df.dropna(subset=list(new_cols)).shape[0]}"
    )

    return new_cols



def calculate_all_fundamental_factors(
    df: pd.DataFrame
) -> tuple[pd.DataFrame, list[str]]:
    """Performs all calculations of fundamental factors. As some calculations depend on
    previously calculated factor values, the order of the functions matters."""

    new_cols = set()
    set_fundamental_base_variables(df)
    new_cols.update(calculate_ratios(df))
    new_cols.update(calculate_growth_factors(df))
    new_cols.update(calculate_periodic_change_at_scaled(df))
    new_cols.update(calculate_periodic_change(df))
    new_cols.update(calculate_volatility_factors(df))

    new_cols = list(new_cols)

    df[new_cols] = (
        df[["gvkey"] + new_cols]
        .replace([np.inf, -np.inf], np.nan)
        .groupby("gvkey")
        .ffill()[new_cols]
    )
    df = df.dropna(subset=new_cols).reset_index(drop=True)

    # Avoid de-fragmentation
    result_df = df[["gvkey", "datadate"] + new_cols].copy()

    print(new_cols)

    return result_df, list(new_cols)

