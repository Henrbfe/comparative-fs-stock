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


def calculate_ratios(df: pd.DataFrame, excluded_cols: list[str]) -> set[str]:
    """Calculates ratios between different factors."""

    new_cols = set()

    for key, factors in RATIO_FACTORS.items():
        if key in excluded_cols:
            continue
        for factor in factors:
            if factor in excluded_cols:
                continue
            colname = f"{factor}_{key}"
            if key == "nix":
                df[colname] = df[factor] / df[key].abs()
            else:
                df[colname] = df[factor] / df[key]
            new_cols.add(colname)

    # Special calculations
    use_cash_conv = True
    if "inv" not in excluded_cols and "cogs" not in excluded_cols:
        df["inv_days"] = calculate_days_outstanding(df, "inv", "cogs")
        new_cols.add("inv_days")
        df["inv_turnover"] = calculate_turnover(df, "inv", "cogs")
        new_cols.add("inv_turnover")
        if "ap" not in excluded_cols:
            df["ap_turnover"] = calculate_ap_turnover(df)
            new_cols.add("ap_turnover")
    else:
        use_cash_conv = False
    if "rec" not in excluded_cols and "sale" not in excluded_cols:
        df["rec_days"] = calculate_days_outstanding(df, "rec", "sale")
        new_cols.add("rec_days")
        df["rec_turnover"] = calculate_turnover(df, "rec", "sale")
        new_cols.add("rec_turnover")
    else:
        use_cash_conv = False
    if "ap" not in excluded_cols and "cogs" not in excluded_cols:
        df["ap_days"] = calculate_days_outstanding(df, "ap", "cogs")
        new_cols.add("ap_days")
    else:
        use_cash_conv = False
    if use_cash_conv:
        df["cash_conversion"] = df["inv_days"] + df["rec_days"] - df["ap_days"]
        new_cols.add("cash_conversion")

    if "at" not in excluded_cols and "sale" not in excluded_cols:
        df["at_turnover"] = calculate_turnover(df, "at", "sale")
        new_cols.add("at_turnover")

    if "ni_qtr" not in excluded_cols and "be" not in excluded_cols:
        df["niq_be"] = df["ni_qtr"] / calculate_time_shifted_values(df, "be", shift=1)
        new_cols.add("niq_be")

    if "ni_qtr" not in excluded_cols and "at" not in excluded_cols:
        df["niq_at"] = df["ni_qtr"] / calculate_time_shifted_values(df, "at", shift=1)
        new_cols.add("niq_at")

    if "capx_sale" not in excluded_cols:

        df["capex_abn"] = df["capx_sale"] / (
            (
                calculate_time_shifted_values(df, "capx_sale", shift=5)
                + calculate_time_shifted_values(df, "capx_sale", shift=9)
                + calculate_time_shifted_values(df, "capx_sale", shift=13)
            )
            / 3
        )
        new_cols.add("capex_abn")
    if "op" not in excluded_cols and "at" not in excluded_cols:
        df["op_atl1"] = df["op"] / calculate_time_shifted_values(df, "at", shift=5)
        new_cols.add("op_atl1")
    if "ope" not in excluded_cols and "be" not in excluded_cols:
        df["ope_bel1"] = df["ope"] / calculate_time_shifted_values(df, "be", shift=5)
        new_cols.add("ope_bel1")
    if "gp" not in excluded_cols and "at" not in excluded_cols:
        df["gp_atl1"] = df["gp"] / calculate_time_shifted_values(df, "at", shift=5)
        new_cols.add("gp_atl1")
    if "cop" not in excluded_cols and "at" not in excluded_cols:
        df["cop_atl1"] = df["cop"] / calculate_time_shifted_values(df, "at", shift=5)
        new_cols.add("cop_atl1")
    if "aliq" not in excluded_cols and "at" not in excluded_cols:
        df["aliq_at"] = df["aliq"] / calculate_time_shifted_values(df, "at", shift=5)
        new_cols.add("aliq_at")

    new_cols = remove_insufficent_cols(df, new_cols)

    return new_cols

def calculate_growth_factors(df: pd.DataFrame, excluded_cols: list[str]) -> set[str]:
    """Calculate growth factors as the percentage change
    between the start and end of a given period. The periods used
    are 1 year and 3 years (365 days and 1095 days)."""

    new_cols = set()

    for years, factors in GROWTH_FACTORS.items():
        window = 4 * years + 1
        for factor in factors:
            if factor in excluded_cols:
                continue
            colname = f"{factor}_gr{years}"
            df[colname] = periodic_percentage_growth(df, factor, window)
            new_cols.add(colname)

    new_cols = remove_insufficent_cols(df, new_cols)

    return new_cols


def calculate_periodic_change_at_scaled(df: pd.DataFrame, excluded_cols: list[str]) -> set[str]:
    """Calculates the change in factor value from start to end of given period
    divided (scaled by) total assets (at)."""

    new_cols = set()

    for years, factors in CHANGE_AT_SCALED_FACTORS.items():
        window = 4 * years + 1
        for factor in factors:
            if factor in excluded_cols:
                continue
            # Paper uses _gr1a ending, but _ch1a seems more consistent with naming scheme
            colname = f"{factor}_ch{years}a"
            df[colname] = periodic_diff(df, factor, window) / df["at"]
            new_cols.add(colname)

    if "lnoa" not in excluded_cols and "at" not in excluded_cols:
        df["lnoa_gr1a"] = periodic_diff(df, "lnoa", window=5) / periodic_diff(
            df, "at", window=5
        )
        new_cols.add("lnoa_gr1a")

    new_cols = remove_insufficent_cols(df, new_cols)

    return new_cols

def calculate_periodic_change(df: pd.DataFrame, excluded_cols: list[str]) -> set[str]:
    """Calculates the difference in factor value from start and end of a given period."""

    new_cols = set()

    for years, factors in PERIODIC_CHANGE_FACTORS.items():
        # These are named differently in the paper, but end with ch5 or chg1
        window = 4 * years + 1
        for factor in factors:
            if factor in excluded_cols or factor not in df.columns.values:
                continue
            colname = f"{factor}_ch{years}"
            df[colname] = periodic_diff(df, factor, window)
            new_cols.add(colname)

    new_cols = remove_insufficent_cols(df, new_cols)

    return new_cols


def calculate_volatility_factors(df: pd.DataFrame, excluded_cols: list[str]) -> set[str]:
    """Calculates volatility using a rolling standard deviation."""

    new_cols = set()

    if "ni_qtr" not in excluded_cols and "sale_qtr" not in excluded_cols:
        df["niq_saleq_std"] = calculate_rolling_std(df, "ni_qtr_sale_qtr", window=9)
        new_cols.add("niq_saleq_std")
    
    if "ni_qtr" not in excluded_cols and "be" not in excluded_cols:
        df["roeq_be_std"] = calculate_rolling_std(df, "ni_qtr_be", window=17)
        new_cols.add("roeq_be_std")
    if "ni" not in excluded_cols and "be" not in excluded_cols:
        df["roe_be_std"] = calculate_rolling_std(df, "ni_be", window=21)
        new_cols.add("roe_be_std")
    if "ocf_qtr" not in excluded_cols and "sale_qtr" not in excluded_cols:
        df["ocfq_saleq_std"] = calculate_rolling_std(df, "ocf_qtr_sale_qtr", window=17)
        new_cols.add("ocfq_saleq_std")

    new_cols = remove_insufficent_cols(df, new_cols)

    return new_cols


def nan_base_cols(df: pd.DataFrame, base_cols: list[str]) -> list[str]:
    """Identify columns to exclude based on nan entries.

    Args:
        df (pd.DataFrame): Fundamental data
        base_cols (list[str]): List of columns to evaluate for exclusion.

    Returns:
        list[str]: Columns to be excluded
    """
    nan_portions = df[base_cols].isna().sum() / df.shape[0]
    return nan_portions[nan_portions > 0.5].index.values


def filter_companies(df: pd.DataFrame, fund_cols: list[str]) -> list[str]:
    df["qrt"] = pd.to_datetime(df["datadate"]).dt.quarter
    quarter_counts = df[["gvkey", "qrt"]].groupby("gvkey")["qrt"].nunique()
    presence_counts = df[~df[fund_cols].isna().any(axis=1)].groupby("gvkey")["gvkey"].count()
    quarter_counts = quarter_counts[presence_counts.index.values]
    presence_portion = presence_counts / quarter_counts
    return presence_portion[presence_portion > 0.9].index.values


def calculate_all_fundamental_factors(
    df: pd.DataFrame
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Select fundamental columns and companies based on data presence. Then calculates
    fundamental features.

    Args:
        df (pd.DataFrame): Fundamental data

    Returns:
        tuple[pd.DataFrame, list[str], list[str]]:
            Dataframe with fundamental features, list of features, list of companies included.
    """

    new_cols = set()
    base_cols = set_fundamental_base_variables(df)
    excluded_cols = list(nan_base_cols(df, base_cols))
    base_cols = [col for col in base_cols if col not in excluded_cols]
    companies = filter_companies(df, base_cols)
    df = df[df["gvkey"].isin(companies)].reset_index(drop=True)
    new_cols.update(calculate_ratios(df, excluded_cols))
    new_cols.update(calculate_growth_factors(df, excluded_cols))
    new_cols.update(calculate_periodic_change_at_scaled(df, excluded_cols))
    new_cols.update(calculate_periodic_change(df, excluded_cols))
    new_cols.update(calculate_volatility_factors(df, excluded_cols))

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

    return result_df, list(new_cols), companies
