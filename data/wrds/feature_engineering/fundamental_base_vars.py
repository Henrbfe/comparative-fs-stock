"""
Code for setting the base variables used in calculating fundamental ratios, growths etc.
The naming and definitions are based on the replication crisis paper:
https://www.nber.org/system/files/working_papers/w28432/w28432.pdf
"""

import pandas as pd


def set_fundamental_base_variables(df : pd.DataFrame):
    """ Contains renaming and calculations of the base variables used in
    fundamental factor calculations. """

    set_income_statement_vars(df)
    set_cash_flow_vars(df)
    set_balance_sheet_vars(df)
    set_accruals_vars(df)


def set_income_statement_vars(df):
    """ Set base variables for values from the income statement. """

    df["sale"] = df["saley"].fillna(df["revty"])
    df["cogs"] = df["cogsy"]
    if "gpy" in df.columns.values: # gpy not in NA for some reason
        df["gp"] = df["gpy"].fillna(df["sale"] - df["cogs"])
    else:
        df["gp"] = df["sale"] - df["cogs"]
    df["xsga"] = df["xsgay"]
    # xad N/A
    # xrd N/A
    # xlr N/A
    #df["spi"] = df["spiy"]
    df["opex"] = df["xopry"]
    df["ebitda"] = df["oibdpy"]
    df["dp"] = df["dpy"]
    df["ebit"] = df["oiadpy"]
    df["int"] = df["xinty"]
    df["op"] = df["ebitda"] # can include + xrd if available
    df["ope"] = df["ebitda"] - df["int"]
    df["pi"] = df["piy"]
    df["tax"] = df["txty"]
    #df["xido"] = df["xiy"]
    #df["nri"] = df["spi"] #+ df["xido"]
    df["ni"] = df["iby"]
    df["nix"] = df["ni"] #+ df["xido"]
    df["fi"] = df["nix"] + df["int"]
    # dvc N/A
    #df["div"] = df["dvty"].fillna(df["dvy"]) if "dvty" in df.columns.values else df["dvy"]
    df["ni_qtr"] = df["ibq"]
    df["sale_qtr"] = df["saleq"]


def set_cash_flow_vars(df):
    """ Set base variables for values from cash flow statement. """

    df["capx"] = df["capxy"]
    df["capx_sale"] = df["capx"] / df["sale"]
    df["ocf"] = df["oancfy"]
    df["fcf"] = df["ocf"] - df["capx"]
    if "purtshry" not in df.columns.values:
        df["eqbb"] = df["prstkcy"]
    else:
        df["eqbb"] = df["prstkcy"] + df["purtshry"]
    #df["eqis"] = df["sstky"]
    #df["eqnetis"] = df["eqis"] - df["eqbb"]
    #df["eqpo"] = df["div"] + df["eqbb"]
    #df["eqnpo"] = df["div"] - df["eqnetis"]
    #df["dltnetis"] = df["dltisy"] - df["dltry"]
    df["dstnetis"] = df["dlcchy"].fillna(df["dlcq"]) # check if values are comparable
    #df["dbnetis"] = df["dltnetis"] + df["dstnetis"]
    #df["netis"] = df["eqnetis"] + df["dbnetis"]

    df["fincf"] = (df["fincfy"]
        #.fillna(
            #df["netis"] - df["dvy"] + df["fiaoy"] + df["txbcof"])
        .fillna(0)
    )


def set_balance_sheet_vars(df):
    """ Set base variables for values from the balance sheet statement. """

    df["at"] = df["atq"]
    df["ca"] = df["actq"]
    df["nca"] = df["at"] - df["ca"]
    df["rec"] = df["rectq"]
    df["cash"] = df["cheq"]
    df["inv"] = df["invtq"]
    df["intan"] = df["intanq"]
    df["ivao"] = df["ivaoq"]
    df["ivst"] = df["ivstq"]
    #df["ppeg"] = df["ppegtq"] N/A
    df["ppen"] = df["ppentq"]

    df["lt"] = df["ltq"]
    df["cl"] = df["lctq"]
    df["ncl"] = df["lt"] - df["cl"]
    df["ap"] = df["apq"]
    df["debtst"] = df["dlcq"]
    #df["txp"] = df["txpq"] N/A
    df["debtlt"] = df["dlttq"]
    #df["txditc"] = df["txditcq"]

    df["pstk"] = df["pstkq"]
    df["debt"] = df["dlttq"] + df["dlcq"]
    df["netdebt"] = df["debt"] - df["cash"]
    df["seq"] = df["seqq"]
    df["be"] = df["seq"] + 0 - df["pstk"] # 0 since df["txditc"] is missing
    #df["bev"] = df["icapt"] + df["dlcq"] - df["cash"] icapt N/A
    df["bev"] = df["seq"] + df["netdebt"] + df["mibq"]

    df["nwc"] = df["ca"] - df["cl"]
    df["coa"] = df["ca"] - df["cash"]
    df["col"] = df["cl"] - df["dlcq"].fillna(0)
    df["cowc"] = df["coa"] - df["col"]
    df["ncoa"] = df["at"] - df["coa"] - df["ivao"]
    df["ncol"] = df["lt"] - df["cl"] - df["dlttq"]
    df["nncoa"] = df["ncoa"] - df["ncol"]
    df["fna"] = (df["ivst"] + df["ivao"]).fillna(0)
    df["fnl"] = df["debt"] + df["pstk"].fillna(0)
    df["nfna"] = df["fna"] - df["fnl"]
    df["oa"] = df["coa"] + df["ncoa"]
    df["ol"] = df["col"] + df["ncol"]
    df["noa"] = df["oa"] - df["ol"]
    df["lnoa"] = df["ppentq"] + df["intan"] + df["aoq"] - df["loq"] + df["dp"]
    df["caliq"] = (df["ca"] - df["inv"]).fillna(df["cash"] + df["rectq"])
    #df["ppeinv"] = df["ppegty"] + df["inv"]

    # Ortiz-Molina and Phillips Liquidity
    df["aliq"] = (df["cash"]
                  + 0.75 * df["coa"]
                  + 0.5 * (df["at"] - df["ca"] - df["intan"].fillna(0)))


def set_accruals_vars(df):
    """ Set base variables for accrual accounting values. """

    df["oacc"] = df["ni"] - df["oancfy"]
    df["tacc"] = df["oacc"] + df["nfna"] # TODO yearly change of nfna
    if "wcapty" not in df.columns.values:
        df["wcapty"] = 0
    df["ocf"] = (df["oancfy"]
                 .fillna(df["ni"] - df["oacc"])
                 .fillna(df["ni"] + df["dp"] - df["wcapty"])
                 .fillna(0))
    df["ocf_qtr"] = df["ibq"] + df["dpq"]
    df["cop"] = df["ebitda"] - df["oacc"] # xrd N/A
