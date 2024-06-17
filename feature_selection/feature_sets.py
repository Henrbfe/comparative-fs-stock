"""
Overview of feature selected subsets
"""

from data.constants import (
    TECHNICAL_FEATURES,
    FUNDAMENTAL_FEATURES,
    MACRO_FEATURES,
    SECTOR_FEATURES,
)


# NASNOR dataset

NASNOR_VAR_FILTER = [
        'ni_ch1a', 'nix_sale', 'oacc_nix', 'ocf_sale', 'ocf_be', 'eqbb_ch3a', 'ca_cl',
        'nwc_at', 'fincf_ch3a', 'nix_ch3a', 'ebitda_ch1a', 'gp_ch3a',
        'ocf_ch1a', 'cop_at', 'pi_sale', 'ocf_cl', 'op_at', 'gp_sale_ch5',
        'ebitda_sale', 'nix_be', 'ni_qtr_sale_qtr', 'cl_lt', 'sale_be', 'pi_nix',
        'prccd', 'ebit_sale', 'ni_qtr_be', 'fincf_at', 'PCPIHO_PC_PP_PT', 'opex_at',
        'eqbb_at', 'oacc_at', 'ni_sale', 'ni_ch3a', 'eqbb_ch1a', 'ebitda_at', 'gp_at',
        'PCPIT_PC_PP_PT', 'ebitda_ch3a', 'cash_at', 'gp_sale', 'ni_at', 'at_be',
        'fna_ch1a', 'cshoc', 'fna_ch3a', 'PCPIR_PC_PP_PT', 'PCPIEC_PC_PP_PT',
        'ebitda_cl', 'cash_lt', 'ni_be', 'fincf_ch1a', 'gp_ch1a', 'nix_ch1a',
        'PCPIA_PC_PP_PT', 'ocf_ch3a', 'sale_nwc', 'ocf_at', 'cash_cl'
    ]

NASNOR_MRMR_FILTER = [
    "momentum_1d", "PCPIHO_PC_PP_PT", "PBEVE", "ocf_sale", "momentum_1w", "PLOGORE",
    "PWOOLF", "PCPIM_PC_PP_PT", "gsector_10", "PSAWMAL", "ocf_cl", "momentum_2w",
    "PWOOL", "PWOOLC", "ocf_at", "gsector_40", "PHARD", "cash_at", "PCPIHO_IX",
    "PTIMB", "ocf_be", "PCOFFROB", "PCPIRE_PC_PP_PT", "PCPIT_IX", "cop_at", "PSUNO",
    "PHIDE", "cash_cl", "PCOFF", "rsi", "PMILK", "PBANSOP", "PALUM", "ni_qtr_sale_qtr",
    "PUREA", "ggroup_30", "PRAWM", "cash_lt", "PCHANA", "PPORK", "ebitda_cl",
    "gsubind_40", "PCOIL", "PLOGSK", "PMEAT", "gp_at", "PCOALSA", "PTEAINDIA", "PORANG",
]

NASNOR_FORWARD = [
    'momentum_1d', 'momentum_1w', 'PCPI_IX', 'PCPIM_PC_PP_PT', 'PCPIT_IX',
    'PCPIO_IX', 'PCOCO', 'PCOFFOTM', 'PLMMODY', 'PNGAS', 'PNICK', 'PORANG', 'PPALLA',
    'PPMETA', 'PPORK', 'PPOULT', 'PSOIL', 'PTEAMOM', 'PUREA', 'pi_sale',
    'ebitda_sale', 'ocf_sale', 'cash_at', 'gsector_35', 'gsector_60', 'gind_10',
    'gind_50', 'gsubind_50', 'gsubind_60', 'ggroup_40'
]

NASNOR_BACKWARD = [
    'momentum_1d', 'momentum_1w', 'PCPI_PC_PP_PT', 'PCPIHO_PC_PP_PT', 'PCPIM_IX',
    'PCPIM_PC_PP_PT', 'PCPIR_PC_PP_PT', 'PALLFNF', 'PBEEF', 'PBEVE', 'PGASO',
    'PHEATOIL', 'POILDUB', 'POLVOIL', 'PPOTASH', 'PSAWMAL', 'PSAWORE', 'PSEAF',
    'PSUGA', 'at_be', 'ni_qtr_sale_qtr', 'fincf_ch1a', 'nix_ch3a', 'ocf_sale',
    'gp_ch3a', 'nix_be', 'op_at', 'ebitda_ch3a', 'gsector_10', 'gsector_35'
]

NASNOR_GA = [
    'nix_ch3a', 'PHARD', 'gsubind_30', 'PSMEA', 'gsector_60', 'PCPIT_IX', 'PSOFT',
    'PTEAINDIA', 'PCPIO_PC_PP_PT', 'momentum_1w', 'PCPIHO_IX', 'PPMETA',
    'gsector_45', 'PPALLA', 'PCOTTIND', 'PSUGAISA', 'PSAWMAL', 'd_oscillator_4w',
    'PMILK', 'gind_70', 'PSEAF', 'PLOGORE', 'eqbb_ch3a', 'sale_nwc', 'POILWTI',
    'PRUBB', 'oacc_nix', 'PSUNO', 'PFANDB', 'PNICK', 'PCPIRE_PC_PP_PT', 'gind_40',
    'PSUGA', 'eqbb_at', 'oacc_at', 'nix_ch1a', 'gsector_40', 'ebitda_cl', 'PCERE',
    'PDAP', 'POILDUB', 'PPOTASH', 'gsector_10', 'cshoc', 'PCOBA', 'PCPIH_PC_PP_PT',
    'PAGRI', 'ebitda_ch1a', 'PCPIM_PC_PP_PT', 'PSUGAUSA', 'gsector_20', 'PCOFFROB',
    'PSOIL', 'gsubind_20', 'ocf_be', 'eqbb_ch1a', 'sale_be', 'PVOIL', 'PCPIRE_IX',
    'gsubind_50', 'PCPIR_IX', 'ebitda_ch3a', 'PHIDE', 'ocf_sale', 'PCPIO_IX',
    'cash_at', 'PBANSOP', 'PPOULT', 'gind_50', 'cash_cl', 'PWOOL', 'gsector_15',
    'PFERT', 'prccd', 'PSALM', 'ni_ch3a', 'gsector_35', 'pi_nix', 'rsi', 'PCOIL',
    'PNGASUS', 'PCPIED_PC_PP_PT', 'PUREA', 'gp_ch1a', 'PTIN', 'PCPIM_IX', 'PAPPLE',
    'PFOOD', 'POATS', 'momentum_4w', 'PHEATOIL', 'ni_ch1a', 'fna_ch1a',
    'ni_qtr_sale_qtr', 'PCOCO', 'PCPI_PC_PP_PT', 'ggroup_30', 'PLEAD'
]

NASNOR_PSO = [
    'PAGRI', 'momentum_1w', 'PLEAD', 'nix_ch3a', 'PPOIL', 'ebitda_cl', 'PBARL',
    'PCOPP', 'PSILVER', 'PMETA', 'PPOTASH', 'PEXGMETA', 'PALUM', 'PGOLD',
    'PCPIHO_IX', 'PALLFNF', 'ocf_ch1a', 'eqbb_at', 'PINDU', 'cash_cl', 'PTEAMOM',
    'ni_sale', 'PPOULT', 'gsector_15', 'gind_50', 'PFSHMEAL', 'ni_at', 'PSMEA',
    'PWOOL', 'PSUGAUSA', 'gsubind_50', 'gp_sale', 'ocf_cl', 'ocf_ch3a', 'prccd',
    'PGNUTS', 'PUREA', 'PPORK', 'gsector_55', 'momentum_1d', 'gind_40', 'PTEAINDIA',
    'ebitda_sale', 'eqbb_ch3a', 'PSUGA', 'PVOIL', 'ebitda_at', 'PROIL', 'op_at',
    'gsector_20', 'ggroup_40', 'oacc_nix', 'ggroup_10', 'PCPIO_IX', 'POILDUB',
    'PSAWORE', 'gsector_40', 'gsector_30', 'gsubind_60', 'PLOGSK', 'PPLAT',
    'POILBRE', 'cshoc', 'PCPIR_IX', 'gind_10', 'PCOALAU', 'gind_70', 'ocf_be',
    'gsector_10', 'PALLMETA', 'gsubind_80', 'POILWTI', 'momentum_4w', 'PCOCO',
    'fna_ch1a', 'PCHANA', 'd_oscillator_6w', 'gsubind_10', 'PBANSOP', 'gsector_50',
    'PRAWM', 'fincf_ch3a', 'gsubind_30', 'PCOFF', 'ggroup_50', 'gsubind_40',
    'PCPIM_PC_PP_PT', 'POATS', 'rsi', 'PMAIZMT', 'nix_be', 'gsector_35', 'PCPI_IX',
    'PFOOD', 'PORANG', 'PTOMATO', 'PWOOLC', 'cash_lt', 'd_oscillator_4w', 'PPALLA',
    'PCOFFROB', 'gp_sale_ch5', 'PCOALSA', 'ni_qtr_sale_qtr', 'PTEA', 'PSOYB',
    'nix_sale', 'PCPIO_PC_PP_PT', 'PCPIA_IX', 'momentum_2w', 'PSHRI', 'gsector_25'
]

NASNOR_MGO = [
    'gsector_15', 'oacc_nix', 'gsubind_20', 'PNRG', 'ni_qtr_sale_qtr', 'PSEAF',
    'eqbb_ch3a', 'PSHRI', 'PCPIEC_IX', 'fincf_ch3a', 'PWOOLF', 'PPORK', 'PWOOLC',
    'PCPI_IX', 'ocf_cl', 'momentum_1w', 'nwc_at', 'PMEAT', 'cash_at', 'ca_cl',
    'momentum_1d', 'PSMEA', 'PCOFFOTM', 'PCHANA', 'gsector_20', 'PCOTTIND',
    'gp_sale_ch5'
]

# USA dataset

USA_VAR_FILTER = [
    "momentum_2w",
    "prccd",
    "cash_cl",
    "ni_qtr_sale_qtr",
    "cshoc",
    "ebitda_at",
    "debt_at",
    "momentum_4w",
    "ocf_qtr_sale_qtr",
    "cshtrd",
    "capx_at",
    "opex_at",
    "momentum_1w",
]

# USA dataset

USA_MRMR_FILTER = [
    "gsector_35", "d_oscillator_6w", "PCPI_IX", "ebitda_at", "PMETA", "gsector_50",
    "d_oscillator_4w", "momentum_1d", "debt_at", "PPMETA", "gsector_20",
    "gsector_15", "PNFUEL", "sharpe_1w", "rsi", "opex_at", "cshoc", "ni_qtr_sale_qtr",
    "gsector_45", "gsector_55", "gsector_10", "momentum_4w", "gsector_25", "prccd",
    "gsector_60", "momentum_1w", "gsector_40", "ocf_qtr_sale_qtr", "PCPI_PC_PP_PT",
    "gsector_30", "cash_cl", "momentum_2w", "cshtrd", "POILAPSP", "capx_at", "PNRG",
]

USA_FORWARD = [
    'sharpe_1w', 'momentum_1d', 'momentum_1w', 'rsi', 'd_oscillator_4w',
    'gsector_10', 'gsector_25', 'gsector_50', 'cash_cl', 'PNFUEL'
]

USA_BACKWARD = [
    "momentum_1d", "momentum_1w", "rsi", "d_oscillator_4w", "gsector_10",
    "gsector_20", "gsector_40", "gsector_50", "PMETA", "PPMETA"
]

USA_GA = [
    "gsector_25",
    "ocf_qtr_sale_qtr",
    "PMETA",
    "d_oscillator_6w",
    "gsector_30",
    "gsector_15",
    "momentum_1w",
]

USA_PSO = [
    "ebitda_at",
    "momentum_1w",
    "ocf_qtr_sale_qtr",
    "cshtrd",
    "momentum_4w",
    "PCPI_IX",
]

USA_MGO = [
    'sharpe_1w', 'momentum_1w', 'rsi', 'd_oscillator_6w', 'gsector_10',
    'gsector_15', 'gsector_20', 'gsector_25', 'gsector_35', 'gsector_40',
    'gsector_45', 'gsector_55', 'gsector_60', 'cash_cl', 'debt_at',
    'ocf_qtr_sale_qtr', 'PNRG', 'PNFUEL', 'POILAPSP', 'PMETA'
]

# Japan dataset

JPN_VAR_FILTER = [
    'prccd', 'cshoc', 'capx_at', 'spi_at', 'nri_at', 'fi_at', 'fincf_at', 'netis_at',
    'eqnetis_at', 'eqis_at', 'dbnetis_at', 'dltnetis_at', 'dstnetis_at',
    'eqnpo_at', 'eqbb_at', 'div_at', 'tacc_at', 'nwc_at', 'debt_at', 'ni_at',
    'cash_at', 'gp_sale', 'ebitda_sale', 'ebit_sale', 'pi_sale', 'ni_sale',
    'nix_sale', 'fcf_sale', 'ocf_sale', 'ni_qtr_sale_qtr', 'ocf_qtr_sale_qtr',
    'ope_be', 'ni_be', 'nix_be', 'fcf_be', 'debtlt_be', 'debt_be', 'ni_qtr_be',
    'at_be', 'gp_bev', 'ebitda_bev', 'ebit_bev', 'fi_bev', 'cop_bev', 'be_bev',
    'debt_bev', 'cash_bev', 'debtlt_bev', 'debtst_bev', 'sale_bev', 'gp_ppen',
    'ebitda_ppen', 'fcf_ppen', 'lt_ppen', 'oacc_nix', 'tacc_nix', 'pi_nix',
    'int_debt', 'ebitda_debt', 'ocf_debt', 'debtst_debt', 'debtlt_debt',
    'int_debtlt', 'cash_cl', 'caliq_cl', 'ca_cl', 'cash_lt', 'cl_lt', 'inv_ca',
    'rec_ca', 'fcf_ocf', 'ebit_int', 'sale_nwc', 'tax_pi', 'inv_days', 'rec_days',
    'ap_days', 'cash_conversion', 'inv_turnover', 'at_turnover', 'rec_turnover',
    'ap_turnover', 'niq_be', 'niq_at', 'capex_abn', 'op_atl1', 'ope_bel1',
    'gp_atl1', 'cop_atl1', 'aliq_at', 'at_gr1', 'nca_gr1', 'lt_gr1', 'ncl_gr1',
    'be_gr1', 'debt_gr1', 'inv_gr1', 'capx_gr2', 'at_gr3', 'ca_gr3', 'nca_gr3',
    'lt_gr3', 'cl_gr3', 'ncl_gr3', 'be_gr3', 'debt_gr3', 'cogs_gr3', 'opex_gr3',
    'capx_gr3', 'sale_qtr_gr3', 'inv_gr3', 'gp_ch1a', 'ocf_ch1a', 'cash_ch1a',
    'inv_ch1a', 'rec_ch1a', 'intan_ch1a', 'debtst_ch1a', 'ap_ch1a', 'debtlt_ch1a',
    'coa_ch1a', 'col_ch1a', 'cowc_ch1a', 'ncol_ch1a', 'ol_ch1a', 'fnl_ch1a',
    'nfna_ch1a', 'ebitda_ch1a', 'ebit_ch1a', 'ope_ch1a', 'ni_ch1a', 'dp_ch1a',
    'fcf_ch1a', 'nwc_ch1a', 'nix_ch1a', 'dltnetis_ch1a', 'dstnetis_ch1a',
    'dbnetis_ch1a', 'fincf_ch1a', 'tax_ch1a', 'div_ch1a', 'eqbb_ch1a', 'eqpo_ch1a',
    'capx_ch1a', 'be_ch1a', 'gp_ch3a', 'ocf_ch3a', 'cash_ch3a', 'inv_ch3a',
    'rec_ch3a', 'intan_ch3a', 'debtst_ch3a', 'ap_ch3a', 'debtlt_ch3a', 'coa_ch3a',
    'col_ch3a', 'cowc_ch3a', 'ncol_ch3a', 'ol_ch3a', 'fnl_ch3a', 'nfna_ch3a',
    'ebitda_ch3a', 'ebit_ch3a', 'ope_ch3a', 'ni_ch3a', 'dp_ch3a', 'fcf_ch3a',
    'nwc_ch3a', 'nix_ch3a', 'dltnetis_ch3a', 'dstnetis_ch3a', 'dbnetis_ch3a',
    'fincf_ch3a', 'tax_ch3a', 'div_ch3a', 'eqbb_ch3a', 'eqpo_ch3a', 'capx_ch3a',
    'lnoa_gr1a', 'gp_at_ch5', 'ni_be_ch5', 'ocf_at_ch5', 'gp_sale_ch5',
    'niq_saleq_std', 'roeq_be_std', 'roe_be_std', 'ocfq_saleq_std', 'o_score',
    'intrinsic_value', 'PPOULT'
]

JPN_MRMR_FILTER = [
    "PLITH", "prccd", "xsga_gr3", "PTEAMOM", "PORANG", "momentum_1w",
    "pstk_bev", "PTOMATO", "PGNUTS", "PCPI_IX", "roeq_be_std", "PCOCO",
    "PTEASL", "PLOGSK", "debt_gr1", "PCHROM", "eqnetis_at", "momentum_2w",
    "PBEVE", "ni_be", "PENTM", "PPALLA", "POLVOIL", "momentum_1d",
    "PPOTASH", "eqbb_ch1a", "PTIN", "PSORG", "nca_gr1", "momentum_4w",
    "PSOFT", "tax_ch1a", "PNGASJP", "ggroup_20", "PCPI_PC_CP_A_PT",
    "PINDU", "nix_be", "roe_be_std", "PMETA", "PREODOM", "PEXGMETA",
    "ap_ch1a", "PWHEAMT", "rsi", "PRICENPQ", "capx_gr3", "ebitda_cl",
    "gsector_25", "PWOOLF", "niq_be",
]

JPN_FORWARD = [
    "sharpe_1w", "momentum_1d", "xido_at", "pstk_bev", "fna_ch1a", "fna_ch3a", "PFANDB", "PCOIL",
    "PCOFFROB", "PIORECR", "PLOGORE", "PNGASEU", "PNGASJP", "PSHRI", "PSUGAISA", "PURAN", "PWOOLF",
    "PCERE", "PSUGA", "PTIMB", "PWOOL", "PCHANA", "gsector_15", "gsector_35", "gsector_55",
    "ggroup_40", "ggroup_50", "gsubind_50", "gsubind_80", "gsubind_25"
]

JPN_BACKWARD = [
    'momentum_1w', 'momentum_4w', 'nri_at', 'fincf_at', 'eqis_at', 'oacc_at',
    'gp_sale', 'pi_sale', 'ocf_be', 'gp_bev', 'ebit_bev', 'cash_bev', 'sale_bev',
    'oacc_nix', 'int_debtlt', 'inv_turnover', 'cl_gr1', 'cl_gr3', 'ncl_gr3',
    'opex_gr3', 'ap_ch1a', 'fnl_ch1a', 'ni_ch1a', 'dstnetis_ch3a', 'niq_be_ch1',
    'ocfq_saleq_std', 'PNRG', 'PURAN', 'PSILLUMP', 'ggroup_20'
]

JPN_GA = [
    'cash_lt', 'momentum_2w', 'PCERE', 'div_ch3a', 'at_turnover', 'ca_gr1',
    'PBANSOP', 'PVOIL', 'inv_turnover', 'PWOOL', 'int_debt', 'gsubind_10',
    'POILWTI', 'PCPI_IX', 'cash_bev', 'PCHROM', 'PSILVER', 'ebitda_debt',
    'nix_ch1a', 'PBEEF', 'ca_gr3', 'PSAWMAL', 'rec_ch3a', 'ni_ch3a', 'nwc_ch1a',
    'PCPI_PC_CP_A_PT', 'col_ch1a', 'capx_at', 'PREODOM', 'POILAPSP', 'gsector_55',
    'ni_at_ch5', 'inv_ch1a', 'intrinsic_value', 'be_bev', 'capx_ch3a',
    'ocf_at_ch1', 'gsector_35', 'tax_pi', 'ca_cl', 'momentum_4w', 'ncl_gr3',
    'PSUGAISA', 'PALUM', 'capx_gr2', 'PTEAINDIA', 'PTOMATO', 'ap_ch3a', 'op_at',
    'dbnetis_at', 'PPOULT', 'nwc_ch3a', 'at_gr1', 'capx_ch1a', 'niq_at_ch1',
    'fnl_ch1a', 'debt_gr1', 'ebitda_ch1a', 'col_ch3a', 'sale_gr1', 'ncol_ch3a',
    'gind_40', 'eqnpo_at', 'PLOGORE', 'PLEAD', 'LMICS', 'PNRG', 'ni_at',
    'dbnetis_ch1a', 'PCOPP', 'PSEAF', 'ocf_ch1a', 'gsubind_80', 'be_ch1a',
    'gsubind_60', 'eqnetis_at', 'nca_gr3', 'f_score', 'ocf_debt', 'ol_ch3a',
    'ap_turnover', 'lt_gr3', 'PAPPLE', 'fcf_ch3a', 'ebitda_ch3a', 'PCOAL',
    'gsubind_15', 'PCOIL', 'PCOFFOTM', 'PWOOLF', 'PAGRI', 'ebitda_sale', 'PALLFNF',
    'PSALM', 'gsector_45', 'fcf_ppen', 'dp_ch1a', 'gind_30', 'gp_ch3a', 'at_gr3',
    'intan_ch3a', 'PNICK', 'PNGASEU', 'cl_gr1', 'PPLAT', 'rec_turnover',
    'ni_be_ch5', 'PZINC', 'tacc_nix', 'cash_ch3a', 'rec_ch1a', 'inv_gr3', 'spi_at',
    'POLVOIL', 'PPROPANE', 'be_gr3', 'ni_qtr_sale_qtr', 'cl_gr3', 'niq_at',
    'gsector_40', 'PWHEAMT', 'ocf_qtr_sale_qtr', 'eqis_at', 'nix_sale', 'debt_be',
    'PHIDE', 'gp_atl1', 'ebit_int', 'cshoc', 'PNFUEL', 'fnl_ch3a', 'tacc_at',
    'cogs_gr1', 'gsubind_20', 'PSAWORE', 'PORANG', 'fincf_at', 'lt_gr1', 'aliq_at',
    'fna_ch1a', 'ni_sale', 'cshtrd', 'PTEAMOM', 'ocf_sale', 'ocf_ch3a', 'debt_at',
    'ebit_at', 'sale_qtr_gr3', 'ncol_ch1a', 'ope_ch3a', 'ebit_sale', 'gsubind_30',
    'PTIMB', 'PLAMB', 'POILBRE', 'ap_days', 'o_score', 'PSOFT', 'capx_gr3',
    'gsubind_25', 'gind_60', 'eqbb_at', 'PGNUTS', 'POATS', 'div_at', 'intan_ch1a',
    'eqbb_ch3a', 'PSUGAUSA', 'netis_at', 'debtst_bev', 'PCOALAU', 'PCOFFROB',
    'gind_50', 'cl_lt', 'PSOYB', 'ebit_bev', 'PENTM', 'xsga_gr1', 'inv_days',
    'ocf_at', 'sale_gr3'
]

JPN_PSO = [
    'PPLAT', 'nca_gr3', 'capx_gr1', 'nfna_ch3a', 'ni_be_ch5', 'gsector_45',
    'int_debt', 'eqbb_ch1a', 'op_atl1', 'PINDU', 'PVOIL', 'PZINC', 'ope_ch3a',
    'gind_60', 'niq_at', 'ncol_ch3a', 'capx_ch1a', 'PCHROM', 'opex_gr3', 'PURAN',
    'gsector_55', 'fincf_at', 'rec_turnover', 'PFOOD', 'ope_be', 'col_ch1a',
    'PCOCO', 'nix_be', 'PSUGA', 'fna_ch1a', 'gsubind_40', 'cowc_ch3a',
    'dstnetis_ch1a', 'fincf_ch1a', 'gsubind_60', 'debtlt_debt', 'PCPI_IX',
    'PNGASEU', 'PSALM', 'ope_bel1', 'pi_sale', 'f_score', 'be_gr1', 'sale_nwc',
    'PCHANA', 'fincf_ch3a', 'gsubind_30', 'ni_at_ch5', 'dstnetis_at', 'ocf_sale',
    'ni_ch3a', 'oacc_at', 'POILDUB', 'be_gr3', 'ggroup_20', 'at_be', 'cash_ch3a',
    'PTEAMOM', 'ebit_ch1a', 'PNGASUS', 'ocf_at_ch1', 'eqbb_ch3a', 'PWOOL',
    'PRICENPQ', 'PTIMB', 'PCOFFOTM', 'eqnpo_at', 'PSAWMAL', 'PSILLUMP',
    'gsubind_10', 'ocf_ch1a', 'PLOGSK', 'sale_qtr_gr3', 'gind_30', 'gsubind_70',
    'PCOALSA', 'fcf_ocf', 'PBEVE', 'ggroup_10', 'at_gr1', 'xsga_gr3', 'fcf_sale',
    'ebit_int', 'cash_at', 'fcf_be', 'capx_gr3', 'ebitda_ch1a', 'PTEAINDIA',
    'PLAMB', 'PIORECR', 'PCOFF', 'fcf_ppen', 'cash_ch1a', 'PMANGELE', 'caliq_cl',
    'fcf_ch1a', 'PEXGALL', 'fi_bev', 'be_ch1a', 'gsector_40', 'ap_days', 'inv_ch1a',
    'PCOPP', 'nri_at', 'ni_sale', 'cash_lt', 'nix_ch3a', 'PFSHMEAL', 'PVANPENT',
    'nwc_ch1a', 'sale_qtr_gr1', 'PMAIZMT', 'cl_gr1', 'eqpo_ch3a', 'xsga_gr1',
    'momentum_1d', 'cl_lt', 'gp_ppen', 'PRUBB', 'PTOMATO', 'div_at', 'PGNUTS',
    'PLMMODY', 'PFERT', 'dstnetis_ch3a', 'niq_be_ch1', 'PSMEA', 'capx_ch3a',
    'dbnetis_at', 'ocfq_saleq_std', 'PAPPLE', 'ggroup_30', 'coa_ch1a', 'sale_gr1',
    'ol_ch3a', 'ebit_at', 'pstk_bev', 'debt_bev', 'momentum_4w', 'gsector_20',
    'cash_bev', 'gp_atl1', 'rsi', 'PPMETA', 'gind_20', 'ap_turnover', 'debtst_ch1a',
    'PMETA', 'cl_gr3', 'inv_days', 'PHARD', 'tax_ch3a', 'cash_conversion',
    'sale_bev', 'col_ch3a', 'ebitda_cl', 'PUREA', 'PTEASL', 'cop_bev', 'debt_be',
    'gsector_60', 'POILAPSP', 'PCOIL', 'ncol_ch1a', 'PDAP', 'PHEATOIL',
    'inv_turnover', 'ocf_be', 'roe_be_std', 'ope_ch1a', 'inv_ca', 'ca_gr3',
    'ebitda_bev', 'PALLMETA', 'tacc_at', 'ncl_gr1', 'PALLFNF', 'oacc_nix',
    'coa_ch3a', 'sale_be', 'PMEAT', 'ocf_at', 'ebitda_debt', 'div_ch1a'
]

JPN_MGO = [
    'nwc_at', 'gsector_35', 'lt_ppen', 'cl_gr3', 'xsga_gr3', 'PNICK', 'int_debtlt',
    'POILAPSP', 'PMETA', 'PSOYB', 'PPOTASH', 'PHARD', 'PTEAINDIA', 'PTOMATO',
    'niq_at_ch1'
]

FEATURE_SETS = {
    "fra": {
        "mrmr_filter": [
            'cshoc', 'PAGRI', 'cshtrd', 'PCPIHARC_IX',
            'PFSHMEAL', 'd_oscillator_4w', 'PPMETA', 'ocf_qtr_sale_qtr',
            'ebit_sale', 'PGOLD', 'd_oscillator_6w', 'PSUGAUSA', 'PCHANA',
            'rsi', 'PBEEF', 'PTIN', 'PLAMB', 'PALLMETA', 'PSILVER',
            'PSEAF', 'PCPIHAA_PC_CP_A_PT', 'PSALM', 'PCPIHARC_PC_PP_PT',
            'PCPIHAHO_PC_CP_A_PT', 'PCOPP', 'PCOCO', 'PCPIA_PC_CP_A_PT',
            'PCPIHO_PC_CP_A_PT', 'PMANGELE', 'PCPIHAED_IX', 'PCPIEC_PC_PP_PT',
            'pi_nix', 'ebit_int', 'PMETA', 'PEXGMETA', 'momentum_1d', 'PCOALAU',
            'PCPIHAF_PC_CP_A_PT', 'PIORECR', 'PENTM', 'PCPIHA_PC_CP_A_PT',
            'PCPIR_PC_PP_PT', 'PTOMATO', 'PCOIL', 'PCPIHAC_PC_PP_PT', 'PCPIF_PC_CP_A_PT',
            'PCPI_IX', 'PINDU', 'PPOIL'
        ]
    },
    "nasnor": {
        "all": TECHNICAL_FEATURES["nasnor"]
        + FUNDAMENTAL_FEATURES["nasnor"]
        + MACRO_FEATURES["nasnor"]
        + SECTOR_FEATURES["nasnor"],
        "var_filter": NASNOR_VAR_FILTER,
        "mrmr_filter": NASNOR_MRMR_FILTER[:30],
        "forward": NASNOR_FORWARD,
        "backward": NASNOR_BACKWARD,
        "ga": NASNOR_GA,
        "pso": NASNOR_PSO,
        "mgo": NASNOR_MGO,
        "union_x_y_xgb": list(set(NASNOR_MGO) | set(NASNOR_MRMR_FILTER[:30])),
        "union_x_z_xgb": list(set(NASNOR_MGO) | set(NASNOR_PSO)),
        "union_y_z_xgb": list(set(NASNOR_MRMR_FILTER[:30]) | set(NASNOR_PSO)),
        "union_x_y_z_xgb": list(set(NASNOR_MGO) | set(NASNOR_MRMR_FILTER[:30]) | set(NASNOR_PSO)),
        "inter_x_y_xgb": list(set(NASNOR_MGO) & set(NASNOR_MRMR_FILTER[:30])),
        "inter_x_z_xgb": list(set(NASNOR_MGO) & set(NASNOR_PSO)),
        "inter_y_z_xgb": list(set(NASNOR_MRMR_FILTER[:30]) & set(NASNOR_PSO)),
        "inter_x_y_z_xgb": list(set(NASNOR_MGO) & set(NASNOR_MRMR_FILTER[:30]) & set(NASNOR_PSO)),
        "union_x_y_svr": list(set(NASNOR_MGO) | set(NASNOR_MRMR_FILTER[:30])),
        "union_x_z_svr": list(set(NASNOR_MGO) | set(NASNOR_FORWARD)),
        "union_y_z_svr": list(set(NASNOR_MRMR_FILTER[:30]) | set(NASNOR_FORWARD)),
        "union_x_y_z_svr": list(set(NASNOR_MGO) | set(NASNOR_MRMR_FILTER[:30]) | set(NASNOR_FORWARD)),
        "inter_x_y_svr": list(set(NASNOR_MGO) & set(NASNOR_MRMR_FILTER[:30])),
        "inter_x_z_svr": list(set(NASNOR_MGO) & set(NASNOR_FORWARD)),
        "inter_y_z_svr": list(set(NASNOR_MRMR_FILTER[:30]) & set(NASNOR_FORWARD)),
        "inter_x_y_z_svr": list(set(NASNOR_MGO) & set(NASNOR_MRMR_FILTER[:30]) & set(NASNOR_FORWARD)),
    },
    "usa": {
        "all": TECHNICAL_FEATURES["usa"]
        + FUNDAMENTAL_FEATURES["usa"]
        + MACRO_FEATURES["usa"]
        + SECTOR_FEATURES["usa"],
        "var_filter": USA_VAR_FILTER,
        "mrmr_filter": USA_MRMR_FILTER[:10],
        "forward": USA_FORWARD,
        "backward": USA_BACKWARD,
        "ga": USA_GA,
        "pso": USA_PSO,
        "mgo": USA_MGO,
        "union_x_y_svr": list(set(USA_BACKWARD) | set(USA_PSO)),
        "union_x_z_svr": list(set(USA_BACKWARD) | set(USA_FORWARD)),
        "union_y_z_svr": list(set(USA_PSO) | set(USA_FORWARD)),
        "union_x_y_z_svr": list(set(USA_BACKWARD) | set(USA_PSO) | set(USA_FORWARD)),
        "inter_x_y_svr": list(set(USA_BACKWARD) & set(USA_PSO)),
        "inter_x_z_svr": list(set(USA_BACKWARD) & set(USA_FORWARD)),
        "inter_y_z_svr": list(set(USA_PSO) & set(USA_FORWARD)),
        "inter_x_y_z_svr": list(set(USA_BACKWARD) & set(USA_PSO) & set(USA_FORWARD)),
        "union_x_y_xgb": list(set(USA_BACKWARD) | set(USA_MGO)),
        "union_x_z_xgb": list(set(USA_BACKWARD) | set(USA_FORWARD)),
        "union_y_z_xgb": list(set(USA_MGO) | set(USA_FORWARD)),
        "union_x_y_z_xgb": list(set(USA_BACKWARD) | set(USA_MGO) | set(USA_FORWARD)),
        "inter_x_y_xgb": list(set(USA_BACKWARD) & set(USA_MGO)),
        "inter_x_z_xgb": list(set(USA_BACKWARD) & set(USA_FORWARD)),
        "inter_y_z_xgb": list(set(USA_MGO) & set(USA_FORWARD)),
        "inter_x_y_z_xgb": list(set(USA_BACKWARD) & set(USA_MGO) & set(USA_FORWARD)),
    },
    "jpn": {
        "all": TECHNICAL_FEATURES["jpn"]
        + FUNDAMENTAL_FEATURES["jpn"]
        + MACRO_FEATURES["jpn"]
        + SECTOR_FEATURES["jpn"],
        "var_filter": JPN_VAR_FILTER,
        "mrmr_filter": JPN_MRMR_FILTER[:30],
        "forward": JPN_FORWARD,
        "backward": JPN_BACKWARD,
        "ga": JPN_GA,
        "pso": JPN_PSO,
        "mgo": JPN_MGO,
        "union_x_y_xgb": list(set(JPN_PSO) | set(JPN_BACKWARD)),
        "union_x_z_xgb": list(set(JPN_PSO) | set(JPN_MRMR_FILTER[:30])),
        "union_y_z_xgb": list(set(JPN_BACKWARD) | set(JPN_MRMR_FILTER[:30])),
        "union_x_y_z_xgb": list(set(JPN_PSO) | set(JPN_BACKWARD) | set(JPN_MRMR_FILTER[:30])),
        "inter_x_y_xgb": list(set(JPN_PSO) & set(JPN_BACKWARD)),
        "inter_x_z_xgb": list(set(JPN_PSO) & set(JPN_MRMR_FILTER[:30])),
        "inter_y_z_xgb": list(set(JPN_BACKWARD) & set(JPN_MRMR_FILTER[:30])),
        "inter_x_y_z_xgb": list(set(JPN_PSO) & set(JPN_BACKWARD) & set(JPN_MRMR_FILTER[:30])),
        "union_x_y_svr": list(set(JPN_PSO) | set(JPN_FORWARD)),
        "union_x_z_svr": list(set(JPN_PSO) | set(JPN_GA)),
        "union_y_z_svr": list(set(JPN_FORWARD) | set(JPN_GA)),
        "union_x_y_z_svr": list(set(JPN_PSO) | set(JPN_FORWARD) | set(JPN_GA)),
        "inter_x_y_svr": list(set(JPN_PSO) & set(JPN_FORWARD)),
        "inter_x_z_svr": list(set(JPN_PSO) & set(JPN_GA)),
        "inter_y_z_svr": list(set(JPN_FORWARD) & set(JPN_GA)),
        "inter_x_y_z_svr": list(set(JPN_PSO) & set(JPN_FORWARD) & set(JPN_GA)),
    },
}


def categorize_features(features: list[str], dataset: str):
    """Categorizes given features into one of four categories:
        - Technical
        - Fundamental
        - Macro
        - Sector
    Args:
        features (list[str]): Features to categorize
        dataset (str): Dataset containing the features
    Returns:
        tuple[list[str]]: List of features for each category
    """
    technical_features = []
    fundamental_features = []
    macro_features = []
    sector_features = []
    for feature in features:
        if feature in TECHNICAL_FEATURES[dataset]:
            technical_features.append(feature)
        elif feature in FUNDAMENTAL_FEATURES[dataset]:
            fundamental_features.append(feature)
        elif feature in MACRO_FEATURES[dataset]:
            macro_features.append(feature)
        elif feature in SECTOR_FEATURES[dataset]:
            sector_features.append(feature)
        else:
            print(f"feature not found for categorization in {dataset}! {feature}")
    return technical_features, fundamental_features, macro_features, sector_features
