# FEATURE CONTRACT — NOTEBOOK 02

GENERATED_AT: 2026-02-26 00:53:09 UTC

COLUMN NAMING:
- ASSET LEVEL: {DAG_NODE}__{TICKER}__{FEATURE_NAME}__{WINDOW}
- MARKET LEVEL: {DAG_NODE}__{FEATURE_NAME}__{WINDOW(optional)}

MARKDOWN_TABLE_EXPORT_FAILED
EXPORT_EXCEPTION: Missing optional dependency 'tabulate'.  Use pip or conda to install tabulate.

CSV_BACKUP: feature_contract.csv

SAMPLE_ROWS
```
        feature_column dag_node            feature_family ticker window                    inputs                                     formula      leakage_policy
           MACRO__dtb3    MACRO         macro_risk_inputs         daily         fred_macro_series                          pass_through(DTB3) uses_dates_<=t_only
         MACRO__t10y2y    MACRO         macro_risk_inputs         daily         fred_macro_series                        pass_through(T10Y2Y) uses_dates_<=t_only
         MACRO__vixcls    MACRO         macro_risk_inputs         daily         fred_macro_series                        pass_through(VIXCLS) uses_dates_<=t_only
 ML__pca_ret_pc1__252d       ML rolling_pca_latent_factor          252d cross_section_log_returns PCA_scores_pc1(standardized_returns_window) uses_dates_<=t_only
 ML__pca_ret_pc2__252d       ML rolling_pca_latent_factor          252d cross_section_log_returns PCA_scores_pc2(standardized_returns_window) uses_dates_<=t_only
 ML__pca_ret_pc3__252d       ML rolling_pca_latent_factor          252d cross_section_log_returns PCA_scores_pc3(standardized_returns_window) uses_dates_<=t_only
MOM__DBC__cum_ret__10d      MOM            price_momentum    DBC    10d               log_returns      exp(sum(log_return[t-window+1:t])) - 1 uses_dates_<=t_only
MOM__DBC__cum_ret__21d      MOM            price_momentum    DBC    21d               log_returns      exp(sum(log_return[t-window+1:t])) - 1 uses_dates_<=t_only
 MOM__DBC__cum_ret__5d      MOM            price_momentum    DBC     5d               log_returns      exp(sum(log_return[t-window+1:t])) - 1 uses_dates_<=t_only
MOM__DBC__cum_ret__63d      MOM            price_momentum    DBC    63d               log_returns      exp(sum(log_return[t-window+1:t])) - 1 uses_dates_<=t_only
MOM__EEM__cum_ret__10d      MOM            price_momentum    EEM    10d               log_returns      exp(sum(log_return[t-window+1:t])) - 1 uses_dates_<=t_only
MOM__EEM__cum_ret__21d      MOM            price_momentum    EEM    21d               log_returns      exp(sum(log_return[t-window+1:t])) - 1 uses_dates_<=t_only
 MOM__EEM__cum_ret__5d      MOM            price_momentum    EEM     5d               log_returns      exp(sum(log_return[t-window+1:t])) - 1 uses_dates_<=t_only
MOM__EEM__cum_ret__63d      MOM            price_momentum    EEM    63d               log_returns      exp(sum(log_return[t-window+1:t])) - 1 uses_dates_<=t_only
MOM__EFA__cum_ret__10d      MOM            price_momentum    EFA    10d               log_returns      exp(sum(log_return[t-window+1:t])) - 1 uses_dates_<=t_only
MOM__EFA__cum_ret__21d      MOM            price_momentum    EFA    21d               log_returns      exp(sum(log_return[t-window+1:t])) - 1 uses_dates_<=t_only
 MOM__EFA__cum_ret__5d      MOM            price_momentum    EFA     5d               log_returns      exp(sum(log_return[t-window+1:t])) - 1 uses_dates_<=t_only
MOM__EFA__cum_ret__63d      MOM            price_momentum    EFA    63d               log_returns      exp(sum(log_return[t-window+1:t])) - 1 uses_dates_<=t_only
MOM__GLD__cum_ret__10d      MOM            price_momentum    GLD    10d               log_returns      exp(sum(log_return[t-window+1:t])) - 1 uses_dates_<=t_only
MOM__GLD__cum_ret__21d      MOM            price_momentum    GLD    21d               log_returns      exp(sum(log_return[t-window+1:t])) - 1 uses_dates_<=t_only
 MOM__GLD__cum_ret__5d      MOM            price_momentum    GLD     5d               log_returns      exp(sum(log_return[t-window+1:t])) - 1 uses_dates_<=t_only
MOM__GLD__cum_ret__63d      MOM            price_momentum    GLD    63d               log_returns      exp(sum(log_return[t-window+1:t])) - 1 uses_dates_<=t_only
MOM__HYG__cum_ret__10d      MOM            price_momentum    HYG    10d               log_returns      exp(sum(log_return[t-window+1:t])) - 1 uses_dates_<=t_only
MOM__HYG__cum_ret__21d      MOM            price_momentum    HYG    21d               log_returns      exp(sum(log_return[t-window+1:t])) - 1 uses_dates_<=t_only
 MOM__HYG__cum_ret__5d      MOM            price_momentum    HYG     5d               log_returns      exp(sum(log_return[t-window+1:t])) - 1 uses_dates_<=t_only
```
