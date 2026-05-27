"""
RiskML Capstone — Streamlit Dashboard.

Read-only presentation layer for the riskml-capstone project. Reads
pre-computed artifacts from reports/tables/ and reports/figures/ produced
by notebooks NB01 through NB07, plus harvested dashboard figures.
No model training, no data API calls, no overwrites of upstream artifacts.

Dashboard structure (8 sections, 29 tabs):
    §1 Overview                       — 1A Research Question · 1B Big Picture · 1C Key Finding · 1D How to Read
    §2 Data & Feature Engineering    — 2A Data Sources · 2B ETF Universe · 2C Feature Families · 2D Sentiment Boundary · 2E Pipeline Map
    §3 Forecasting Design             — 3A What the Model Outputs · 3B Walk-Forward · 3C RMSE/MAE Logic · 3D Leakage Buffer
    §4 DAG Constraint Experiment      — 4A Manual DAG · 4B Allowed Features · 4C Blocked Features · 4D Baseline ↔ Causal Twins
    §5 Results & Interpretability     — 5A Forecast Accuracy · 5B Entropy Diagnostics · 5C Feature Importance · 5D Hypothesis Verdicts
    §6 Portfolio & Stress Testing     — 6A Inverse-Vol Weights · 6B Equity Curves · 6C Drawdowns · 6D Regime & Stress
    §7 Engineering & Deployment       — 7A Package Extraction · 7B CI Tests · 7C Docker · 7D Azure Plan
    §8 Artifact Manifest              — (no tabs)

Content sourcing convention:
    Prose marked with the comment "# Source: AI-drafted connective prose;
    reviewed and approved by Steve." is connective tissue drafted with AI
    assistance and reviewed for accuracy. All other prose is pulled verbatim
    or near-verbatim from the capstone report (riskml_Report_2026_04_10.docx),
    the reverse-engineered outline, or the Pipeline Visualized document
    authored jointly with Andrew.

Run locally:
    streamlit run app/streamlit_dashboard.py

Author : Steven Archuleta
Course : MScFE 690 Capstone Project (WorldQuant University)
Partner: Andrew Nilsen (theoretical sections, Pipeline Visualized document)
"""

from pathlib import Path

import pandas as pd
import streamlit as st


# -----------------------------------------------------
# Page configuration — must be the first Streamlit call.
# -----------------------------------------------------
st.set_page_config(
    page_title="RiskML Capstone Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -------------------------------------------------
# Path resolution
# Allows the app to run from any working directory.
# -------------------------------------------------
def _find_repo_root(start: Path) -> Path:
    """Walk upward from start until a .git or pyproject.toml is found."""
    for candidate in [start] + list(start.parents):
        if (candidate / ".git").exists() or (candidate / "pyproject.toml").exists():
            return candidate
    return start


REPO_ROOT = _find_repo_root(Path(__file__).resolve().parent)
TABLES_DIR = REPO_ROOT / "reports" / "tables"
FIGURES_DIR = REPO_ROOT / "reports" / "figures"


# ---------------------------------------------------------------------
# Cached loaders. @st.cache_data avoids re-reading parquet/CSV on every
# sidebar interaction, which keeps the UI responsive.
# ---------------------------------------------------------------------
@st.cache_data
def load_table(filename: str) -> pd.DataFrame | None:
    """Return a CSV from reports/tables/ as a DataFrame, or None if absent."""
    path = TABLES_DIR / filename
    if not path.exists():
        return None
    return pd.read_csv(path)


def figure_path(filename: str) -> Path | None:
    """Return absolute path to a figure if present, else None."""
    path = FIGURES_DIR / filename
    return path if path.exists() else None


def render_figure(filename: str, caption: str = "") -> None:
    """Display a figure if it exists; show a friendly note if it does not."""
    path = figure_path(filename)
    if path is None:
        st.info(f"Figure not yet available: `{filename}`")
        return
    st.image(str(path), caption=caption, width="stretch")


def render_table(filename: str, caption: str = "") -> None:
    """Display a CSV table if it exists; show a friendly note if it does not."""
    df = load_table(filename)
    if df is None:
        st.info(f"Table not yet available: `{filename}`")
        return
    if caption:
        st.caption(caption)
    st.dataframe(df, width="stretch")


# -------------------
# Sidebar navigation.
# -------------------
st.sidebar.title("📊 RiskML Capstone")
st.sidebar.caption("Causal-Aware ML-Driven Risk Forecasting")

# Tiny gap before credits via a single empty markdown line.
st.sidebar.markdown("")

# Credits block — centered, default text color (renders well in both
# light and dark Streamlit themes).
st.sidebar.markdown(
    "<div style='text-align: center; line-height: 1.5;'>"
    "01 June 2026<br>"
    "WorldQuant University<br>"
    "MScFE<br>"
    "Master of Science<br>"
    "in<br>"
    "Financial Engineering<br>"
    "Steven Archuleta (USA)<br>"
    "Andrew Nilsen (Indonesia)"
    "</div>",
    unsafe_allow_html=True,
)

st.sidebar.divider()

PAGES = [
    "§1 Overview",
    "§2 Data & Feature Engineering",
    "§3 Forecasting Design",
    "§4 DAG Constraint Experiment",
    "§5 Results & Interpretability",
    "§6 Portfolio & Stress Testing",
    "§7 Engineering & Deployment",
    "§8 Artifact Manifest",
]
page = st.sidebar.radio("Navigate", PAGES)

st.sidebar.divider()

st.sidebar.markdown("Repo: [`riskml-capstone` ↗](https://github.com/stevearchuleta/riskml-capstone)")
# ================================================================
# FINAL REPORT PDF DOWNLOAD - ACTIVE BUTTON OR PENDING PLACEHOLDER
# ================================================================
pdf_path = REPO_ROOT / "reports" / "riskml_capstone_final.pdf"
if pdf_path.exists():
    with open(pdf_path, "rb") as f:
        st.sidebar.download_button(
            label="📄 Download Report",
            data=f.read(),
            file_name="riskml_capstone_final.pdf",
            mime="application/pdf",
            width="stretch",
        )
else:
    st.sidebar.caption("📄 Final Report PDF")


st.sidebar.caption(
    "Read-only dashboard. All numbers derive from saved artifacts produced "
    "by notebooks NB01 through NB06."
)


# =====================================================================
# §1 Overview
# =====================================================================
if page == "§1 Overview":
    st.title("§1 Overview")
    # Source: AI-drafted connective prose; reviewed and approved by Steve.
    st.markdown(
        "*The Research Question  ·  The Structure of the Experiment  ·  "
        "The Result  ·  How to Navigate this Dashboard*"
    )

    tab_1a, tab_1b, tab_1c, tab_1d = st.tabs(
        [
            "1A · Research Question",
            "1B · Big Picture",
            "1C · Key Finding",
            "1D · How to Read",
        ]
    )

    # -----------------------------------------------------------------
    # 1A Research Question
    # Source: report Ch. 1.3 verbatim; report §1.1 paragraph 6 verbatim.
    # -----------------------------------------------------------------
    with tab_1a:
        st.subheader("Research Question")
        st.markdown(
            "> **How does imposing manual causal constraints on an ML-driven "
            "risk forecasting pipeline affect forecast accuracy, portfolio "
            "performance, and interpretability compared to unconstrained "
            "baseline models?**"
        )
        st.markdown(
            "This capstone project evaluates whether a small, manually specified "
            "directed acyclic graph — encoding economically motivated directional "
            "constraints over factor exposures, sentiment signals, and risk "
            "estimates — measurably improves out-of-sample risk forecast accuracy, "
            "portfolio-level risk control, and factor-exposure interpretability "
            "relative to an unconstrained machine-learning baseline, within a "
            "single reproducible pipeline."
        )
        st.markdown(
            "The question is operationalized through a structured ablation design "
            "that progressively adds pipeline components — NLP sentiment, "
            "ML-constructed factors, and DAG constraints — to an unconstrained "
            "baseline model, isolating the marginal contribution of each stage."
        )

    # -----------------------------------------------------------------
    # 1B Big Picture
    # Source: custom DAG figure + report §3.5 paragraphs 645-650 verbatim.
    # -----------------------------------------------------------------
    with tab_1b:
        st.subheader("The Manual DAG")
        render_figure(
            "notebook07_dag_clean.png",
            caption="Manual DAG enforced at the risk-forecasting stage (NB04). "
            "Three causal chains plus two exogenous conditioning edges.",
        )
        st.markdown(
            "The central methodological contribution of the capstone is a small, "
            "manually specified directed acyclic graph that encodes economically "
            "motivated directional constraints on the modeling pipeline. The DAG "
            "restricts which features are available to each modeling stage and "
            "enforces a sequential estimation order aligned with domain assumptions "
            "about the plausible direction of information flow."
        )
        st.markdown(
            "**The DAG functions as an information-flow contract and governance "
            "regularizer that reduces the search space for spurious relationships.** "
            "The DAG does not claim to represent the true data-generating process."
        )
        st.markdown(
            "The manual DAG contains **nine nodes and seven directed arrows** "
            "organized into three causal chains, plus two exogenous conditioning "
            "edges:"
        )
        st.markdown(
            "- **Sentiment → Momentum → Returns** — news tone influences momentum "
            "before momentum estimates enter the return forecast stage.\n"
            "- **Value → Returns** — relative valuation signals (HML beta) "
            "contribute directly to expected returns.\n"
            "- **Volatility → Risk → Allocation** — observed historical volatility "
            "informs the forward risk forecast, which determines portfolio weights.\n"
            "- **MACRO → Risk** and **REGIME → Risk** — macroeconomic indicators "
            "(VIXCLS, T10Y2Y, DTB3) and the binary VIX-regime indicator are made "
            "available to the Risk estimation stage as conditioning variables."
        )

    # -----------------------------------------------------------------
    # 1C Key Finding
    # Source: corrected lead sentence
    #         + four verdict cards in 2x2 grid.
    # -----------------------------------------------------------------
    with tab_1c:
        st.subheader("Key Finding")
        st.markdown(
            "> **The DAG constraint appears to trade raw forecast accuracy for "
            "portfolio risk control. In the tree-model comparison, CAUSAL_XGBOOST "
            "produced 16.2% higher RMSE than unconstrained XGBoost, but reduced "
            "maximum drawdown to −5.83% and raised Calmar to 1.498 — the strongest "
            "tree-family drawdown/Calmar profile in the study.**"
        )
        st.divider()
        st.markdown("**Three formal hypotheses + one secondary interpretability diagnostic:**")

        # 2x2 verdict cards.
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### H1 · Forecast Accuracy")
            st.error("**Not confirmed**")
            st.caption(
                "The DAG-constrained model produced higher aggregate RMSE "
                "(0.0811) than both the unconstrained XGBoost baseline (0.0698, "
                "+16.2%) and the EWMA naive forecast (0.0708, +14.5%); the "
                "constrained model won 0 of 14 per-ticker RMSE comparisons, and "
                "Diebold-Mariano HAC tests found no accuracy difference "
                "statistically distinguishable from noise (min p = 0.328)."
            )

            st.markdown("##### H3 · NLP Sentiment Contribution")
            st.warning("**Directionally supported**")
            st.caption(
                "A matched-window post-2020 evaluation produced an aggregate RMSE "
                "delta of −0.000324 (−0.32%) in the correct direction, but no "
                "Diebold-Mariano test was applied so statistical significance "
                "cannot be claimed; characterized as directionally supported "
                "rather than confirmed."
            )

        with col2:
            st.markdown("##### H2 · Portfolio Risk Control")
            st.warning("**Partially confirmed, family-specific**")
            st.caption(
                "In the tree comparison, CAUSAL_XGBOOST achieved the shallowest "
                "drawdown (−5.83%) and highest Calmar ratio (1.498); CAUSAL_LSTM "
                "produced the strongest risk-targeting; but BASELINE_LSTM owned "
                "the best full-period Sharpe (0.777), so the DAG benefit is real "
                "but uneven across model families."
            )

            st.markdown("##### Secondary Interpretability Diagnostic")
            st.success("**Consistent at both tiers**")
            st.caption(
                "Normalized Shannon entropy of XGBoost gain shares rose under DAG "
                "constraints for all three representative tickers (GLD +0.003, "
                "SPY +0.012, TLT +0.031); factor-exposure entropy from rolling "
                "Fama-French regressions is highest for CAUSAL_XGBOOST."
            )

    # -----------------------------------------------------------------
    # 1D How to Read
    # Source: AI-drafted connective prose; reviewed and approved by Steve.
    # -----------------------------------------------------------------
    with tab_1d:
        st.subheader("How to Read This Dashboard")
        st.markdown(
            "The dashboard is organized as a left-to-right tour of the capstone "
            "pipeline, with each section corresponding to a stage of the work:"
        )
        st.markdown(
            "1. **§1 Overview** — the question, the DAG, the headline result.\n"
            "2. **§2 Data & Feature Engineering** — the raw material and how the "
            "feature matrix is built.\n"
            "3. **§3 Forecasting Design** — what the models predict and how "
            "accuracy is measured.\n"
            "4. **§4 DAG Constraint Experiment** — the gate, what it allows, "
            "what it blocks, and the matched-pair experiment that tests it.\n"
            "5. **§5 Results & Interpretability** — what worked, what did not, "
            "and the formal verdicts.\n"
            "6. **§6 Portfolio & Stress Testing** — how the forecasts translate "
            "into portfolio behavior, including the April 2025 stress episode.\n"
            "7. **§7 Engineering & Deployment** — the package, the CI, the "
            "Docker container, and the planned Azure deployment.\n"
            "8. **§8 Artifact Manifest** — a verification view listing every "
            "CSV and PNG the dashboard reads."
        )
        st.info(
            "Tabs within each section answer one question per tab."
        )


# =====================================================================
# §2 Data & Feature Engineering
# =====================================================================
elif page == "§2 Data & Feature Engineering":
    st.title("§2 Data & Feature Engineering")
    # Source: AI-drafted connective prose; reviewed and approved by Steve.
    st.markdown(
        "*The data sources  the asset universe · The engineered feature families · "
        "The sentiment-coverage boundary · A tour of NB01 through NB06.*"
    )

    tab_2a, tab_2b, tab_2c, tab_2d, tab_2e = st.tabs(
        [
            "2A · Data Sources",
            "2B · ETF Universe",
            "2C · Feature Families",
            "2D · Sentiment Boundary",
            "2E · Pipeline Map",
        ]
    )

    # -----------------------------------------------------------------
    # 2A Data Sources
    # Source: report §1.5 paragraph 22 verbatim + Pipeline Visualized
    # paragraph 1 verbatim.
    # -----------------------------------------------------------------
    with tab_2a:
        st.subheader("Tracking the Data")
        st.markdown(
            "The pipeline operates over **2,798 trading days** "
            "(2015-01-05 through 2026-02-19). Four data sources feed the panel:"
        )
        st.markdown(
            "- **Yahoo Finance** — daily adjusted close prices for 14 diversified "
            "ETFs.\n"
            "- **Kenneth French Data Library** — daily Fama-French factors "
            "(Mkt-RF, SMB, HML, RF).\n"
            "- **FRED** (Federal Reserve Economic Data) — DTB3 (3-month Treasury), "
            "VIXCLS (CBOE Volatility Index), T10Y2Y (10-year minus 2-year yield).\n"
            "- **Alpha Vantage NEWS_SENTIMENT** — financial-market article records "
            "aggregated into daily market-level sentiment scores."
        )
        st.markdown(
            "All four sources are pulled in NB01, aligned to the ETF trading "
            "calendar, and saved as Parquet files."
        )
        render_figure(
            "vixcls_timeseries.png",
            caption="VIXCLS aligned to the ETF trading calendar — the 2020 COVID "
            "spike and the April 2025 tariff-shock episode are both visible as "
            "single sharp peaks.",
        )

    # -----------------------------------------------------------------
    # 2B ETF Universe
    # Source: report §1.5 paragraph 22 verbatim.
    # -----------------------------------------------------------------
    with tab_2b:
        st.subheader("The 14-ETF Asset Universe")
        st.markdown(
            "The asset universe consists of 14 diversified ETFs spanning equity, "
            "fixed income, credit, and commodities:"
        )
        st.markdown(
            "- **U.S. large-cap equity:** SPY, QQQ\n"
            "- **U.S. small-cap equity:** IWM\n"
            "- **International developed equity:** EFA\n"
            "- **Emerging-market equity:** EEM\n"
            "- **Sector equity:** XLK (technology), XLF (financials), "
            "XLE (energy), XLV (healthcare)\n"
            "- **U.S. Treasuries:** TLT\n"
            "- **Investment-grade credit:** LQD\n"
            "- **High-yield credit:** HYG\n"
            "- **Gold:** GLD\n"
            "- **Broad commodities:** DBC"
        )
        render_figure(
            "etf_universe_returns.png",
            caption="Cumulative log-return index (base = 1.0) for all 14 ETFs "
            "from 2015 through 2026.",
        )

    # -----------------------------------------------------------------
    # 2C Feature Families
    # Source: outline §3.4 + Pipeline Visualized paragraph 2 verbatim.
    # -----------------------------------------------------------------
    with tab_2c:
        st.subheader("How Features Are Engineered")
        st.markdown(
            "For every trading day and every ETF, the pipeline computes "
            "**152 engineered features in the full panel**, with a "
            "**151-feature non-sentiment working matrix** used for the main "
            "2016–2025 risk-stage evaluation. The seven prefix-tagged families "
            "are:"
        )

        feature_families = pd.DataFrame(
            [
                {"Prefix": "VOL__", "Count": 70, "Family": "Volatility (EWMA + realized-vol windows)", "DAG status at risk stage": "Allowed"},
                {"Prefix": "MOM__", "Count": 56, "Family": "Momentum (cumulative log returns 5/10/21/63 days)", "DAG status at risk stage": "Blocked"},
                {"Prefix": "VAL__", "Count": 18, "Family": "Value (rolling HML beta + factor controls)", "DAG status at risk stage": "Blocked"},
                {"Prefix": "MACRO__", "Count": 3, "Family": "Macro (VIXCLS, T10Y2Y, DTB3)", "DAG status at risk stage": "Allowed"},
                {"Prefix": "ML__", "Count": 3, "Family": "Latent ML factors (rolling PCA components)", "DAG status at risk stage": "Blocked"},
                {"Prefix": "REGIME__", "Count": 1, "Family": "Binary VIX-regime indicator", "DAG status at risk stage": "Allowed"},
                {"Prefix": "SENT__", "Count": 1, "Family": "Lagged market-sentiment feature", "DAG status at risk stage": "Blocked"},
            ]
        )

        # Bold-italicize the three rows whose DAG status is "Allowed" (VOL__,
        # MACRO__, REGIME__) so the gate-passing families stand out visually.
        def _style_allowed_rows(row: pd.Series) -> list[str]:
            if row["DAG status at risk stage"] == "Allowed":
                return ["font-weight: bold; font-style: italic"] * len(row)
            return [""] * len(row)

        styled_families = feature_families.style.apply(_style_allowed_rows, axis=1)
        st.dataframe(styled_families, width="stretch", hide_index=True)

        st.markdown(
            "**Total: 152 engineered features in the full panel; 151 features in "
            "the non-sentiment working matrix; 74 features retained at the "
            "risk-forecasting stage after DAG prefix gating** (VOL 70 + MACRO 3 + "
            "REGIME 1 = 74), with 77 features blocked by the gate."
        )
        render_figure(
            "notebook02_feature_counts_by_dag.png",
            caption="Feature counts by DAG family, color-coded by allowed-at-risk-"
            "stage status.",
        )

    # -----------------------------------------------------------------
    # 2D Sentiment Boundary
    # Source: outline §3.4 sentiment block + report §3.3.4 + Pipeline
    # Visualized paragraph 28 verbatim.
    # -----------------------------------------------------------------
    with tab_2d:
        st.subheader("The Sentiment Coverage Boundary")
        st.markdown(
            "Alpha Vantage NEWS_SENTIMENT provides a large post-2020 corpus — "
            "**458,662 financial-market articles** in this project — but the "
            "provider's archive begins in January 2020. **The resulting "
            "pre-2020 gap is a structural vendor-coverage boundary, not a "
            "research-design failure.**"
        )
        st.markdown(
            "NB01 normalized, deduplicated, aggregated, and one-trading-day-"
            "lagged the corpus into a daily market-level sentiment score. "
            "Because pre-2020 sentiment is structurally unavailable from the "
            "vendor, it is excluded from the main 2016–2025 risk-stage matrix "
            "and therefore isolated to the **H3 matched-window comparison** in "
            "NB03, where the sentiment-inclusive and sentiment-excluded models "
            "are compared on identical post-2020 data (1,496 rows, 300 test "
            "observations per ticker). This **matched-window design** avoids "
            "an unfair full-sample comparison driven by missingness rather "
            "than signal quality."
        )
        render_figure(
            "feature_missingness_top30.png",
            caption="Top 30 features by missing fraction. SENT__sentiment_market "
            "tops the list at roughly 45% missing — the structural pre-2020 gap.",
        )

    # -----------------------------------------------------------------
    # 2E Pipeline Map
    # Source: outline §4 NB01-NB07 table verbatim.
    # -----------------------------------------------------------------
    with tab_2e:
        st.subheader("From Notebook to Notebook")
        st.markdown(
            "The seven-notebook pipeline structures the analysis. NB01 through "
            "NB06 form the analytical backbone; NB07 serves the presentation "
            "layer."
        )
        notebook_map = pd.DataFrame(
            [
                {"NB": "NB01", "Role": "Data extraction and ETL", "Output": "Raw price panel + factor + macro + sentiment Parquet files"},
                {"NB": "NB02", "Role": "Feature engineering and target construction", "Output": "152-column feature matrix + frozen 20-day forward realized-vol target"},
                {"NB": "NB03", "Role": "Stylized EDA and unconstrained baselines", "Output": "EWMA, XGBOOST, BASELINE_MLP, BASELINE_LSTM forecasts + matched-window H3 test"},
                {"NB": "NB04", "Role": "DAG-constrained risk forecasting", "Output": "CAUSAL_XGBOOST, CAUSAL_MLP, CAUSAL_LSTM forecasts + entropy diagnostics"},
                {"NB": "NB05", "Role": "Portfolio construction and backtesting", "Output": "Seven-model volatility-targeted portfolio backtests"},
                {"NB": "NB06", "Role": "Validation, ablation, regime, stress, inference", "Output": "Ablation table + regime-conditional metrics + stress-episode analysis + Diebold-Mariano HAC tests"},
                {"NB": "NB07", "Role": "Presentation and dashboard support", "Output": "This dashboard, plus the artifact manifest"},
            ]
        )
        st.dataframe(notebook_map, width="stretch", hide_index=True)


# =====================================================================
# §3 Forecasting Design
# =====================================================================
elif page == "§3 Forecasting Design":
    st.title("§3 Forecasting Design")
    # Source: AI-drafted connective prose; reviewed and approved by Steve.
    st.markdown(
        "*These are the two most important conceptual clarifications in this "
        "capstone project: (1) what the models actually output, and (2) how "
        "those models use the output in two completely separate ways.*"
    )

    tab_3a, tab_3b, tab_3c, tab_3d = st.tabs(
        [
            "3A · What the Model Outputs",
            "3B · Walk-Forward",
            "3C · RMSE/MAE Logic",
            "3D · Leakage Buffer",
        ]
    )

    # -----------------------------------------------------------------
    # 3A What the Model Outputs
    # Source: Pipeline Visualized paragraphs 4-6 and 13-19 verbatim.
    # -----------------------------------------------------------------
    with tab_3a:
        st.subheader("The Model Outputs a Predicted 20-day Forward Volatility (σ̂); The Model Does Not Output Portfolio Weights")
        render_figure(
            "pipeline_two_tracks.png",
            caption="The same forecast σ̂ᵢ feeds both the accuracy track (NB03/NB04) "
            "and the portfolio track (NB05). The model never outputs portfolio "
            "weights directly.",
        )
        st.markdown(
            "On every test date, each model outputs **one number per ETF: σ̂ᵢ** — "
            "a predicted 20-day forward volatility. That single output is used in "
            "two completely separate ways simultaneously:"
        )
        st.markdown(
            "**Track 1 — Accuracy (NB03/NB04/NB06).** σ̂ᵢ is compared against the "
            "actual realized σᵢ to compute RMSE and MAE. This describes how "
            "accurate the volatility forecast is."
        )
        st.markdown(
            "**Track 2 — Portfolio weighting (NB05).** Every 21 trading days "
            "(a rebalance event), the same σ̂ᵢ values are plugged into the "
            "inverse-vol formula. Low predicted vol → high weight. High predicted "
            "vol → low weight. These raw weights are then capped, re-normalized, "
            "and scaled."
        )
        st.success(
            "**The most important thing to understand: the model never outputs "
            "portfolio weights.** The model outputs σ̂ᵢ, and NB05 converts those "
            "forecasts into weights."
        )

    # -----------------------------------------------------------------
    # 3B Walk-Forward
    # Source: Pipeline Visualized paragraph 41 verbatim.
    # -----------------------------------------------------------------
    with tab_3b:
        st.subheader("How Walk-Forward Validation Works")
        st.markdown(
            "The test window has **500 trading days, divided into 25 "
            "non-overlapping blocks of 20 days each**. At the start of block 1, "
            "the model is trained on everything up to that point, forecasts the "
            "next 20 days, then stops. At the start of block 2, the model is "
            "retrained on all data up to that point (now 20 days longer), "
            "forecasts the next 20 days, and so on."
        )
        st.markdown(
            "With each successive block, the training window expands, and the "
            "model is refit on the full history available at that time, "
            "ensuring that all forecasts are generated strictly from past "
            "information (without any forward leakage)."
        )
        render_figure(
            "notebook03_acf_diagnostics.png",
            caption="Autocorrelation Function (ACF) diagnostics for daily log "
            "returns. The decaying ACF helps motivate short-horizon volatility "
            "modeling.",
        )

    # -----------------------------------------------------------------
    # 3C RMSE/MAE Logic
    # Source: Pipeline Visualized paragraphs 40 and 44 verbatim.
    # -----------------------------------------------------------------
    with tab_3c:
        # Gray-print definitions stack: RMSE first, MAE second, then aggregation
        # note. st.caption() renders in small gray text, exactly the textbook
        # reference card the reader needs above the explanation. LaTeX renders
        # via KaTeX in the browser — supported identically across Streamlit
        # local, Streamlit Cloud, and Docker.
        st.caption(
            "**RMSE — Root Mean Squared Error.** "
            r"$\text{RMSE} = \sqrt{\dfrac{1}{N}\sum_{i=1}^{N}(\hat{\sigma}_i - \sigma_i)^2}$. "
            "RMSE squares each error before averaging, so large mistakes count "
            "much more than small ones."
        )
        st.caption(
            "**MAE — Mean Absolute Error.** "
            r"$\text{MAE} = \dfrac{1}{N}\sum_{i=1}^{N}|\hat{\sigma}_i - \sigma_i|$. "
            "MAE measures the average size of errors, treating all mistakes "
            "equally."
        )
        st.caption(
            "These errors are averaged across all 14 ETFs and all 500 test "
            "days to produce one overall score per model."
        )

        st.subheader("Why RMSE and MAE Can Disagree")
        st.markdown(
            "On every test date, each model outputs the predicted 20-day "
            "forward realized volatility (σ̂ᵢ) for each of the 14 ETFs. Twenty "
            "trading days later, the actual realized volatility (σᵢ) is "
            "knowable from realized returns. The error for that one forecast "
            "is the difference between predicted and actual values "
            "(σ̂ᵢ − σᵢ)."
        )
        st.markdown(
            "**Why the two metrics can disagree.** RMSE is dominated by the "
            "worst forecasts. If the CAUSAL_XGBOOST model makes smaller errors "
            "most of the time but occasionally produces a large miss during a "
            "stress spike — exactly the kind of event where the constrained "
            "model has less momentum information to lean on — then RMSE rises "
            "while MAE on calm days improves."
        )
        render_figure(
            "notebook03_distribution_diagnostics.png",
            caption="Distribution diagnostics for daily log returns: heavy tails "
            "are visible, which is precisely why RMSE and MAE behave differently "
            "and why both are reported.",
        )
        st.info(
            "The Diebold-Mariano HAC test result (minimum p = 0.328) means the "
            "16% RMSE gap between the CAUSAL_XGBOOST model and the unconstrained "
            "XGBoost model is **not statistically distinguishable from noise**, "
            "which is why H1 is not claimed as confirmed."
        )

    # -----------------------------------------------------------------
    # 3D Leakage Buffer
    # Source: Pipeline Visualized paragraph 43 verbatim + report §3.2 paragraph 419.
    # -----------------------------------------------------------------
    with tab_3d:
        st.subheader("The 20-Day Leakage Buffer")
        st.markdown(
            "The 20-day forward target for date t uses returns from t+1 through "
            "t+20. **If the model were trained on data that included any of "
            "those future returns, the model would be cheating (it would be "
            "learning the answer).**"
        )
        st.markdown(
            "The 20-day buffer between the end of the training set and the start "
            "of each forecast block is what prevents cheating. Every governance "
            "rule in the pipeline (the Fresh-Kernel Reliability Rule, the "
            "Artifact-Truth Rule) ultimately exists to enforce this boundary."
        )
        st.markdown(
            "The target is stored in a dedicated Parquet file "
            "(`data/processed/target_fwd_vol.parquet`) that is **physically "
            "separate from the feature matrix**. The physical separation enforces "
            "a leakage boundary: no downstream modeling notebook can accidentally "
            "include target information as a feature column."
        )
        render_figure(
            "notebook02_vol_feature_vs_target_SPY.png",
            caption="Volatility feature versus 20-day forward realized-vol target "
            "for SPY. The visual contract: features at time t describe the past; "
            "the target at time t describes the future.",
        )


# =====================================================================
# §4 DAG Constraint Experiment
# =====================================================================
elif page == "§4 DAG Constraint Experiment":
    st.title("§4 DAG Constraint Experiment")
    # Source: AI-drafted connective prose; reviewed and approved by Steve.
    st.markdown(
        "*The gate· The 74 features allowed through · The 77 features blocked · "
        "The matched-pair experiment that isolates the gate's effect.*"
    )

    tab_4a, tab_4b, tab_4c, tab_4d = st.tabs(
        [
            "4A · Manual DAG",
            "4B · Allowed Features",
            "4C · Blocked Features",
            "4D · Baseline ↔ Causal Twins",
        ]
    )

    # -----------------------------------------------------------------
    # 4A Manual DAG
    # Source: report §3.5 paragraphs 645-650, 656 verbatim.
    # -----------------------------------------------------------------
    with tab_4a:
        st.caption("**DAG** — Directed Acyclic Graph")
        st.subheader("The DAG as an Information-Flow Contract")
        st.markdown(
            "The manual DAG contains **nine nodes and seven directed arrows**:"
        )
        st.markdown(
            "**Five core information-flow edges:**\n"
            "- Sentiment → Momentum (Sentiment Amplifies Momentum)\n"
            "- Momentum → Returns (Momentum Forecasts Returns)\n"
            "- Value → Returns (Value (HML) Explains Returns)\n"
            "- Volatility → Risk (Volatility Estimates Risk)\n"
            "- Risk → Allocation (Risk Constrains Allocation)\n\n"
            "**Two exogenous conditioning edges:**\n"
            "- MACRO → Risk\n"
            "- REGIME → Risk"
        )
        st.markdown(
            "The DAG constraint is implemented operationally through "
            "**prefix-based feature gating**. Each modeling stage receives only "
            "the feature columns whose DAG-node prefixes appear in the "
            "allowed-parent set for that stage."
        )
        st.markdown(
            "The double-underscore naming convention established in NB02 enables "
            "programmatic prefix parsing: splitting each column name on the "
            "double-underscore delimiter yields the DAG node as the first token. "
            "This is implemented in NB04 by selecting feature columns whose "
            "names start with the permitted DAG-node prefixes for each stage."
        )

    # -----------------------------------------------------------------
    # 4B Allowed Features
    # Source: outline §3.4 + Pipeline Visualized paragraph 26.
    # -----------------------------------------------------------------
    with tab_4b:
        st.subheader("74 Features Pass the Gate")
        render_figure(
            "dag_gate_inventory.png",
            caption="The 74 features allowed through the DAG risk gate at NB04 "
            "(top row, teal) versus the 77 features blocked at the gate (bottom "
            "rows, gray plus the coral SENT family).",
        )
        st.markdown(
            "Three families pass straight through into the DAG-constrained NB04 "
            "models. The DAG says these are legitimate direct parents of the "
            "Risk node: current volatility, macroeconomic conditions, and regime "
            "state are all economically defensible reasons why forward risk "
            "should be high or low."
        )
        st.markdown(
            "- **VOL__** (70 features) — direct parent of the Risk node; "
            "realized-volatility and EWMA-volatility signals.\n"
            "- **MACRO__** (3 features) — exogenous conditioners of the risk "
            "environment (VIXCLS, T10Y2Y, DTB3).\n"
            "- **REGIME__** (1 feature) — binary VIX-regime indicator that "
            "conditions volatility dynamics at the risk stage."
        )

    # -----------------------------------------------------------------
    # 4C Blocked Features
    # Source: outline §3.4 + Pipeline Visualized paragraph 27.
    # -----------------------------------------------------------------
    with tab_4c:
        st.subheader("77 Features Are Blocked at the Gate")
        st.markdown(
            "Three feature families plus the coral sentiment feature are "
            "available to the unconstrained NB03 models but are **hard-blocked "
            "in NB04**. The DAG says momentum, value factor loadings, and PCA "
            "scores belong upstream — they influence returns, not risk directly."
        )
        st.markdown(
            "- **MOM__** (56 features) — cumulative-return features across "
            "5/10/21/63-day horizons; upstream of Returns, not Risk.\n"
            "- **VAL__** (18 features) — rolling HML beta and factor-control "
            "features; Value explains Returns, not direct Risk.\n"
            "- **ML__** (3 features) — rolling PCA latent factors; no direct "
            "DAG parent role at the Risk node.\n"
            "- **SENT__** (1 feature) — narrative signal flows through Momentum "
            "rather than directly into Risk."
        )
        st.markdown(
            "The unconstrained XGBoost actually learns that momentum features "
            "are highly predictive of volatility (XLF's 10-day return ends up "
            "as one of its top features for predicting SPY volatility), and "
            "**that is precisely the problem the DAG is designed to prevent**. "
            "That cross-factor shortcut works in-sample but violates economic "
            "logic about how volatility is generated."
        )

    # -----------------------------------------------------------------
    # 4D Baseline ↔ Causal Twins
    # Source: Pipeline Visualized paragraphs 29-37 verbatim.
    # -----------------------------------------------------------------
    with tab_4d:
        st.subheader("Matched-Pair Experiment Design")
        render_figure(
            "baseline_causal_twins.png",
            caption="Three matched architecture pairs: XGBOOST ↔ CAUSAL_XGBOOST, "
            "BASELINE_MLP ↔ CAUSAL_MLP, BASELINE_LSTM ↔ CAUSAL_LSTM. EWMA stands "
            "alone as the minimum credible benchmark.",
        )
        st.markdown(
            "**The one and only difference between each baseline and its causal "
            "twin is the feature set.** Hyperparameters, architecture, training "
            "windows, and walk-forward splits are frozen identical. The DAG gate "
            "removes 77 features before the model ever sees the data."
        )
        st.success(
            "**Any RMSE / MAE difference between the XGBOOST model and the "
            "CAUSAL_XGBOOST model is therefore attributable solely to the "
            "feature restriction — nothing else changed.** The distinction is "
            "deliberately minimal by design, and that minimalism is the whole "
            "point of the experimental setup."
        )
        st.markdown(
            "**The EWMA model stands alone.** It has no causal twin because "
            "it has no features and requires no training. It is a formula "
            "applied directly to past squared returns. Its role in the "
            "pipeline is as the minimum credible benchmark — if a "
            "sophisticated ML model cannot beat the EWMA model, it has not "
            "justified its complexity."
        )


# =====================================================================
# §5 Results & Interpretability
# =====================================================================
elif page == "§5 Results & Interpretability":
    st.title("§5 Results & Interpretability")
    # Source: AI-drafted connective prose; reviewed and approved by Steve.
    st.markdown(
        "*Forecast accuracy results, · Interpretability via Shannon entropy · "
        "Feature-importance evidence · The three hypothesis verdicts plus "
        "one secondary interpretability diagnostic.*"
    )

    tab_5a, tab_5b, tab_5c, tab_5d = st.tabs(
        [
            "5A · Forecast Accuracy",
            "5B · Entropy Diagnostics",
            "5C · Feature Importance",
            "5D · Hypothesis Verdicts",
        ]
    )

    # -----------------------------------------------------------------
    # 5A Forecast Accuracy
    # Source: report §4.2 paragraphs 1073-1076 verbatim.
    # -----------------------------------------------------------------
    with tab_5a:
        st.subheader("Causal vs Baseline RMSE and MAE")
        st.markdown(
            "Forecast accuracy of the causal pipeline relative to unconstrained "
            "baselines, measured over **500 test-block observations per "
            "ticker** (25 walk-forward blocks of 20 trading days each)."
        )
        render_table(
            "notebook04_causal_vs_baseline_rmse_mae.csv",
            caption="RMSE and MAE, causal vs baseline models, per ticker and "
            "aggregate.",
        )
        render_figure(
            "notebook04_causal_vs_baseline_rmse_delta.png",
            caption="Per-ticker RMSE delta (causal minus baseline). Positive "
            "bars indicate tickers where the constrained model has higher RMSE "
            "than the unconstrained baseline.",
        )
        st.markdown(
            "The aggregate test metrics (equal-weight mean across all 14 tickers) "
            "for the constrained model are **RMSE = 0.0811 and MAE = 0.0501**. "
            "The unconstrained XGBoost baseline achieves RMSE = 0.0698 and "
            "MAE = 0.0491; the EWMA benchmark achieves RMSE = 0.0708 and "
            "MAE = 0.0483."
        )
        st.markdown(
            "The constrained model **wins 0 of 14 individual-ticker RMSE "
            "comparisons** against the unconstrained XGBoost baseline. Against "
            "EWMA, the constrained model wins 5 of 14 tickers on RMSE: DBC, "
            "GLD, QQQ, SPY, and XLK. On MAE, the constrained model improves "
            "for 6 of 14 tickers versus XGBoost (EEM, GLD, IWM, QQQ, XLK, XLV)."
        )

    # -----------------------------------------------------------------
    # 5B Entropy Diagnostics
    # Source: report §1.4 paragraph 20 + report §5.1 paragraph 1595 verbatim.
    # -----------------------------------------------------------------
    with tab_5b:
        st.subheader("Two-Tier Shannon Entropy Diagnostics")
        st.markdown(
            "**Feature-importance entropy** (Tier 1, NB04): normalized Shannon "
            "entropy of XGBoost gain shares. Lower entropy means the model "
            "concentrates predictive weight on fewer features; higher entropy "
            "means more balanced reliance across features."
        )
        render_table(
            "notebook04_entropy_comparison.csv",
            caption="Normalized feature-importance entropy per ticker — "
            "constrained vs unconstrained.",
        )
        render_figure(
            "notebook04_entropy_comparison.png",
            caption="Causal vs baseline feature-importance entropy comparison.",
        )
        st.markdown(
            "Normalized Shannon entropy of XGBoost gain shares **increases under "
            "DAG constraints for all three representative tickers**: GLD "
            "(+0.003), SPY (+0.012), and TLT (+0.031)."
        )
        st.divider()
        st.markdown(
            "**Factor-exposure entropy** (Tier 2, NB06): computed from rolling "
            "Fama-French factor-loading regressions across all 24 rebalancing "
            "segments. A portfolio-level interpretability measure."
        )
        render_figure(
            "notebook06_factor_exposure_entropy.png",
            caption="Factor-exposure entropy across rebalancing segments.",
        )
        st.markdown(
            "**CAUSAL_XGBOOST achieves the highest mean factor-exposure entropy "
            "across all 24 rebalancing segments** (0.7933 vs 0.7894 for EWMA "
            "and 0.7853 for XGBOOST, a +0.008 improvement over the unconstrained "
            "baseline). Both tiers point in the same direction: DAG prefix "
            "gating produces a more uniformly distributed information-loading "
            "structure."
        )
        st.info(
            "**Throughout the capstone, entropy is interpreted as a dispersion "
            "measure rather than an indicator of economic optimality.** An "
            "increase in entropy signals reduced concentration in a single "
            "dominant feature, not guaranteed improvement in risk-adjusted "
            "performance."
        )

    # -----------------------------------------------------------------
    # 5C Feature Importance
    # Source: report §4.2 paragraph 1085 verbatim + dashboard NB04 artifact.
    # -----------------------------------------------------------------
    with tab_5c:
        st.subheader("Constrained Feature Importance — DAG Prefix Purity")
        render_figure(
            "notebook04_causal_feature_importance_SPY.png",
            caption="SPY feature importance under the DAG-constrained model. "
            "Only VOL__, MACRO__, and REGIME__ prefixes appear.",
        )
        st.markdown(
            "Feature importance analysis confirms that **the DAG gating "
            "mechanism operates correctly at the model level**. For all three "
            "representative tickers (SPY, TLT, GLD), the constrained model's "
            "feature importance table contains only VOL-prefixed, MACRO-prefixed, "
            "and REGIME-prefixed features."
        )
        st.success(
            "**Zero features from the forbidden prefix set (MOM, VAL, ML, SENT) "
            "appear in the constrained importance output**, providing "
            "artifact-level proof that the DAG constraint held through model "
            "training."
        )

    # -----------------------------------------------------------------
    # 5D Hypothesis Verdicts
    # Source: report §5.1 paragraphs 1592-1595 verbatim
    #         (with corrected H1 wording).
    # -----------------------------------------------------------------
    with tab_5d:
        st.subheader("Formal Hypothesis Verdicts")

        st.markdown("#### H1 (Forecast Accuracy) — Not Confirmed")
        st.error(
            "The DAG-constrained model produced higher aggregate RMSE (0.0811) "
            "than both the unconstrained XGBoost baseline (0.0698, +16.2%) and "
            "the EWMA naive forecast (0.0708, +14.5%); the constrained model "
            "won 0 of 14 per-ticker RMSE comparisons, and Diebold-Mariano HAC "
            "tests found no accuracy difference statistically distinguishable "
            "from noise (min p = 0.328)."
        )
        st.markdown(
            "The accuracy cost is therefore real in point-estimate terms but "
            "indistinguishable from sampling variation over the 500-observation "
            "test window with 20-day overlapping forecast horizons. The "
            "seven-model portfolio expansion in NB05 does not alter this "
            "verdict."
        )

        st.divider()
        st.markdown("#### H2 (Portfolio Risk Control) — Partially Confirmed and Family-Specific")
        st.warning(
            "The DAG constraint improves tail-risk control most clearly in the "
            "tree family and partially regularizes the neural families, but the "
            "strongest full-period Sharpe in NB05 belongs to the unconstrained "
            "LSTM."
        )
        st.markdown(
            "**Tree family:** CAUSAL_XGBOOST is the most attractive risk-adjusted "
            "tree portfolio — Sharpe 0.667, maximum drawdown −5.83%, and Calmar "
            "1.498 versus 0.583 / −7.38% / 1.174 for EWMA and 0.567 / −6.03% / "
            "1.343 for unconstrained XGBOOST."
        )
        st.markdown(
            "**Neural family:** the evidence is mixed but informative. "
            "CAUSAL_LSTM lowers turnover and improves risk-targeting error "
            "relative to BASELINE_LSTM, while CAUSAL_MLP modestly improves "
            "Sharpe and final equity relative to BASELINE_MLP."
        )
        st.markdown(
            "**Outright winner on Sharpe:** BASELINE_LSTM achieves the highest "
            "full-period Sharpe ratio (0.777) and the highest annualized return "
            "(11.40%), but with deeper drawdown (−9.10%) and the highest "
            "transaction-cost burden among the stronger models."
        )

        st.divider()
        st.markdown("#### H3 (NLP Sentiment Contribution) — Directionally Supported")
        st.warning(
            "A matched-window evaluation in NB03 trained both a sentiment-"
            "inclusive model (XGBOOST_H3_WITH_SENT) and a sentiment-excluded "
            "model (XGBOOST_H3_NO_SENT) on the identical post-2020 window "
            "(1,496 rows, 300 test observations per ticker). The aggregate RMSE "
            "delta is −0.000324 (WITH_SENT 0.10016 versus NO_SENT 0.10049)."
        )
        st.markdown(
            "The direction is consistent with H3: sentiment features reduce "
            "forecast error on a fair matched-window comparison. The effect "
            "size is small (0.32% relative RMSE improvement). **No "
            "Diebold-Mariano HAC test was applied to the H3 comparison; "
            "statistical significance cannot be claimed without that formal "
            "inference step.** H3 is characterized as directionally supported "
            "rather than confirmed."
        )

        st.divider()
        st.markdown("#### Secondary Interpretability Diagnostic — Consistent at Both Tiers")
        st.success(
            "Normalized Shannon entropy of XGBoost gain shares rose under DAG "
            "constraints for all three representative tickers (GLD +0.003, "
            "SPY +0.012, TLT +0.031); factor-exposure entropy from rolling "
            "Fama-French regressions is highest for CAUSAL_XGBOOST (0.7933 vs "
            "0.7894 for EWMA and 0.7853 for unconstrained XGBoost)."
        )
        st.markdown(
            "Both tiers point in the same direction: DAG prefix gating produces "
            "a more uniformly distributed information-loading structure, whether "
            "measured at the feature-selection stage or at the realized portfolio "
            "factor-loading stage."
        )


# =====================================================================
# §6 Portfolio & Stress Testing
# =====================================================================
elif page == "§6 Portfolio & Stress Testing":
    st.title("§6 Portfolio & Stress Testing")
    # Source: AI-drafted connective prose; reviewed and approved by Steve.
    st.markdown(
        "*The inverse-vol weighting logic · The equity curves · The drawdowns · "
        "The regime-conditional plus stress-episode evidence.*"
    )

    tab_6a, tab_6b, tab_6c, tab_6d = st.tabs(
        [
            "6A · Inverse-Vol Weights",
            "6B · Equity Curves",
            "6C · Drawdowns",
            "6D · Regime & Stress",
        ]
    )

    # -----------------------------------------------------------------
    # 6A Inverse-Vol Weights
    # Source: Pipeline Visualized paragraphs 48-55 verbatim.
    # -----------------------------------------------------------------
    with tab_6a:
        st.subheader("How Predicted Volatility Becomes Portfolio Weight")
        st.markdown(
            "**The core intuition:** if an ETF's predicted volatility is low, "
            "the pipeline gives it a larger share of the portfolio (because a "
            "low-vol asset can absorb more capital while still contributing the "
            "same amount of risk). If predicted volatility is high, the weight "
            "shrinks. The inverse-volatility weight formula enforces this "
            "mechanically."
        )
        st.markdown(
            "**At each 21-day rebalance, NB05 executes five steps:**"
        )
        st.markdown(
            "**Step 1 — Raw inverse-vol weights.** For each ETF i, compute "
            "raw_weightᵢ = (0.10 ÷ σ̂ᵢ) × (1/14). The 0.10 is the 10% "
            "annualized vol target. The 1/14 distributes the risk budget equally "
            "across the 14 ETFs as a starting point. An ETF forecasted at 10% "
            "vol gets weight 0.0714. One forecasted at 5% vol gets 0.143. One "
            "forecasted at 20% vol gets 0.036."
        )
        st.markdown(
            "**Step 2 — Single-asset caps.** Each weight is clipped to [0.02, "
            "1.0]. The floor (0.02) prevents short positions; the ceiling "
            "prevents a single ETF from consuming the whole portfolio."
        )
        st.markdown(
            "**Step 3 — Risky-weight renormalization.** If the sum of capped "
            "weights exceeds 1.0 (meaning the raw formula wanted to invest more "
            "than 100% of capital), the weights are divided by their sum to "
            "bring the total risky allocation back to exactly 1.0."
        )
        st.markdown(
            "**Step 4 — Portfolio-level scaler.** A scaler is computed so that "
            "the resulting portfolio's ex-ante volatility (estimated via the "
            "Ledoit-Wolf covariance matrix) hits the 10% target. The code caps "
            "this at 1.0 (no leverage)."
        )
        st.markdown(
            "**Step 5 — Cash sleeve.** Whatever risky weight is not used "
            "(1.0 minus the sum of final risky weights) sits in cash."
        )
        st.info(
            "**The pipeline never levers up.** When the scaler would push risky "
            "weights above 100%, it is capped at 1.0 and the excess sits in "
            "cash. A constrained model that forecasts slightly higher "
            "volatility (as the CAUSAL_XGBOOST model does) will "
            "systematically tilt toward a smaller risky sleeve and a larger "
            "cash position. **That conservatism is exactly why H2 is "
            "partially confirmed even though H1 is not.**"
        )

    # -----------------------------------------------------------------
    # 6B Equity Curves
    # Source: existing dashboard captions, verified against NB05 figures.
    # -----------------------------------------------------------------
    with tab_6b:
        st.subheader("Walk-Forward Portfolio Performance")
        st.markdown(
            "Walk-forward backtest over **499 trading days** (2023-12-29 through "
            "2025-12-31). Equal-risk-budget inverse-volatility weighting, 10% "
            "annualized vol target, Ledoit-Wolf covariance shrinkage, monthly "
            "rebalancing, 10 bps transaction costs."
        )
        render_table(
            "notebook05_portfolio_performance.csv",
            caption="Risk and return metrics per model.",
        )
        render_figure(
            "notebook05_equity_curves.png",
            caption="Equity curves — top panel baselines, bottom panel causal "
            "models.",
        )
        # Source: AI-drafted connective prose; reviewed and approved by Steve.
        st.markdown(
            "**Reading the equity curves:** the BASELINE_LSTM model produces "
            "the strongest cumulative growth in the baseline panel; the "
            "CAUSAL_LSTM model and the CAUSAL_XGBOOST model track each other "
            "closely in the causal panel. Architecture choice matters more "
            "than DAG gating for raw growth."
        )

    # -----------------------------------------------------------------
    # 6C Drawdowns
    # Source: existing dashboard captions + report §5.1 paragraph 1593 fact.
    # -----------------------------------------------------------------
    with tab_6c:
        st.subheader("Drawdown Comparison")
        render_figure(
            "notebook05_drawdown_comparison.png",
            caption="Drawdown comparison — top panel baselines, bottom panel "
            "causal models.",
        )
        st.markdown(
            "**Maximum drawdown** answers: what was the worst peak-to-trough "
            "loss this portfolio ever experienced? If a portfolio peaked at "
            "$1.00, then fell to $0.942 before recovering, max drawdown = "
            "−5.8%. Max drawdown captures only the single worst episode."
        )
        st.success(
            "**Within the primary tree comparison: the CAUSAL_XGBOOST model "
            "achieves the shallowest maximum drawdown (−5.83%) versus −6.03% "
            "for the unconstrained XGBoost baseline model and −7.38% for the "
            "EWMA model.** The Calmar ratio (annualized return ÷ |max "
            "drawdown|) reaches 1.498 for the CAUSAL_XGBOOST model, the "
            "highest in the tree family."
        )

    # -----------------------------------------------------------------
    # 6D Regime & Stress
    # Source: existing dashboard captions + Pipeline Visualized
    # paragraph 55 (verbatim takeaway).
    # -----------------------------------------------------------------
    with tab_6d:
        st.subheader("Regime-Conditional and Stress-Episode Evidence")
        tab_regime, tab_stress = st.tabs(
            ["Regime-Conditional Performance", "April 2025 Stress Episode"]
        )

        with tab_regime:
            render_table(
                "notebook06_regime_portfolio_metrics.csv",
                caption="Portfolio metrics conditional on VIX regime.",
            )
            render_figure(
                "notebook06_regime_portfolio_comparison.png",
                caption="Regime-conditional Sharpe ratios and maximum drawdowns.",
            )
            # Source: AI-drafted connective prose; reviewed and approved by Steve.
            st.markdown(
                "All three models — the EWMA model, the XGBOOST model, and "
                "the CAUSAL_XGBOOST model — produce strongly positive Sharpe "
                "ratios in low-VIX regimes and strongly negative Sharpe "
                "ratios in high-VIX regimes; **the CAUSAL_XGBOOST model "
                "achieves the smallest HIGH_VIX maximum drawdown (−9.29%) of "
                "the three.**"
            )

        with tab_stress:
            render_table(
                "notebook06_stress_drawdown.csv",
                caption="April 2025 tariff-shock episode — drawdown summary.",
            )
            render_figure(
                "notebook06_stress_drawdown.png",
                caption="April 2025 stress-episode drawdown by model.",
            )
            st.markdown(
                "**Within the primary tree comparison, the EWMA model "
                "declines to −7.38%, the XGBOOST model to −6.03%, and the "
                "CAUSAL_XGBOOST model to −5.83%** during the April 2025 "
                "tariff-shock episode."
            )
            st.success(
                "**Central pattern of the tree-model comparison, stated "
                "plainly:** the CAUSAL_XGBOOST model produced 16.2% higher "
                "aggregate RMSE than the unconstrained XGBOOST model, yet "
                "delivered the shallowest drawdown (−5.83%) and the highest "
                "Calmar ratio (1.498) in the tree family. Forecast accuracy "
                "and portfolio risk control move together less than one might "
                "expect; in this experiment, the DAG constraint appears to "
                "support stronger tree-model portfolio risk control at the "
                "cost of raw point-forecast accuracy."
            )


# =====================================================================
# §7 Engineering & Deployment
# =====================================================================
elif page == "§7 Engineering & Deployment":
    st.title("§7 Engineering & Deployment")
    # Source: AI-drafted connective prose; reviewed and approved by Steve.
    st.markdown(
        "*The riskml/ Python package · The CI test suite · The Docker container · "
        "Azure Container Apps deployment.*"
    )

    tab_7a, tab_7b, tab_7c, tab_7d = st.tabs(
        [
            "7A · Package Extraction",
            "7B · CI Tests",
            "7C · Docker",
            "7D · Microsoft Azure",
        ]
    )

    # -----------------------------------------------------------------
    # 7A Package Extraction
    # Source: AI-drafted connective prose; reviewed and approved by Steve.
    # -----------------------------------------------------------------
    with tab_7a:
        st.subheader("riskml/ · From Notebook Research to Reusable Package")
        st.markdown(
            "Research in this capstone happened in notebooks (NB01 through "
            "NB07).<br>"
            "Production code lives in the **riskml/** Python package "
            "(an installable, importable, testable layer that mirrors the "
            "DAG's subgraph structure).",
            unsafe_allow_html=True,
        )
        st.markdown(
            "The **riskml/** package contains two extracted functions, "
            "one from each DAG subgraph:"
        )
        st.markdown(
            "- **`riskml/etl/market_data.py`** — `extract_price_panel()` and "
            "`download_etf_prices()`, extracted from NB01.\n"
            "- **`riskml/features/transforms.py`** — `compute_momentum_features()` "
            "(return-generation subgraph, Momentum node) and "
            "`compute_realized_volatility()` (risk-allocation subgraph, "
            "Volatility node, feeds NB04).\n"
            "- **`riskml/storage.py`** — production-aware data loader supporting "
            "Azure Blob Storage via env var `AZURE_STORAGE_CONNECTION_STRING` "
            "with automatic local fallback."
        )
        st.info(
            "Each DAG subgraph contributes one feature function: momentum "
            "from the return-generation path, and realized volatility from "
            "the risk-allocation path that feeds NB04.<br>"
            "These functions form the testable core of the feature pipeline.<br>"
            "The dashboard reads notebook artifacts directly.<br>"
            "The other feature families remain in notebook form."
        )

    # -----------------------------------------------------------------
    # 7B CI Tests
    # Source: AI-drafted connective prose; reviewed and approved by Steve.
    # -----------------------------------------------------------------
    with tab_7b:
        st.subheader("Continuous Integration — GitHub Actions")
        st.markdown(
            "Every push to `main` triggers a CI workflow that runs "
            "**14 deterministic tests** via `pytest`, organized into "
            "three categories: **smoke tests** (catastrophic-failure "
            "detection), **pipeline tests** (function-level correctness "
            "of the `riskml/` package), and **invariant tests** "
            "(project-wide constants and architectural assertions "
            "documented in the capstone report)."
        )
        st.markdown(
            "The workflow file lives at `.github/workflows/ci.yml`. "
            "Each run checks out the repository, sets up Python 3.11, "
            "installs the package and its dependencies, runs Ruff for "
            "code-style enforcement, executes `pytest`, and finally "
            "runs a smoke import to confirm the package is reachable "
            "from a clean environment."
        )
       
        tab_smoke, tab_pipeline, tab_invariants = st.tabs(
            ["Smoke Tests", "Pipeline Tests", "Invariant Tests"]
        )

        # --------------------------------------------------
        # Smoke Tests sub-tab — text only, no SVGs.
        # --------------------------------------------------
        with tab_smoke:
            st.markdown(
                "Smoke tests are the most basic CI checks. Smoke tests "
                "catch catastrophic failures (the package is broken, "
                "the wrong Python version is installed) without "
                "exercising any specific feature. Smoke tests run first "
                "in the CI pipeline; if a smoke test fails, no other "
                "tests run because the package itself is unimportable. "
                "Two tests live in this category."
            )
            st.markdown(
                "**Smoke Test #1 — Riskml Import function.** Smoke "
                "Test #1 asserts that `import riskml` succeeds and "
                "returns a non-`None` module object. The smallest "
                "possible CI signal: if this test fails, every "
                "downstream test would fail with the same import "
                "error, so failing fast here saves time "
                "and unambiguously surfaces the root cause."
            )
            st.markdown(
                "**Smoke Test #2 — Python Version function.** Smoke "
                "Test #2 asserts that the Python interpreter on the "
                "Ubuntu Linux VM runner reports version 3.11 or newer. "
                "Smoke Test #2 guards against environment drift where "
                "the CI image silently upgrades or downgrades the "
                "interpreter mid-project, which would shift the "
                "installed library versions and could change "
                "numerical results."
            )
            st.caption(
                "No figures are rendered for the smoke tests — the "
                "tests themselves are too small to warrant illustration."
            )

        # --------------------------------------------------
        # Pipeline Tests sub-tab — 5 tests, 1 SVG each.
        # --------------------------------------------------
        with tab_pipeline:
            st.markdown(
                "Pipeline tests exercise the actual Python functions "
                "inside the `riskml/` package. Each pipeline test "
                "hands the function synthetic (but realistic) input "
                "data, calls the function, and asserts the output is "
                "shaped, named, and numerically scaled as expected. "
                "Five tests live in this category, each exercising one "
                "extracted function from the riskml package."
            )
            st.divider()

            # === Pipeline Test #1 ===
            st.markdown(
                "##### Pipeline Test #1 — Storage Round-Trip Local Filesystem Fallback"
            )
            st.caption("Storage Local Fallback function")
            render_figure(
                "tests/pipeline/1_test_storage_local_fallback"
                "/3_round_trip_flow.svg"
            )
            st.markdown(
                "Pipeline Test #1 verifies that the storage function "
                "correctly falls back to local-filesystem reads when "
                "no Azure Blob Storage connection string is set in the "
                "environment — the exact configuration that both the "
                "laptop and the GitHub Actions runner use. The pytest "
                "`tmp_path` fixture provides a hermetic temporary "
                "directory unique to this test, so no shared state "
                "can leak in or out. *Anyone refactoring the Azure "
                "cloud path could silently break the local-fallback "
                "branch, and Pipeline Test #1 catches that the moment "
                "the break happens.*"
            )
            st.divider()

            # === Pipeline Test #2 ===
            st.markdown(
                "##### Pipeline Test #2 — Price Panel Extraction from yfinance MultiIndex"
            )
            st.caption("Extract Price Panel MultiIndex function")
            render_figure(
                "tests/pipeline/2_test_extract_price_panel_multiindex"
                "/2_raw_dataframe_with_multiindex.svg"
            )
            st.markdown(
                "Pipeline Test #2 verifies that the price-panel "
                "extraction function correctly peels apart yfinance's "
                "two-level MultiIndex column structure (price-type at "
                "the outer level, ticker at the inner level) into a "
                "flat ticker-by-date panel. The test feeds in a "
                "realistic five-business-day input with both Adjusted "
                "Close and Volume columns for SPY and QQQ, then "
                "asserts the output contains only the two "
                "adjusted-close columns with no missing values. *If "
                "anyone broke the level-zero selection inside the "
                "function, the panel would arrive with Volume "
                "columns leaked in or with the ticker level still "
                "nested.*"
            )
            st.divider()

            # === Pipeline Test #3 ===
            st.markdown(
                "##### Pipeline Test #3 — Momentum Features Shape, Naming, and Warmup"
            )
            st.caption("Compute Momentum Features function")
            render_figure(
                "tests/pipeline/3_test_compute_momentum_features"
                "/3_momentum_output_structure.svg"
            )
            st.markdown(
                "Pipeline Test #3 verifies that the momentum-features "
                "function produces deterministic outputs (seed 692) "
                "with the DAG-aligned `MOM__<ticker>__cumret__<N>d` "
                "naming convention, the right shape across both "
                "rolling windows (five-day and twenty-one-day), and "
                "finite values past the warmup period. The first four "
                "rows must be `NaN` because the five-day window "
                "cannot fill them; rows twenty-one onward must be "
                "entirely non-`NaN`. *Drift in the column-naming "
                "convention would silently break every downstream DAG "
                "prefix-gating operation in NB04.*"
            )
            st.divider()

            # === Pipeline Test #4 ===
            st.markdown(
                "##### Pipeline Test #4 — Defensive Contract Empty Input Rejection"
            )
            st.caption("Compute Momentum Features Rejects Empty function")
            render_figure(
                "tests/pipeline/4_test_compute_momentum_features_rejects_empty"
                "/3_defensive_vs_silent.svg"
            )
            st.markdown(
                "Pipeline Test #4 verifies that the momentum-features "
                "function raises a `ValueError` with the substring "
                "`Empty` in the error message when given an empty "
                "DataFrame, rather than silently returning an empty "
                "result. Defensive engineering rule: a silent return "
                "on garbage input is the kind of bug that surfaces a "
                "hundred lines downstream, far from the bug's origin. "
                "*Pipeline Test #4 makes silent failure structurally "
                "impossible — the function must speak up at the moment "
                "of the offense.*"
            )
            st.divider()

            # === Pipeline Test #5 ===
            st.markdown(
                "##### Pipeline Test #5 — Realized Volatility Annualization Sanity Check"
            )
            st.caption("Compute Realized Volatility function")
            render_figure(
                "tests/pipeline/5_test_compute_realized_volatility"
                "/3_annualization_sanity_check.svg"
            )
            st.markdown(
                "Pipeline Test #5 verifies that the realized-volatility "
                "function produces correctly-shaped outputs with the "
                "DAG-aligned `VOL__<ticker>__rvol__<N>d` naming, and "
                "that annualization is correct: with daily sigma "
                "approximately 0.01, the annualized output must fall "
                "in the 0.10 to 0.25 range (the expected √252 "
                "scaling). *If anyone forgets the annualization "
                "factor, the output lands near 0.01 instead of 0.159 "
                "and Pipeline Test #5 fails on the next push.*"
            )

        # --------------------------------------------------
        # Invariant Tests sub-tab — 1 shared helper + 7 leads.
        # --------------------------------------------------
        with tab_invariants:
            st.markdown(
                "Invariant tests do not exercise executable code. "
                "Invariant tests lock in the project-wide constants "
                "and architectural assertions documented in the M6 "
                "capstone report. Each invariant test reads either a "
                "committed notebook source file (as JSON) or a "
                "committed CSV artifact, then asserts that a specific "
                "constant appears with a specific value. Seven tests "
                "live in this category."
            )
            st.divider()

            # === Shared helper introduction (rendered once at the top) ===
            st.markdown(
                "##### How the invariant tests read notebook source code"
            )
            st.caption("Read Notebook Source helper function")
            render_figure(
                "tests/invariants/1_test_random_seed_is_locked_at_692_across_all_notebooks"
                "/1_read_notebook_source_helper.svg"
            )
            st.markdown(
                "Six of the seven invariant tests share a common "
                "helper function. The helper opens a Jupyter notebook "
                "(which is internally JSON), concatenates every "
                "code-cell source into one long string, then returns "
                "that string for the test to search. Each invariant "
                "test is, therefore, a regression guard around one "
                "specific sentence in the capstone report — if the "
                "report says \"random seed 692\" anywhere, an "
                "invariant test enforces the claim; if the report "
                "says \"nine nodes and seven edges,\" an invariant "
                "test enforces the claim."
            )
            st.divider()

            # === Invariant Test #1 ===
            st.markdown(
                "##### Invariant Test #1 — Random Seed Locked at 692 Across All Notebooks"
            )
            st.caption("Random Seed Is Locked At 692 Across All Notebooks function")
            render_figure(
                "tests/invariants/1_test_random_seed_is_locked_at_692_across_all_notebooks"
                "/2_test_random_seed_regex_anatomy_and_loop.svg"
            )
            st.markdown(
                "Invariant Test #1 verifies that every notebook from "
                "NB01 through NB06 declares `RANDOM_SEED = 692` and "
                "nothing else. Invariant Test #1 compiles a regex "
                "pattern that captures the integer to the right of "
                "the equals sign, loops across all six notebooks, "
                "and asserts every captured value equals the literal "
                "string `'692'`. *Reproducibility is the anchor of "
                "the entire study — a different seed in any notebook "
                "would invalidate every empirical claim downstream.*"
            )
            st.divider()

            # === Invariant Test #2 ===
            st.markdown(
                "##### Invariant Test #2 — Five Core DAG Edges Encoded in NB04"
            )
            st.caption("Causal Edges Dict Encodes Five Core Directed Edges function")
            render_figure(
                "tests/invariants/2_test_causal_edges_dict_encodes_five_core_directed_edges"
                "/2_test_causal_edges_dag_architecture.svg"
            )
            st.markdown(
                "Invariant Test #2 verifies that NB04's `CAUSAL_EDGES` "
                "dictionary literally contains all five core "
                "information-flow edges of the manual DAG: "
                "Sentiment→Momentum, Momentum→Returns, "
                "Value→Returns, Volatility→Risk, and Risk→Allocation. "
                "Invariant Test #2 searches the notebook source for "
                "each edge literal as a substring. *If a future edit "
                "silently removed one of these edges, the DAG figure "
                "would drift from the report's Table 4 caption, and "
                "Invariant Test #2 catches the drift before the next "
                "commit hits `main`.*"
            )
            st.divider()

            # === Invariant Test #3 ===
            st.markdown(
                "##### Invariant Test #3 — Nine-Node, Seven-Edge DAG Figure Architecture"
            )
            st.caption("DAG Figure Has Nine Nodes And Seven Directed Edges function")
            render_figure(
                "tests/invariants/3_test_dag_figure_has_nine_nodes_and_seven_directed_edges"
                "/2_test_dag_figure_complete_architecture.svg"
            )
            st.markdown(
                "Invariant Test #3 verifies that the DAG figure in "
                "NB04 declares all nine nodes (Sentiment, Momentum, "
                "Returns, Value, Volatility, Risk, Allocation, Macro, "
                "Regime) and all seven directed edges (the five core "
                "edges from Invariant Test #2 plus Macro→Risk and "
                "Regime→Risk). The capstone report and the figure "
                "caption explicitly claim these counts. *Any silent "
                "drift between the figure code and the report's prose "
                "claim would surface immediately in Invariant Test #3, "
                "not after the dashboard reaches production.*"
            )
            st.divider()

            # === Invariant Test #4 ===
            st.markdown(
                "##### Invariant Test #4 — Risk-Stage Gating Allow-List and Forbid-List"
            )
            st.caption("Risk Stage Gating Lists Match Report Specification function")
            render_figure(
                "tests/invariants/4_test_risk_stage_gating_lists_match_report_specification"
                "/3_risk_stage_belt_and_suspenders.svg"
            )
            st.markdown(
                "Invariant Test #4 verifies that NB04's risk model "
                "accepts only the `VOL__`, `MACRO__`, and `REGIME__` "
                "prefixes (the allow-list) and explicitly forbids "
                "`MOM__`, `VAL__`, `ML__`, and `SENT__` (the "
                "forbid-list). Invariant Test #4 asserts both Python "
                "list literals exist in the notebook source with "
                "exactly those members. *This is the central causal "
                "claim of the entire project — if a future "
                "feature-engineering edit slipped a `MOM__` feature "
                "past the gate, the H1 hypothesis result would be "
                "invalidated, and Invariant Test #4 catches the slip "
                "immediately.*"
            )
            st.divider()

            # === Invariant Test #5 ===
            st.markdown(
                "##### Invariant Test #5 — 74 Features Retained After Risk-Stage Gating"
            )
            st.caption("Risk Stage Feature Count Equals 74 function")
            render_figure(
                "tests/invariants/5_test_risk_stage_feature_count_equals_74"
                "/1_gating_summary_csv_structure.svg"
            )
            st.markdown(
                "Invariant Test #5 verifies that the committed "
                "gating-summary CSV at "
                "`reports/tables/notebook04_dag_gating_summary.csv` "
                "lists exactly 74 features with "
                "`allowed_for_risk == True`. Invariant Test #5 reads "
                "the CSV, coerces the boolean column to a clean "
                "`True`/`False` representation regardless of dtype "
                "quirks, then sums the trues. *The 74-feature count "
                "is a frozen empirical constant of the capstone — a "
                "silent change here would mean either the gating "
                "logic or the feature inventory has drifted out of "
                "sync with the report.*"
            )
            st.divider()

            # === Invariant Test #6 ===
            st.markdown(
                "##### Invariant Test #6 — Asset Class Map XLV Regression Guard"
            )
            st.caption("Asset Class Map Assigns Sector Equity Tickers Correctly function")
            render_figure(
                "tests/invariants/6_test_asset_class_map_assigns_sector_equity_tickers_correctly"
                "/4_asset_class_xlv_regression_guard.svg"
            )
            st.markdown(
                "Invariant Test #6 verifies that all four "
                "sector-equity tickers (XLF, XLK, XLE, and XLV) map "
                "to `SECTOR_EQUITY` in NB04's `ASSET_CLASS_MAP` "
                "dictionary. XLV was discovered UNMAPPED early in "
                "the project, which silently broke sector-level "
                "aggregation downstream until the bug was caught "
                "manually. *Invariant Test #6 is the regression guard "
                "against the XLV bug ever returning — if anyone "
                "removes XLV from the map again, the next push fails.*"
            )
            st.divider()

            # === Invariant Test #7 ===
            st.markdown(
                "##### Invariant Test #7 — Fourteen-ETF Universe Locked in NB01"
            )
            st.caption("Ticker Universe Contains Fourteen Required ETFs function")
            render_figure(
                "tests/invariants/7_test_ticker_universe_contains_fourteen_required_etfs"
                "/2_ticker_universe_organized_by_asset_class.svg"
            )
            st.markdown(
                "Invariant Test #7 verifies that NB01's `TICKERS` "
                "list contains exactly the 14 ETFs spanning U.S. "
                "large-cap equity (SPY, QQQ), small-cap equity "
                "(IWM), international developed equity (EFA), "
                "emerging-market equity (EEM), sector equity (XLK, "
                "XLF, XLE, XLV), Treasuries (TLT), investment-grade "
                "credit (LQD), high-yield credit (HYG), gold (GLD), "
                "and broad commodities (DBC). *A silent change to "
                "the TICKERS list would alter every empirical claim "
                "downstream — the 2,798-trading-day panel, the "
                "458,662 sentiment articles, the 152-feature "
                "engineered matrix, and every backtest result.*"
            )

    # -----------------------------------------------------------------
    # 7C Docker
    # Source: AI-drafted connective prose; reviewed and approved by Steve.
    # -----------------------------------------------------------------
    with tab_7c:
        st.subheader("Containerization")
        st.markdown(
            "**The dashboard ships as a Docker container** — a portable, "
            "reproducible execution environment that runs identically on "
            "any Linux host with a Docker engine."
        )
        st.markdown(
            "**Dockerfile design decisions:**\n"
            "1. **Single-stage build.**\n"
            "2. **`python:3.11-slim` base image** — Debian-based, "
            "glibc-compatible.\n"
            "3. **Non-root runtime user** (`appuser`, UID 1001).\n"
            "4. **HEALTHCHECK** against Streamlit's `/_stcore/health` "
            "endpoint.\n"
            "5. **Layer ordering optimized for cache reuse** — manifest "
            "copied before source so dependency-only edits skip the "
            "pip-install layer.\n"
            "6. **`pip install .`** — installs the package as a frozen "
            "production artifact."
        )
        st.markdown(
            "The image is approximately 1.9 GB, typical for scientific "
            "Python. The same image runs in the production Azure "
            "Container App described in §7D."
        )

    # -----------------------------------------------------------------
    # 7D Azure Plan
    # -----------------------------------------------------------------
    with tab_7d:
        st.subheader("Azure Container Apps — Live Production Deployment")
        st.markdown(
            "The dashboard runs in production on Azure Container Apps. "
            "Five Azure resources in a single resource group provide the "
            "cloud infrastructure; a federated OpenID Connect Service "
            "Principal authenticates the GitHub Actions runner for "
            "automated deployments; atomic rollback to any prior "
            "revision is verified end-to-end."
        )
        st.success(
            "**Public production URL:**  \n"
            "https://riskml-dashboard.agreeablewave-c95c8817.eastus.azurecontainerapps.io"
        )
        st.markdown("### Azure Resource Inventory")
        st.markdown(
            "Five resources in resource group `rg-riskml-capstone`, "
            "region `eastus`:\n"
            "- **Azure Container Registry** (`riskmlarchuleta`) — private "
            "registry. Holds the dashboard image; admin disabled; AcrPush "
            "rights scoped to the Service Principal.\n"
            "- **Azure Container Apps environment** (`riskml-env`) — "
            "regional shell providing shared VNet, ingress controller, "
            "and Log Analytics binding.\n"
            "- **Azure Container App** (`riskml-dashboard`) — the running "
            "dashboard. CPU 0.5 vCPU, memory 1.0 GiB, scale-to-zero "
            "(min 0, max 1 replica), public HTTPS ingress on port 8501, "
            "system-assigned managed identity for ACR authentication.\n"
            "- **Log Analytics workspace** (`workspace-rgriskmlcapstoneZqNO`) "
            "— lifecycle bound to the Container Apps environment.\n"
            "- **Action group** (`riskml-budget-alerts`) — routes "
            "budget-alert emails at 50%, 80%, and 100% thresholds "
            "against the $20/month budget."
        )
        st.markdown("### Continuous Deployment Pipeline")
        st.markdown(
            "Every push to the `main` branch triggers an automated "
            "end-to-end deploy via GitHub Actions:\n"
            "1. **Federated OpenID Connect authentication** — the "
            "GitHub Actions runner mints an OIDC token; Azure Active "
            "Directory verifies the token's subject claim against the "
            "federated credential on Service Principal "
            "`sp-riskml-capstone-cd`. No client secret, no long-lived "
            "credential.\n"
            "2. **`docker build` + `docker push`** to the Container "
            "Registry, tagged with the git short-SHA plus `latest`.\n"
            "3. **`az containerapp update`** flips the Container App "
            "to the new image, atomically creating a new immutable "
            "revision.\n"
            "4. **Verification** asserts "
            "`provisioningState == Succeeded`; otherwise the workflow "
            "fails."
        )
        st.success(
            "The Service Principal is scoped to exactly two resources — "
            "`AcrPush` on the Container Registry, `Contributor` on the "
            "Container App. If the GitHub credentials were ever to leak, "
            "the blast radius is precisely two resources, with no "
            "ability to escalate, lateral-move, or read other data in "
            "the subscription."
        )
        st.markdown("### Atomic Rollback")
        st.markdown(
            "The CD pipeline's atomic-rollback capability is verified "
            "end-to-end:\n"
            "- All revisions are preserved; any can be reactivated in "
            "**6.8 seconds** via `az containerapp revision activate` — "
            "no image rebuild, no source-code touch, pure control-plane "
            "atomic flip.\n"
            "- A subsequent `git revert HEAD` + `git push` closes the "
            "loop, landing the fix through the same CD pipeline and "
            "creating a new revision."
        )
        st.markdown("### Cost Posture")
        st.markdown(
            "**Steady-state burn rate: approximately $5.10/month** — "
            "driven entirely by the Container Registry Basic SKU flat "
            "fee. The Container App bills only when actively serving "
            "requests; with scale-to-zero configured, idle hours cost "
            "zero. A typical CD run adds approximately five cents in "
            "incremental compute. A $20/month budget alert monitors "
            "this and emails at the 50%, 80%, and 100% thresholds; "
            "no alert has ever fired."
        )
        st.info(
            "ACA abstracts Kubernetes for single-container workloads; "
            "AKS would be the right choice for multi-service production "
            "systems. ACA delivers the same containerized portability "
            "without the operational overhead."
        )

# =====================================================================
# §8 Artifact Manifest
# =====================================================================
elif page == "§8 Artifact Manifest":
    st.title("§8 Artifact Manifest")
    st.markdown(
        "Every artifact the dashboard reads, with a present/absent flag."
    )

    expected_artifacts = [
        # NB04 tables and figures.
        ("NB04 — tables", "notebook04_causal_vs_baseline_rmse_mae.csv"),
        ("NB04 — tables", "notebook04_entropy_comparison.csv"),
        ("NB04 — figures", "notebook04_causal_vs_baseline_rmse_delta.png"),
        ("NB04 — figures", "notebook04_entropy_comparison.png"),
        ("NB04 — figures", "notebook04_causal_feature_importance_SPY.png"),
        # NB05 tables and figures.
        ("NB05 — tables", "notebook05_portfolio_performance.csv"),
        ("NB05 — figures", "notebook05_equity_curves.png"),
        ("NB05 — figures", "notebook05_drawdown_comparison.png"),
        # NB06 tables and figures.
        ("NB06 — tables", "notebook06_regime_portfolio_metrics.csv"),
        ("NB06 — tables", "notebook06_stress_drawdown.csv"),
        ("NB06 — figures", "notebook06_regime_portfolio_comparison.png"),
        ("NB06 — figures", "notebook06_stress_drawdown.png"),
        ("NB06 — figures", "notebook06_factor_exposure_entropy.png"),
        # NB07 custom DAG figure.
        ("NB07 — figures", "notebook07_dag_clean.png"),
        # Dashboard-specific harvested figures.
        ("Dashboard — figures", "pipeline_two_tracks.png"),
        ("Dashboard — figures", "dag_gate_inventory.png"),
        ("Dashboard — figures", "baseline_causal_twins.png"),
        ("Dashboard — figures", "etf_universe_returns.png"),
        ("Dashboard — figures", "vixcls_timeseries.png"),
        ("Dashboard — figures", "feature_missingness_top30.png"),
        # NB02 and NB03 figures referenced by the dashboard.
        ("NB02 — figures", "notebook02_feature_counts_by_dag.png"),
        ("NB02 — figures", "notebook02_vol_feature_vs_target_SPY.png"),
        ("NB03 — figures", "notebook03_acf_diagnostics.png"),
        ("NB03 — figures", "notebook03_distribution_diagnostics.png"),
        # Pipeline test diagrams.
        ("Tests — pipeline", "tests/pipeline/1_test_storage_local_fallback/3_round_trip_flow.svg"),
        ("Tests — pipeline", "tests/pipeline/2_test_extract_price_panel_multiindex/2_raw_dataframe_with_multiindex.svg"),
        ("Tests — pipeline", "tests/pipeline/3_test_compute_momentum_features/3_momentum_output_structure.svg"),
        ("Tests — pipeline", "tests/pipeline/4_test_compute_momentum_features_rejects_empty/3_defensive_vs_silent.svg"),
        ("Tests — pipeline", "tests/pipeline/5_test_compute_realized_volatility/3_annualization_sanity_check.svg"),
        # Invariant test diagrams.
        ("Tests — invariants", "tests/invariants/1_test_random_seed_is_locked_at_692_across_all_notebooks/1_read_notebook_source_helper.svg"),
        ("Tests — invariants", "tests/invariants/1_test_random_seed_is_locked_at_692_across_all_notebooks/2_test_random_seed_regex_anatomy_and_loop.svg"),
        ("Tests — invariants", "tests/invariants/2_test_causal_edges_dict_encodes_five_core_directed_edges/2_test_causal_edges_dag_architecture.svg"),
        ("Tests — invariants", "tests/invariants/3_test_dag_figure_has_nine_nodes_and_seven_directed_edges/2_test_dag_figure_complete_architecture.svg"),
        ("Tests — invariants", "tests/invariants/4_test_risk_stage_gating_lists_match_report_specification/3_risk_stage_belt_and_suspenders.svg"),
        ("Tests — invariants", "tests/invariants/5_test_risk_stage_feature_count_equals_74/1_gating_summary_csv_structure.svg"),
        ("Tests — invariants", "tests/invariants/6_test_asset_class_map_assigns_sector_equity_tickers_correctly/4_asset_class_xlv_regression_guard.svg"),
        ("Tests — invariants", "tests/invariants/7_test_ticker_universe_contains_fourteen_required_etfs/2_ticker_universe_organized_by_asset_class.svg"),
    ]

    rows = []
    for source, filename in expected_artifacts:
        if filename.endswith(".csv"):
            present = (TABLES_DIR / filename).exists()
        else:
            present = (FIGURES_DIR / filename).exists()
        rows.append(
            {
                "source": source,
                "filename": filename,
                "present": "✅" if present else "❌",
            }
        )

    manifest = pd.DataFrame(rows)
    st.dataframe(manifest, width="stretch", hide_index=True)

    n_present = (manifest["present"] == "✅").sum()
    n_total = len(manifest)
    st.metric("Artifacts present", f"{n_present} / {n_total}")
