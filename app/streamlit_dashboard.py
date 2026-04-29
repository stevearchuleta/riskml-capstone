"""
RiskML Capstone — Streamlit Dashboard.

Read-only presentation layer for the riskml-capstone project. Reads
pre-computed artifacts from reports/tables/ and reports/figures/ produced
by notebooks NB04, NB05, and NB06. No model training, no data API calls,
no overwrites of upstream artifacts.

Run locally:
    streamlit run app/streamlit_dashboard.py

Author : Steven Archuleta
Course : MScFE 690 Capstone Project (WorldQuant University)
"""

from pathlib import Path

import pandas as pd
import streamlit as st

# -----------------------------------------------------
# Page configuration — must be the first Streamlit call.
# ------------------------------------------------------
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
    """Walk upward from `start` until a .git or pyproject.toml is found."""
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
    st.dataframe(df, use_container_width=True)


# -------------------
# Sidebar navigation.
# -------------------
st.sidebar.title("📊 RiskML Capstone")
st.sidebar.caption("Causal-Aware ML-Driven Risk Forecasting")
st.sidebar.divider()

PAGES = [
    "Pipeline Overview",
    "Model Comparison",
    "Causal vs Baseline",
    "Portfolio Performance",
    "Validation, Ablation, Stress",
    "Artifact Manifest",
]
page = st.sidebar.radio("Navigate", PAGES)

st.sidebar.divider()
st.sidebar.caption(f"Repo root: `{REPO_ROOT.name}`")
st.sidebar.caption(
    "Read-only dashboard. All numbers derive from saved artifacts produced "
    "by notebooks NB04 through NB06."
)


# -----------------------
# Page: Pipeline Overview
# -----------------------
if page == "Pipeline Overview":
    st.title("Pipeline Overview")
    st.markdown(
        """
        **Causal-Aware, Machine-Learning-Driven, Factor-Informed Risk Forecasting** —
        a Python pipeline integrating NLP sentiment, directed factor constraints,
        and portfolio analytics, evaluated across 14 ETFs over a 2016–2025 window.

        The manual DAG below restricts which feature families flow into the
        risk forecasting stage (NB04). The dashboard reads only the saved
        outputs of that pipeline; no model fitting or data fetching occurs here.
        """
    )
    render_figure(
        "notebook07_dag_clean.png",
        caption="Manual DAG enforced at NB04 — three causal paths gating risk-stage feature inputs.",
    )

# ----------------------
# Page: Model Comparison
# ----------------------
elif page == "Model Comparison":
    st.title("Model Comparison")
    st.markdown(
        "Forecast accuracy of the causal pipeline relative to unconstrained baselines, "
        "measured over 500 test-block observations per ticker."
    )
    render_table(
        "notebook04_causal_vs_baseline_rmse_mae.csv",
        caption="RMSE and MAE — causal vs baseline models.",
    )
    render_figure(
        "notebook04_causal_vs_baseline_rmse_delta.png",
        caption="Per-ticker RMSE delta (causal minus baseline).",
    )
    st.markdown(
        "**Reading this chart:** positive bars indicate tickers where the causal "
        "model has higher RMSE than the baseline; the spread widens monotonically "
        "from GLD (smallest delta) to XLE (largest), showing the DAG-constrained "
        "model trades unconstrained accuracy for interpretability — and the cost "
        "of that trade-off varies systematically by asset."
    )

# -------------------------------------------------------
# Page: Causal vs Baseline (entropy + feature importance)
# -------------------------------------------------------
elif page == "Causal vs Baseline":
    st.title("Causal vs Baseline — Interpretability")
    st.markdown(
        "**Tier 1 entropy** (NB04): normalized Shannon entropy over XGBoost gain shares. "
        "Lower entropy means the model concentrates predictive weight on fewer features."
    )
    render_table(
        "notebook04_entropy_comparison.csv",
        caption="Normalized feature-importance entropy per ticker.",
    )
    render_figure(
        "notebook04_entropy_comparison.png",
        caption="Causal vs baseline — feature-importance entropy.",
    )
    st.divider()
    render_figure(
        "notebook04_causal_feature_importance_SPY.png",
        caption="SPY feature importance under the causal model.",
    )
    st.markdown(
        "**Reading this chart:** the causal model's predictive weight for SPY "
        "concentrates heavily in a small number of allowed features — "
        "primarily volatility and macro inputs — confirming that the DAG "
        "gate at NB04 is doing what it was designed to do: forcing the risk "
        "forecast to depend on risk-relevant inputs only, not on momentum or "
        "value features that the unconstrained baseline would otherwise exploit."
    )

# ---------------------------
# Page: Portfolio Performance
# ---------------------------
elif page == "Portfolio Performance":
    st.title("Portfolio Performance")
    st.markdown(
        "Walk-forward backtest over 499 trading days (2023-12-29 through 2025-12-31). "
        "Equal-risk-budget inverse-volatility weighting, 10% annualized vol target, "
        "Ledoit–Wolf covariance shrinkage, monthly rebalancing, 10 bps transaction costs."
    )
    render_table(
        "notebook05_portfolio_performance.csv",
        caption="Risk and return metrics per model.",
    )

    col1, col2 = st.columns(2)
    with col1:
        render_figure(
            "notebook05_equity_curves.png",
            caption="Equity curves.",
        )
        st.markdown(
            "**Reading the equity curves (top panel — baselines):** "
            "BASELINE_LSTM produces the strongest cumulative growth, followed "
            "by EWMA and XGBOOST clustered tightly together; BASELINE_MLP "
            "trails the group, suggesting the simpler statistical and tree "
            "models keep pace with deep architectures over this window."
        )
        st.markdown(
            "**Reading the equity curves (bottom panel — causal models):** "
            "CAUSAL_LSTM and CAUSAL_XGBOOST track each other closely and "
            "finish near 1.20×, while CAUSAL_MLP lags — the same MLP "
            "underperformance pattern visible in the baseline panel, "
            "confirming the architecture choice matters more than DAG gating "
            "for portfolio outcomes."
        )
    with col2:
        render_figure(
            "notebook05_drawdown_comparison.png",
            caption="Drawdown comparison.",
        )
        st.markdown(
            "**Reading the drawdown panels (baselines, top):** all four "
            "baselines hit their deepest drawdown of approximately −8% during "
            "the April 2025 stress episode visible as the sharp synchronized "
            "trough; recovery is rapid and complete by late 2025."
        )
        st.markdown(
            "**Reading the drawdown panels (causal models, bottom):** "
            "causal models show a similar April 2025 trough but with greater "
            "dispersion across architectures — CAUSAL_MLP reaches roughly −9% "
            "while CAUSAL_XGBOOST stays shallower at approximately −6%, "
            "indicating the DAG constraint does not uniformly reduce "
            "drawdown depth and architecture again dominates."
        )

# ----------------------------------
# Page: Validation, Ablation, Stress
# ----------------------------------
elif page == "Validation, Ablation, Stress":
    st.title("Validation, Ablation, Stress")
    tab_ablation, tab_regime, tab_stress = st.tabs(
        ["Ablation", "Regime Conditional", "Stress Episode"]
    )

    with tab_ablation:
        render_table(
            "notebook06_ablation_table.csv",
            caption="Ablation — incremental feature-family contributions.",
        )
        render_figure("notebook06_ablation_comparison.png")
        st.markdown(
            "**Reading this table:** Stage 1 (unconstrained baseline XGBoost) "
            "achieves RMSE 0.0698 with feature-importance entropy 0.832; "
            "Stage 3 (DAG-constrained causal model) trades a slightly higher "
            "RMSE of 0.0811 for marginally improved interpretability "
            "(feature entropy 0.848 and factor entropy 0.793) and a "
            "substantially better Calmar ratio (1.498 vs 1.343), showing "
            "the DAG constraint pays off in risk-adjusted return more than "
            "in raw forecast accuracy."
        )

    with tab_regime:
        render_table(
            "notebook06_regime_portfolio_metrics.csv",
            caption="Portfolio metrics conditional on VIX regime.",
        )
        render_figure("notebook06_regime_portfolio_comparison.png")
        st.markdown(
            "**Reading the Sharpe ratio chart (left):** all three models — "
            "EWMA, XGBOOST, CAUSAL_XGBOOST — produce strongly positive "
            "Sharpe ratios in low-VIX regimes (≈3.2) and strongly negative "
            "Sharpe ratios in high-VIX regimes (≈−1.5), confirming the "
            "portfolios are calibrated to calm markets and lose money during "
            "stress."
        )
        st.markdown(
            "**Reading the absolute drawdown chart (right):** high-VIX "
            "drawdowns dwarf low-VIX drawdowns by a factor of roughly 5× "
            "across all models; CAUSAL_XGBOOST shows the smallest high-VIX "
            "drawdown of the three, suggesting the DAG constraint provides "
            "modest protection during volatile regimes."
        )

    with tab_stress:
        render_table(
            "notebook06_stress_drawdown.csv",
            caption="April 2025 stress episode — drawdown summary.",
        )
        render_figure("notebook06_stress_drawdown.png")
        st.markdown(
            "**Reading this chart:** all three models lose 4–6% from the "
            "March 31, 2025 reference point during early April, then recover "
            "monotonically through Q3 2025 and finish the window between "
            "+0.8% (EWMA) and +1.2% (XGBOOST and CAUSAL_XGBOOST) — "
            "demonstrating the portfolios fully recovered from the stress "
            "episode within ~6 months, with the two learned models outpacing "
            "the EWMA baseline during the recovery."
        )

# -----------------------
# Page: Artifact Manifest
# -----------------------
elif page == "Artifact Manifest":
    st.title("Artifact Manifest")
    st.markdown(
        "Every CSV and PNG the dashboard reads, with a present/absent flag. "
        "Useful for verifying a fresh clone has all required artifacts."
    )

    expected_artifacts = [
        ("NB04 — tables", "notebook04_causal_vs_baseline_rmse_mae.csv"),
        ("NB04 — tables", "notebook04_entropy_comparison.csv"),
        ("NB04 — figures", "notebook04_dag_figure.png"),
        ("NB04 — figures", "notebook04_causal_vs_baseline_rmse_delta.png"),
        ("NB04 — figures", "notebook04_entropy_comparison.png"),
        ("NB04 — figures", "notebook04_causal_feature_importance_SPY.png"),
        ("NB05 — tables", "notebook05_portfolio_performance.csv"),
        ("NB05 — figures", "notebook05_equity_curves.png"),
        ("NB05 — figures", "notebook05_drawdown_comparison.png"),
        ("NB06 — tables", "notebook06_ablation_table.csv"),
        ("NB06 — tables", "notebook06_regime_portfolio_metrics.csv"),
        ("NB06 — tables", "notebook06_stress_drawdown.csv"),
        ("NB06 — figures", "notebook06_ablation_comparison.png"),
        ("NB06 — figures", "notebook06_regime_portfolio_comparison.png"),
        ("NB06 — figures", "notebook06_stress_drawdown.png"),
        ("NB07 — figures", "notebook07_dag_clean.png"),
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
    st.dataframe(manifest, use_container_width=True, hide_index=True)

    n_present = (manifest["present"] == "✅").sum()
    n_total = len(manifest)
    st.metric("Artifacts present", f"{n_present} / {n_total}")