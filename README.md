# riskml-capstone

**Causal-Aware, Machine-Learning-Driven Risk Forecasting and Factor Construction**

*A Python–Azure Pipeline Integrating NLP, Directed Factor Constraints, and Portfolio Analytics*

[![CI](https://github.com/stevearchuleta/riskml-capstone/actions/workflows/ci.yml/badge.svg)](https://github.com/stevearchuleta/riskml-capstone/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

This repository contains the code, notebooks, and pipeline for my **MScFE 692 Capstone Project** at WorldQuant University. The project investigates whether imposing manual causal constraints on an ML-driven risk forecasting pipeline improves forecast accuracy, portfolio performance, and interpretability compared to unconstrained baselines.

### Research Question

> How does imposing manual causal constraints on an ML-driven risk forecasting pipeline affect forecast accuracy, portfolio performance, and interpretability compared to unconstrained baselines?

### Core Hypotheses

| ID | Hypothesis |
|----|------------|
| H1 | A DAG-constrained pipeline reduces risk forecast error (MAE, RMSE) relative to an unconstrained ML baseline under out-of-sample testing. |
| H2 | A DAG-constrained allocation policy reduces drawdown and improves risk-targeting stability relative to baseline allocation. |
| H3 | Adding NLP sentiment improves near-term risk control and/or allocation stability relative to a no-text baseline (ablation). |
| H4 | Factor-exposure entropy increases (more balanced exposure) under DAG constraints and risk budgeting, relative to baseline. |

---

## Manual DAG Structure

The causal ordering used in this project:

```
Sentiment → Momentum → Returns
Value → Returns
Volatility → Risk → Allocation
```

This small, theory-driven directed acyclic graph restricts information flow to improve stability and interpretability of ML-based risk forecasting.

---

## Project Structure

```
riskml-capstone/
├── README.md
├── pyproject.toml
├── Dockerfile
├── .github/workflows/ci.yml
├── data/                          # Parquet files (gitignored)
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_data_extraction.ipynb
│   ├── 02_feature_engineering_ml_factors.ipynb
│   ├── 03_risk_forecasting_baselines.ipynb
│   ├── 04_risk_forecasting_causal.ipynb
│   ├── 05_portfolio_construction.ipynb
│   ├── 06_validation_ablation_stress.ipynb
│   └── 07_app_demo_and_plots.ipynb
├── riskml/                        # Installable Python package
│   ├── etl/
│   │   ├── prices.py
│   │   └── text_sources.py
│   ├── features/
│   │   ├── tech.py
│   │   ├── fundamentals.py
│   │   ├── nlp.py
│   │   └── regime.py
│   ├── models/
│   │   ├── risk_ewma.py
│   │   ├── risk_har.py
│   │   ├── risk_xgb.py
│   │   └── risk_causal.py
│   ├── causal/
│   │   ├── dag_constraints.py
│   │   └── metrics.py
│   ├── portfolio/
│   │   ├── allocator.py
│   │   └── costs.py
│   └── backtest/
│       └── engine.py
├── tests/
│   ├── test_etl.py
│   ├── test_features.py
│   ├── test_models.py
│   ├── test_causal.py
│   └── test_portfolio.py
├── app/
│   └── streamlit_dashboard.py
├── azure/
│   ├── aml_pipeline.yaml
│   └── env.yaml
├── reports/
│   └── figures/
└── docs/
    ├── project_charter.md
    ├── data_dictionary.md
    ├── feature_contract.md
    └── method_notes.md
```

---

## Installation

### Prerequisites

- Python 3.11.x (tested with 3.11.14)
- Conda (recommended) or pip

### Setup

```bash
# Clone the repository
git clone https://github.com/stevearchuleta/riskml-capstone.git
cd riskml-capstone

# Create conda environment (recommended)
conda create -n capstone python=3.11 -y
conda activate capstone

# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

### Environment Variables

Create a `.env` file in the project root:

```env
FRED_API_KEY=your_fred_api_key_here
# Optional: Azure ML credentials if using cloud pipeline
```

---

## Data Sources

| Dataset | Series | Purpose | Source |
|---------|--------|---------|--------|
| ETF Prices | SPY, QQQ, IWM, EFA, EEM, XLK, XLF, XLE, XLV, TLT, LQD, HYG, GLD, DBC | Returns, Volatility, Backtests | yfinance |
| Fama-French | Mkt-RF, SMB, HML, RF | Factor exposures, Beta decomposition | Fama-French Data Library |
| FRED Macro | DTB3, VIXCLS, T10Y2Y | Risk-free rate, Regime features | FRED API |
| News Headlines | Market/sector-tagged | Sentiment features | RSS feeds |

---

## Pipeline Stages

| Stage | Description | Notebook |
|-------|-------------|----------|
| 1. Ingest | ETF prices, factors, macro, headlines | `01_data_extraction.ipynb` |
| 2. Diagnostics | Stylized facts, stationarity, tails, clustering | `01_data_extraction.ipynb` |
| 3. Features | Momentum, volatility, value proxy, sentiment | `02_feature_engineering_ml_factors.ipynb` |
| 4. Baselines | EWMA, HAR, XGBoost risk models | `03_risk_forecasting_baselines.ipynb` |
| 5. Causal | Manual DAG constraints + causal pipeline | `04_risk_forecasting_causal.ipynb` |
| 6. Portfolio | Risk-target allocation, costs, turnover | `05_portfolio_construction.ipynb` |
| 7. Backtest | Walk-forward with transaction costs | `05_portfolio_construction.ipynb` |
| 8. Validation | Ablations, regime tests, stress tests | `06_validation_ablation_stress.ipynb` |
| 9. Demo | Streamlit dashboard, exports | `07_app_demo_and_plots.ipynb` |

---

## Usage

### Run ETL Pipeline

```bash
python -m riskml.etl.prices
```

### Run Tests

```bash
pytest -n auto --cov=riskml
```

### Launch Dashboard

```bash
streamlit run app/streamlit_dashboard.py
```

### Docker

```bash
docker build -t riskml-capstone .
docker run -it riskml-capstone pytest
```

---

## Validation Design

- **Time-ordered splits**: No lookahead bias; strict train/validation/test separation
- **Single-use test set**: Test set evaluated only once at final reporting
- **Ablation ladder**: Baseline → +NLP → +ML factors → +DAG constraints
- **Regime robustness**: Optional HMM or VIX/yield-curve regime labels

---

## Key Metrics

| Category | Metrics |
|----------|---------|
| Forecast Accuracy | MAE, RMSE, directional accuracy |
| Portfolio Performance | Sharpe ratio, max drawdown, realized vs target volatility |
| Interpretability | Factor exposure entropy, coefficient stability |
| Governance | Leakage checks, data freeze hashes |

---

## References

- Michel, A., Arun, A., Sarmah, B., & Pasquali, S. (2025). *FinCARE: Financial causal analysis with reasoning and evidence*. arXiv.
- Arun, A., et al. (2025). *FinReflectKG: Agentic construction and evaluation of financial knowledge graphs*. arXiv.
- Timm, B. (2021). *Utilizing machine learning to address noise in covariance and correlation matrices* (Master's thesis). Copenhagen Business School.
- López de Prado, M. (2022). *Causal factor investing: Can factor investing become scientific?* SSRN.
- Pedersen, L. H., Fitzgibbons, S., & Pomorski, L. (2021). Enhanced portfolio optimization. *Journal of Finance*, 76(4), 1975–2034.

---

## Author

**Steven "Steve" Archuleta**  
MScFE Candidate, WorldQuant University  
GitHub: [@stevearchuleta](https://github.com/stevearchuleta)

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

- WorldQuant University MScFE Program
- MScFE 660 Risk Management coursework (GWP1, GWP2, GWP3)
- Open-source contributors to pandas, scikit-learn, statsmodels, and yfinance
