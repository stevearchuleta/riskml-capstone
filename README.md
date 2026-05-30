# Causal-Aware, Machine-Learning-Driven, Factor-Informed Risk Forecasting

*A reproducible Python and Azure pipeline integrating NLP sentiment, directed factor constraints, volatility forecasting, and portfolio analytics.*

[![CI](https://github.com/stevearchuleta/riskml-capstone/actions/workflows/ci.yml/badge.svg)](https://github.com/stevearchuleta/riskml-capstone/actions/workflows/ci.yml)
[![CD](https://github.com/stevearchuleta/riskml-capstone/actions/workflows/cd.yml/badge.svg)](https://github.com/stevearchuleta/riskml-capstone/actions/workflows/cd.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](pyproject.toml)
[![Live Dashboard](https://img.shields.io/badge/dashboard-live-success.svg)](https://riskml-dashboard.agreeablewave-c95c8817.eastus.azurecontainerapps.io)

**Authors:** Steven Archuleta (USA) and Andrew Nilsen (Indonesia)
**Institution:** WorldQuant University — MScFE 690
**Submission date:** 01 June 2026

---

## Research Question

> How does imposing manual causal constraints on an ML-driven risk-forecasting pipeline affect forecast accuracy, portfolio performance, and interpretability compared to unconstrained baseline models?

This study evaluates a small, manually specified Directed Acyclic Graph (DAG) as a lightweight information-flow governance layer over a machine-learning risk-forecasting pipeline. The DAG is treated as an auditable modeling constraint, not as a claim about the true data-generating process of financial markets.

---

## Live Dashboard

An eight-page interactive Streamlit dashboard presents the research narrative, the data pipeline, the DAG constraint experiment, the forecast results, the portfolio backtest, the stress analysis, and the deployment architecture. The dashboard tells the full story without requiring a reader to open the report.

**→ https://riskml-dashboard.agreeablewave-c95c8817.eastus.azurecontainerapps.io**

> The application runs on Azure Container Apps with scale-to-zero enabled. A first request after an idle period may incur a 5–15 second cold start.

The dashboard is a read-only presentation layer. The application reads precomputed artifacts from `reports/tables/` and `reports/figures/`; the application does not train models, call external data APIs, or overwrite upstream artifacts.

<!-- Screenshot placeholder: docs/screenshots/dashboard_overview.png -->

---

## Executive Summary

This study tests whether a causal-aware feature gate improves volatility forecasting, portfolio risk control, and interpretability relative to unconstrained machine-learning baselines.

The central finding: the DAG constraint did **not** improve raw forecast accuracy, but the same constraint did improve the primary tree-family portfolio risk-control profile. In the XGBoost comparison, the DAG-constrained model incurred a higher RMSE than the unconstrained XGBoost model, yet produced a better Sharpe ratio, a shallower maximum drawdown, a higher Calmar ratio, and stronger interpretability diagnostics. Causal constraints traded a small, statistically insignificant accuracy cost for measurably steadier portfolio behavior.

| Dimension | Verdict |
|---|---|
| Forecast accuracy | **H1 not confirmed.** The DAG-constrained XGBoost model showed a higher aggregate RMSE than the unconstrained baseline; Diebold–Mariano HAC tests did not reject equal predictive accuracy. |
| Portfolio risk control | **H2 partially confirmed.** Within the primary tree comparison, the constrained model produced the best Sharpe ratio, the shallowest drawdown, and the highest Calmar ratio. |
| NLP sentiment contribution | **H3 directionally supported.** A matched-window post-2020 comparison showed a small RMSE improvement from sentiment, with no statistical-significance claim. |
| Interpretability | **Secondary diagnostic supported.** Normalized feature-importance entropy and factor-exposure entropy both increased under the constraint. |

---

## Headline Results

### Aggregate Forecast Accuracy

Seven model variants were evaluated under an expanding-window walk-forward protocol on a 20-trading-day forward realized-volatility target, with strict time-ordered splits and leakage governance.

| Model | Aggregate RMSE | Aggregate MAE |
|---|---:|---:|
| EWMA | 0.071 | 0.048 |
| XGBoost | 0.070 | 0.049 |
| Baseline MLP | 0.167 | 0.131 |
| Baseline LSTM | 0.095 | 0.070 |
| Causal XGBoost | 0.081 | 0.050 |
| Causal MLP | 0.175 | 0.122 |
| Causal LSTM | 0.088 | 0.062 |

### Primary Tree-Family Portfolio Comparison

| Model | Sharpe Ratio | Max Drawdown † | Calmar Ratio | Avg Realized Vol 21d |
|---|---:|---:|---:|---:|
| EWMA | 0.583 | −7.38% | 1.174 | 6.44% |
| XGBoost | 0.567 | −6.03% | 1.343 | 5.83% |
| Causal XGBoost | **0.667** | **−5.83%** | **1.498** | **5.82%** |

> **† Full-period maximum drawdown** over the 499-day active backtest window. These figures are *not* the April 2025 stress-window losses; the stress-window peak-to-trough losses from the 31 March 2025 reference were −5.86% (EWMA), −4.56% (XGBoost), and −4.43% (Causal XGBoost).

### Full Seven-Model Portfolio Summary

| Model | Annualized Return | Annualized Volatility | Sharpe Ratio | Max Drawdown | Calmar Ratio | Final Equity |
|---|---:|---:|---:|---:|---:|---:|
| EWMA | 8.67% | 6.93% | 0.583 | −7.38% | 1.174 | 1.179 |
| XGBoost | 8.10% | 6.11% | 0.567 | −6.03% | 1.343 | 1.167 |
| MLP | 6.07% | 7.18% | 0.227 | −8.18% | 0.742 | 1.124 |
| LSTM | **11.40%** | 8.55% | **0.777** | −9.10% | 1.253 | **1.238** |
| Causal XGBoost | 8.73% | **6.05%** | 0.667 | **−5.83%** | **1.498** | 1.180 |
| Causal MLP | 6.36% | 6.87% | 0.275 | −8.58% | 0.741 | 1.130 |
| Causal LSTM | 9.50% | 8.02% | 0.611 | −8.69% | 1.094 | 1.197 |

---

## Methodology at a Glance

The pipeline is organized around two parallel tracks. The forecasting track predicts 20-trading-day forward realized volatility for each ETF. The portfolio track converts the volatility forecasts into inverse-volatility weights under a common backtest protocol. Each model outputs a predicted volatility value per asset; no model outputs portfolio weights directly.

---

## Manual DAG Structure

The causal constraint layer is a nine-node, seven-edge directed graph:

```text
Sentiment → Momentum → Returns
Value → Returns
Volatility → Risk → Allocation
MACRO → Risk
REGIME → Risk
```

The risk-stage feature gate admits only the direct parents of the `Risk` node:

| Feature Family | Count | Risk-Stage Status | Reason |
|---|---:|---|---|
| `VOL__` | 70 | Allowed | Direct parent of the Risk node |
| `MACRO__` | 3 | Allowed | Exogenous risk-environment conditioner |
| `REGIME__` | 1 | Allowed | High/low volatility-state conditioner |
| `MOM__` | 56 | Blocked | Upstream of Returns, not a direct Risk parent |
| `VAL__` | 18 | Blocked | Value explains Returns, not Risk |
| `ML__` | 3 | Blocked | No direct DAG parent role at the Risk node |
| `SENT__` | 1 | Blocked / H3 only | Coverage-limited and upstream of Momentum |

The unconstrained non-sentiment matrix contains 151 features. The DAG-constrained risk-stage matrix contains 74 features: `VOL__`, `MACRO__`, and `REGIME__` only — a 51% reduction that acts as an implicit regularizer.

---

## Data Sources

| Dataset | Series / Assets | Purpose | Source |
|---|---|---|---|
| ETF prices | SPY, QQQ, IWM, EFA, EEM, XLK, XLF, XLE, XLV, TLT, LQD, HYG, GLD, DBC | Returns, volatility features, portfolio backtests | Yahoo Finance (`yfinance`) |
| Fama–French factors | Mkt-RF, SMB, HML, RF | Factor exposures, HML value proxy, risk-free rate | Kenneth French Data Library |
| FRED macro | DTB3, VIXCLS, T10Y2Y | Risk-free rate, volatility regime, term-structure features | FRED |
| NLP sentiment | Market-level news sentiment (458,662 articles) | H3 sentiment ablation | Alpha Vantage News Sentiment API |

> Data licenses and commercial-use terms warrant review before any conversion of this research repository into a paid product or live API.

---

## Repository Structure

```text
riskml-capstone/
├── app/             # Streamlit dashboard deployed to Azure Container Apps
├── azure/           # Azure deployment configuration
├── data/            # Input data and cached artifacts; large files are gitignored
├── docs/            # Runbooks, deployment guides, screenshots, and notes
├── notebooks/       # Pipeline notebooks NB01–NB07; execute in order
├── reports/         # Final report, figures, metric tables, and dashboard artifacts
├── riskml/          # Installable Python package
├── tests/           # Smoke, pipeline, and invariant test suites
├── Dockerfile       # Container image definition for the dashboard
├── pyproject.toml   # Package metadata and dependencies
└── README.md
```

---

## Notebook Pipeline

| Notebook | Stage | Main Output |
|---|---|---|
| `01_data_extraction.ipynb` | Data ingestion and diagnostics | ETF prices, macro data, factors, sentiment, EDA artifacts |
| `02_feature_engineering_ml_factors.ipynb` | Feature engineering | Feature matrix, DAG prefix families, frozen forecast target |
| `03_risk_forecasting_baselines.ipynb` | Baseline forecasting | EWMA, XGBoost, MLP, LSTM, and the H3 sentiment comparison |
| `04_risk_forecasting_causal.ipynb` | DAG-constrained forecasting | Causal XGBoost, Causal MLP, Causal LSTM, feature-entropy diagnostics |
| `05_portfolio_construction.ipynb` | Portfolio construction | Inverse-volatility portfolios, transaction costs, equity curves |
| `06_validation_ablation_stress.ipynb` | Validation and stress tests | Ablation table, regime metrics, Diebold–Mariano HAC tests, April 2025 stress window |
| `07_app_demo_and_plots.ipynb` | Dashboard support | Static artifacts and exported plots consumed by Streamlit |

---

## Reproducibility

This study fixes a single global random seed (**692**) across all stochastic operations, so a clean re-run reproduces the reported figures.

```bash
git clone https://github.com/stevearchuleta/riskml-capstone.git
cd riskml-capstone

conda create -n capstone python=3.11 -y
conda activate capstone

pip install -e ".[dev]"
pytest -q
```

Run the seven notebooks NB01 through NB07 in sequence. Each notebook reads a documented upstream artifact and writes a documented downstream artifact, which preserves an auditable artifact lineage from raw data to final figures.

### Environment Variables

Create a `.env` file in the repository root. Populate the file with personal API keys and the relevant Azure identifiers; never commit the file.

```env
FRED_API_KEY=your_fred_api_key_here
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_api_key_here
AZURE_SUBSCRIPTION_ID=your_azure_subscription_id_here
AZURE_RESOURCE_GROUP=your_resource_group_here
AZURE_REGION=your_region_here
```

---

## CI/CD Posture

- **Continuous Integration:** 14 tests run on every push to `main` — 2 smoke, 5 pipeline, and 7 invariant tests — alongside a Ruff lint pass and a package-import check.
- **Continuous Deployment:** GitHub Actions builds the container image, pushes the image to Azure Container Registry, and updates the Azure Container App revision.
- **Authentication:** federated OpenID Connect — no client secret and no long-lived credential of any kind. The deploy identity is scoped to exactly two resources (AcrPush on the registry, Contributor on the Container App).
- **Cost posture:** scale-to-zero holds the steady-state run rate at the Azure Container Registry Basic-SKU floor.
- **Resilience:** an atomic rollback to a prior known-good revision was demonstrated in 6.8 seconds with no image rebuild.

---

## Validation and Governance

| Governance Area | Implementation |
|---|---|
| Temporal integrity | Time-ordered splits and expanding-window walk-forward validation |
| Leakage control | 20-trading-day forecast-horizon buffer and a physically separate target artifact |
| Reproducibility | Fixed random seed, artifact contracts, version-controlled notebooks and package code |
| CI discipline | GitHub Actions, deterministic tests, package-import checks, linting |
| Interpretability | DAG feature gate, feature-importance entropy, factor-exposure entropy |
| Portfolio fairness | A shared allocation protocol applied identically across all forecast families |

---

## Important Limitations

- The DAG encodes economically motivated directional assumptions; the graph is not a discovered causal map of financial markets.
- H1 is not confirmed: the DAG-constrained XGBoost model did not lower RMSE relative to the unconstrained XGBoost model.
- H3 is only directionally supported: the sentiment comparison uses a matched post-2020 window and includes no Diebold–Mariano significance test.
- Alpha Vantage sentiment coverage begins in January 2020, so full-history 2016–2025 sentiment propagation is structurally unavailable from the current provider.
- The dashboard is a research presentation layer, not a live trading system and not a registered investment-advice product.
- Historical backtests do not guarantee future performance.

---

## Commercialization Boundary

This repository is best understood as a reproducible, research-grade proof of capability for a future risk-intelligence product. The public repository is not a robo-advisor and provides no personalized investment advice.

A safer commercialization path would be a private SaaS or API product offering risk diagnostics, model-governed volatility forecasts, stress reporting, and portfolio-risk analytics to regulated or sophisticated users. Any paid product that uses live data, user portfolios, or personalized recommendations should receive legal, compliance, data-license, security, and model-risk review before launch.

---

## Citation

A machine-readable citation is provided in [`CITATION.cff`](CITATION.cff). Please cite this study when referencing the methodology or the results.

```text
Archuleta, S., & Nilsen, A. (2026). Causal-Aware, Machine-Learning-Driven,
Factor-Informed Risk Forecasting: A Python Pipeline Integrating NLP,
Directed Factor Constraints, and Portfolio Analytics. WorldQuant University,
MScFE 690.
```

---

## License

Licensed under the Apache License 2.0. See [`LICENSE`](LICENSE) for full terms.

---

## Acknowledgements

- WorldQuant University, MScFE program
- The Kenneth French Data Library, FRED, and Alpha Vantage for the data access used in the research workflow
- Open-source maintainers of pandas, NumPy, scikit-learn, statsmodels, XGBoost, Streamlit, and yfinance
