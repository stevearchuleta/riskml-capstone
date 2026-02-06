# Project Charter - MScFE 692 Capstone

## Title

Causal-Aware, Machine-Learning-Driven Risk Forecasting and Factor Construction

## Author

Steven "Steve" Archuleta - WorldQuant University

## Research Question

How does imposing manual causal constraints on an ML-driven risk forecasting
pipeline affect forecast accuracy, portfolio performance, and interpretability
compared to unconstrained baselines?

## Hypotheses

- H1: DAG-constrained pipeline reduces forecast error (MAE, RMSE) versus baseline
- H2: DAG-constrained allocation reduces drawdown and improves risk targeting
- H3: NLP sentiment improves near-term risk control (ablation test)
- H4: Factor-exposure entropy increases under DAG constraints

## Success Metrics (minimum)

- Ablation table showing marginal contribution of each module
- Walk-forward out-of-sample evaluation (no leakage)
- Reproducible one-command pipeline run
- Stress test results under at least 2 scenarios
- Final report (APA 7 format) and slides (defense format)

## Non-Negotiables

1. Time-ordered splits (no future leakage)
2. Reproducible environment (Conda + later Docker)
3. CI passing on main branch
4. Azure monthly budget cap: USD 20

## Out of Scope

- Full causal discovery at scale (PC/GES/NOTEARS across hundreds of variables)
- Full knowledge graph construction
- Multi-agent LLM causal systems
- High-frequency trading execution
- Multivariate GARCH

## Decision Log

| Date       | Decision        | Rationale           | Rollback Plan            |
|------------|-----------------|---------------------|--------------------------|
| 2026-02-05 | Manual DAG only | Scope and timeline  | Add PC as appendix only  |
