"""
Project invariant tests for the riskml-capstone repository.

These tests are companions to tests/test_pipeline.py, not replacements.
The existing test_pipeline.py exercises executable code inside the
riskml/ package (storage, market_data, transforms). This file locks in
the project-wide constants and architectural invariants documented in
the M6 capstone report.

Each test asserts one frozen fact about the project. If any test fails,
either (a) a project invariant changed intentionally and this file was
not updated in the same commit, or (b) something drifted that should
not have. Either way the failure is immediately diagnostic.

The strategy is deliberately defensive: tests read the committed
notebook source code (as JSON files) and the committed CSV artifacts
directly. They do not import anything from the riskml/ package, so
they will run successfully even when the package subdirectories remain
empty stubs.

Authored 14 May 2026 — Phase 5 polish addition to the CI pipeline.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

# =====================================================================
# REPOSITORY ROOT RESOLUTION
# Tests live at REPO_ROOT/tests/test_invariants.py, so REPO_ROOT is
# two parents up from this file. The .resolve() call canonicalizes any
# symlinks so pytest works whether invoked from the repo root or from
# inside the tests/ directory.
# =====================================================================

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"
TABLES_DIR = REPO_ROOT / "reports" / "tables"


# =====================================================================
# HELPER: READ A JUPYTER NOTEBOOK AND RETURN ALL CODE-CELL SOURCE AS
# A SINGLE STRING. A .ipynb file is just JSON; this function does no
# execution, no kernel start, no imports of project code.
# =====================================================================

def _read_notebook_source(notebook_path: Path) -> str:
    notebook_json = json.loads(notebook_path.read_text(encoding="utf-8"))
    code_cells = [
        "".join(cell["source"])
        for cell in notebook_json["cells"]
        if cell["cell_type"] == "code"
    ]
    return "\n".join(code_cells)


# =====================================================================
# TEST 1 — RANDOM_SEED IS LOCKED AT 692 ACROSS NB01 THROUGH NB06
# Source of truth: NB01 line 149, NB02 line 138, NB03 line 175,
# NB04 line 160, NB05 line 120, NB06 line 114.
# Why this matters: the random seed is the project's reproducibility
# anchor. Every stochastic operation (numpy, random, torch) is seeded
# from this single value. Drift here invalidates every empirical claim
# in the capstone report.
# =====================================================================

def test_random_seed_is_locked_at_692_across_all_notebooks():
    notebook_names = [
        "01_data_extraction.ipynb",
        "02_feature_engineering_ml_factors.ipynb",
        "03_risk_forecasting_baselines.ipynb",
        "04_risk_forecasting_causal.ipynb",
        "05_portfolio_construction.ipynb",
        "06_validation_ablation_stress.ipynb",
    ]
    pattern = re.compile(r"RANDOM_SEED\s*=\s*(\d+)")

    for notebook_name in notebook_names:
        notebook_path = NOTEBOOKS_DIR / notebook_name
        assert notebook_path.exists(), (
            f"Notebook missing from repository: {notebook_path}"
        )

        notebook_source = _read_notebook_source(notebook_path)
        matched_values = pattern.findall(notebook_source)

        assert matched_values, (
            f"{notebook_name} does not declare RANDOM_SEED anywhere"
        )

        for declared_value in matched_values:
            assert declared_value == "692", (
                f"{notebook_name} declares RANDOM_SEED = {declared_value}; "
                "project standard is 692 across NB01 through NB06"
            )


# =====================================================================
# TEST 2 — CAUSAL_EDGES DICT IN NB04 ENCODES THE FIVE CORE EDGES
# Source of truth: NB04 lines 998 through 1004.
# The CAUSAL_EDGES dict carries five of the seven directed edges in
# the DAG. The other two (MACRO -> RISK, REGIME -> RISK) are encoded
# separately via the ALLOWED_PREFIXES_RISK list (see Test 4). Together
# they implement the full nine-node, seven-edge architecture
# documented in the report's Table 4.
# The five core edges:
#     SENTIMENT  -> MOMENTUM
#     MOMENTUM   -> RETURNS
#     VALUE      -> RETURNS
#     VOLATILITY -> RISK
#     RISK       -> ALLOCATION
# Why this matters: these five core edges describe the
# information-flow path that the risk-forecasting models actually
# follow. Drift here would change the computational meaning of the
# pipeline.
# =====================================================================

def test_causal_edges_dict_encodes_five_core_directed_edges():
    notebook_source = _read_notebook_source(
        NOTEBOOKS_DIR / "04_risk_forecasting_causal.ipynb"
    )

    required_edges = [
        ('"SENTIMENT": ["MOMENTUM"]'),
        ('"MOMENTUM": ["RETURNS"]'),
        ('"VALUE": ["RETURNS"]'),
        ('"VOLATILITY": ["RISK"]'),
        ('"RISK": ["ALLOCATION"]'),
    ]

    for edge_literal in required_edges:
        assert edge_literal in notebook_source, (
            f"NB04 CAUSAL_EDGES dict missing required edge: {edge_literal}"
        )


# =====================================================================
# TEST 3 — DAG FIGURE DEFINITION HAS NINE NODES AND SEVEN EDGES
# Source of truth: NB04 lines 2165 through 2200.
# This is the FULL DAG documented in the M6 capstone report. The
# figure code lists all nine nodes (SENTIMENT, MOMENTUM, RETURNS,
# VALUE, VOLATILITY, RISK, ALLOCATION, MACRO, REGIME) and all seven
# directed edges (the five core edges plus MACRO -> RISK and
# REGIME -> RISK).
# Why this matters: the report's Table 4 and the figure caption
# explicitly claim nine nodes and seven directed edges. This test
# locks the figure code to that claim. If a future notebook edit
# accidentally removed a node from the figure dictionary, the
# report's figure would no longer match the report's text.
# =====================================================================

def test_dag_figure_has_nine_nodes_and_seven_directed_edges():
    notebook_source = _read_notebook_source(
        NOTEBOOKS_DIR / "04_risk_forecasting_causal.ipynb"
    )

    figure_nodes = [
        "SENTIMENT", "MOMENTUM", "RETURNS",
        "VALUE", "VOLATILITY", "RISK", "ALLOCATION",
        "MACRO", "REGIME",
    ]

    for node_name in figure_nodes:
        position_literal = f'"{node_name}": ('
        assert position_literal in notebook_source, (
            f"NB04 dag_positions dict missing node {node_name}"
        )

    figure_edges = [
        ('("SENTIMENT", "MOMENTUM")'),
        ('("MOMENTUM", "RETURNS")'),
        ('("VALUE", "RETURNS")'),
        ('("VOLATILITY", "RISK")'),
        ('("RISK", "ALLOCATION")'),
        ('("MACRO", "RISK")'),
        ('("REGIME", "RISK")'),
    ]

    for edge_literal in figure_edges:
        assert edge_literal in notebook_source, (
            f"NB04 dag_edges_for_plot list missing edge: {edge_literal}"
        )


# =====================================================================
# TEST 4 — RISK-STAGE GATING BLOCKS MOMENTUM, VALUE, ML, AND SENTIMENT
# Source of truth: NB04 lines 1025 and 1026.
# ALLOWED_PREFIXES_RISK = ["VOL__", "MACRO__", "REGIME__"]
# FORBIDDEN_PREFIXES_RISK = ["MOM__", "VAL__", "ML__", "SENT__"]
# The allow-list encodes the MACRO -> RISK and REGIME -> RISK edges
# that complete the seven-edge architecture (the other five are in
# the CAUSAL_EDGES dict from Test 2). The forbid-list defends against
# silent leakage of forbidden feature families into the risk model.
# Why this matters: this is the central causal claim of the capstone
# project. If a future feature-engineering change added a MOM__
# feature to the allowed set, the H1 hypothesis would be invalidated
# and the capstone's core result would be wrong.
# =====================================================================

def test_risk_stage_gating_lists_match_report_specification():
    notebook_source = _read_notebook_source(
        NOTEBOOKS_DIR / "04_risk_forecasting_causal.ipynb"
    )

    allowed_assignment_literal = (
        'ALLOWED_PREFIXES_RISK = ["VOL__", "MACRO__", "REGIME__"]'
    )
    assert allowed_assignment_literal in notebook_source, (
        "NB04 ALLOWED_PREFIXES_RISK does not match expected value. "
        "Risk-stage gating must allow only VOL__, MACRO__, and "
        "REGIME__ prefixes."
    )

    forbidden_assignment_literal = (
        'FORBIDDEN_PREFIXES_RISK = ["MOM__", "VAL__", "ML__", "SENT__"]'
    )
    assert forbidden_assignment_literal in notebook_source, (
        "NB04 FORBIDDEN_PREFIXES_RISK does not match expected value. "
        "Risk-stage gating must forbid MOM__, VAL__, ML__, and "
        "SENT__ prefixes."
    )


# =====================================================================
# TEST 5 — RISK-STAGE FEATURE COUNT EQUALS 74
# Source of truth: NB04 line 2478 (explicit runtime guard) and the
# committed CSV artifact at
# reports/tables/notebook04_dag_gating_summary.csv (the materialized
# result of the last NB04 execution).
# The CSV has three columns: feature, dag_prefix, allowed_for_risk.
# Summing the boolean allowed_for_risk column gives the count of
# features that survive the gating step.
# Why this matters: 74 is the frozen empirical constant documented
# in the capstone report's feature-count narrative. NB04 itself
# contains a runtime guard comparing len(CAUSAL_MLP_FEATURE_COLS) to
# 74; this test makes the same assertion at commit time so the
# constant cannot drift unnoticed.
# =====================================================================

def test_risk_stage_feature_count_equals_74():
    gating_summary_path = TABLES_DIR / "notebook04_dag_gating_summary.csv"

    if not gating_summary_path.exists():
        pytest.skip(
            f"Gating summary artifact not present: {gating_summary_path}. "
            "Re-execute NB04 to materialize the artifact, then re-run "
            "this test."
        )

    import pandas as pd

    gating_df = pd.read_csv(gating_summary_path)

    required_columns = {"feature", "dag_prefix", "allowed_for_risk"}
    missing_columns = required_columns - set(gating_df.columns)
    assert not missing_columns, (
        f"Gating summary CSV is missing columns: {missing_columns}. "
        f"Expected columns: {required_columns}; "
        f"actual columns: {list(gating_df.columns)}"
    )

    # The allowed_for_risk column persists from pandas as either the
    # strings "True" / "False" or as native booleans depending on the
    # writer config. Normalize to a boolean Series before summing.
    allowed_series = gating_df["allowed_for_risk"]
    if allowed_series.dtype == object:
        allowed_series = allowed_series.astype(str).str.strip().eq("True")
    allowed_count = int(allowed_series.sum())

    assert allowed_count == 74, (
        f"Risk-stage feature count = {allowed_count}; expected 74. "
        "Either the gating logic in NB04 changed, or NB02 added or "
        "removed features in the upstream panel."
    )


# =====================================================================
# TEST 6 — ASSET_CLASS_MAP MAPS XLV (AND OTHER SECTOR ETFS) TO
# SECTOR_EQUITY
# Source of truth: NB04 lines 2039 through 2055.
# XLV is the sector-equity ticker that was discovered UNMAPPED early
# in the project history; its presence here with the SECTOR_EQUITY
# label is the regression guard against that bug recurring. The same
# rule applies to XLF, XLK, and XLE.
# Why this matters: the four sector-equity tickers drive a
# substantial fraction of the cross-sectional analysis in the
# capstone. A missing or mis-mapped sector ticker would produce
# silently wrong aggregations in the asset-class comparison table.
# =====================================================================

def test_asset_class_map_assigns_sector_equity_tickers_correctly():
    notebook_source = _read_notebook_source(
        NOTEBOOKS_DIR / "04_risk_forecasting_causal.ipynb"
    )

    sector_equity_tickers = ("XLF", "XLK", "XLE", "XLV")
    for ticker_symbol in sector_equity_tickers:
        ticker_assignment_literal = f'"{ticker_symbol}": "SECTOR_EQUITY"'
        assert ticker_assignment_literal in notebook_source, (
            f"NB04 ASSET_CLASS_MAP must map {ticker_symbol} to "
            "SECTOR_EQUITY"
        )


# =====================================================================
# TEST 7 — TICKER UNIVERSE IS THE FOURTEEN-ETF SET DEFINED IN NB01
# Source of truth: NB01 lines 266 through 271.
# The TICKERS list in NB01 drives every downstream computation:
# price extraction, return calculation, feature engineering, model
# fitting, portfolio construction, and validation. The set is locked
# at fourteen specific ETFs across five asset classes (US equity,
# international equity, sector equity, fixed income, commodity).
# Why this matters: a silent change to the ticker universe would
# alter every empirical claim in the capstone report. This test
# makes silent change impossible.
# =====================================================================

def test_ticker_universe_contains_fourteen_required_etfs():
    notebook_source = _read_notebook_source(
        NOTEBOOKS_DIR / "01_data_extraction.ipynb"
    )

    expected_tickers = {
        # US equity
        "SPY", "QQQ", "IWM",
        # International equity
        "EFA", "EEM",
        # Sector equity
        "XLK", "XLF", "XLE", "XLV",
        # Fixed income
        "TLT", "LQD", "HYG",
        # Commodity
        "GLD", "DBC",
    }

    for ticker_symbol in expected_tickers:
        ticker_literal = f'"{ticker_symbol}"'
        assert ticker_literal in notebook_source, (
            f"NB01 TICKERS list missing required ticker: {ticker_symbol}"
        )
