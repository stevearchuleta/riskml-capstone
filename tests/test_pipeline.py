"""
PIPELINE TESTS
Real CI signal — exercises three extracted package functions:
  riskml.storage.load_parquet
  riskml.etl.market_data.extract_price_panel
  riskml.features.transforms.compute_momentum_features

Tests are deterministic (fixed seed), self-contained (tmp_path),
and exercise realistic input shapes (MultiIndex for yfinance).
"""

import numpy as np
import pandas as pd
import pytest

from riskml.storage import load_parquet
from riskml.etl.market_data import extract_price_panel
from riskml.features.transforms import (
    compute_momentum_features,
    compute_realized_volatility,
)


# ---------------------------------------------------------------------------
# Test 1 — storage.load_parquet local fallback
# ---------------------------------------------------------------------------
def test_storage_local_fallback(tmp_path):
    """load_parquet returns a DataFrame from a local path when no Azure env var is set."""
    df = pd.DataFrame({"price": [100.0, 101.5, 99.75], "volume": [1000, 1500, 1200]})
    fixture_path = tmp_path / "sample.parquet"
    df.to_parquet(fixture_path)

    loaded = load_parquet(str(fixture_path))

    assert isinstance(loaded, pd.DataFrame)
    assert loaded.shape == (3, 2)
    assert list(loaded.columns) == ["price", "volume"]
    assert loaded["price"].iloc[0] == 100.0


# ---------------------------------------------------------------------------
# Test 2 — extract_price_panel handles realistic yfinance MultiIndex output
# ---------------------------------------------------------------------------
def test_extract_price_panel_multiindex():
    """extract_price_panel correctly extracts the Adj Close level from yfinance MultiIndex output."""
    dates = pd.date_range("2024-01-02", periods=5, freq="B")
    columns = pd.MultiIndex.from_product(
        [["Adj Close", "Volume"], ["SPY", "QQQ"]],
        names=["Price", "Ticker"],
    )
    raw = pd.DataFrame(
        [
            [400.0, 380.0, 1_000_000, 800_000],
            [401.0, 381.5, 1_100_000, 850_000],
            [402.5, 382.0, 1_050_000, 820_000],
            [400.5, 380.5, 1_200_000, 900_000],
            [403.0, 383.5, 1_150_000, 870_000],
        ],
        index=dates,
        columns=columns,
    )

    panel = extract_price_panel(raw)

    assert isinstance(panel, pd.DataFrame)
    assert panel.shape == (5, 2)
    assert sorted(panel.columns.tolist()) == ["QQQ", "SPY"]
    assert panel["SPY"].iloc[0] == 400.0
    assert not panel.isna().any().any()


# ---------------------------------------------------------------------------
# Test 3 — compute_momentum_features produces correctly-shaped DAG-aligned output
# ---------------------------------------------------------------------------
def test_compute_momentum_features():
    """Momentum features have correct shape, column naming, and finite values past warmup."""
    rng = np.random.default_rng(692)
    n_days = 100
    returns = pd.DataFrame(
        {
            "SPY": rng.standard_normal(n_days) * 0.01,
            "QQQ": rng.standard_normal(n_days) * 0.012,
        },
        index=pd.date_range("2024-01-02", periods=n_days, freq="B"),
    )

    features = compute_momentum_features(returns, windows=(5, 21))

    assert isinstance(features, pd.DataFrame)
    assert features.shape == (100, 4)
    expected_cols = {
        "MOM__SPY__cumret__5d",
        "MOM__QQQ__cumret__5d",
        "MOM__SPY__cumret__21d",
        "MOM__QQQ__cumret__21d",
    }
    assert set(features.columns) == expected_cols
    assert features.iloc[:4].isna().all().all()
    assert features.iloc[21:].notna().all().all()
    assert np.isfinite(features.iloc[21:].values).all()


# ---------------------------------------------------------------------------
# Test 4 — compute_momentum_features defensive input validation
# ---------------------------------------------------------------------------
def test_compute_momentum_features_rejects_empty():
    """Empty input raises ValueError (defensive contract)."""
    with pytest.raises(ValueError, match="Empty"):
        compute_momentum_features(pd.DataFrame())


# ---------------------------------------------------------------------------
# Test 5 — compute_realized_volatility produces correctly-scaled DAG-aligned output
# ---------------------------------------------------------------------------
def test_compute_realized_volatility():
    """Realized volatility features have correct shape, naming, scale, and finite values."""
    rng = np.random.default_rng(692)
    n_days = 100
    daily_sigma = 0.01
    returns = pd.DataFrame(
        {
            "SPY": rng.standard_normal(n_days) * daily_sigma,
            "QQQ": rng.standard_normal(n_days) * daily_sigma,
        },
        index=pd.date_range("2024-01-02", periods=n_days, freq="B"),
    )

    features = compute_realized_volatility(returns, windows=(10, 21))

    assert isinstance(features, pd.DataFrame)
    assert features.shape == (100, 4)
    expected_cols = {
        "VOL__SPY__rvol__10d",
        "VOL__QQQ__rvol__10d",
        "VOL__SPY__rvol__21d",
        "VOL__QQQ__rvol__21d",
    }
    assert set(features.columns) == expected_cols

    # Strict warmup: first (window - 1) rows must be NaN
    assert features["VOL__SPY__rvol__10d"].iloc[:9].isna().all()
    assert features["VOL__SPY__rvol__21d"].iloc[:20].isna().all()

    # Past warmup: all values must be finite and positive
    post_warmup = features.iloc[21:]
    assert post_warmup.notna().all().all()
    assert np.isfinite(post_warmup.values).all()
    assert (post_warmup.values > 0).all()

    # Annualization sanity: with daily sigma ~0.01, annualized sigma ~0.01 * sqrt(252) ≈ 0.159
    # Allow a wide tolerance (random sample of 100 days has noise around the population value)
    mean_annualized_vol = post_warmup["VOL__SPY__rvol__21d"].mean()
    expected_scale = daily_sigma * np.sqrt(252)
    assert 0.10 < mean_annualized_vol < 0.25, (
        f"Annualized vol {mean_annualized_vol:.4f} outside expected range "
        f"around {expected_scale:.4f}"
    )