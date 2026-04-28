import pandas as pd
import numpy as np

def compute_realized_volatility(
    returns: pd.DataFrame,
    windows: tuple[int, ...] = (10, 21, 63),
    annualization_factor: int = 252,
) -> pd.DataFrame:
    """
    Compute annualized realized volatility features over multiple windows.

    Aligned with the Volatility node of the capstone DAG (Volatility → Risk →
    Allocation). Volatility features feed the risk forecasting stage in NB04
    as inputs to the risk model. Realized volatility is the rolling sample
    standard deviation of daily log returns, scaled by sqrt(annualization_factor)
    to express the result on an annualized basis.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily log returns. Rows are trading dates (DatetimeIndex), columns
        are asset tickers.
    windows : tuple[int, ...]
        Rolling window sizes in trading days. Defaults to (10, 21, 63),
        matching NB02's REALIZED_VOL_WINDOWS constant.
    annualization_factor : int
        Number of trading days per year used for annualization. Defaults to
        252, matching NB02's TRADING_DAYS_PER_YEAR constant.

    Returns
    -------
    pd.DataFrame
        Realized volatility feature matrix. Column naming convention follows
        the DAG-node prefix used throughout the capstone:
            VOL__<TICKER>__rvol__<window>d
        Index matches the input DataFrame; the first (window - 1) rows per
        window will contain NaN values from the rolling computation.

    Notes
    -----
    Uses ddof=1 (unbiased sample standard deviation) and min_periods=window
    (strict warmup, no early estimates), matching NB02 conventions. Output
    is on the annualized scale used for the 10% risk-targeting layer in NB05
    and for the forward realized volatility target in NB02.
    """
    if returns is None or returns.empty:
        raise ValueError("Empty or None returns DataFrame")

    annualization_multiplier = np.sqrt(annualization_factor)
    feature_blocks = []
    for window in windows:
        rolling_std = returns.rolling(window=window, min_periods=window).std(ddof=1)
        realized_vol = rolling_std * annualization_multiplier
        realized_vol.columns = [f"VOL__{col}__rvol__{window}d" for col in realized_vol.columns]
        feature_blocks.append(realized_vol)

    return pd.concat(feature_blocks, axis=1)




def compute_momentum_features(
    returns: pd.DataFrame,
    windows: tuple[int, ...] = (5, 10, 21, 63),
) -> pd.DataFrame:
    """
    Compute cumulative log-return (momentum) features over multiple windows.

    Aligned with the Momentum node of the capstone DAG. Inputs are assumed
    to be daily log returns; cumulative log returns over window w are computed
    as the rolling sum of log returns, which is mathematically equivalent to
    log(1 + cumulative_arithmetic_return) over the same window.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily log returns. Rows are trading dates (DatetimeIndex), columns
        are asset tickers.
    windows : tuple[int, ...]
        Rolling window sizes in trading days. Defaults to (5, 10, 21, 63),
        matching NB02's momentum feature family.

    Returns
    -------
    pd.DataFrame
        Momentum feature matrix. Column naming convention follows the
        DAG-node prefix used throughout the capstone:
            MOM__<TICKER>__cumret__<window>d
        Index matches the input DataFrame; the first (window - 1) rows
        per window will contain NaN values from the rolling computation.
    """
    if returns is None or returns.empty:
        raise ValueError("Empty or None returns DataFrame")

    feature_blocks = []
    for window in windows:
        cumret = returns.rolling(window).sum()
        cumret.columns = [f"MOM__{col}__cumret__{window}d" for col in cumret.columns]
        feature_blocks.append(cumret)

    return pd.concat(feature_blocks, axis=1)