import pandas as pd


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