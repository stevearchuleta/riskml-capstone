import pandas as pd
import yfinance as yf


def extract_price_panel(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Extract adjusted close (or close) price panel from yfinance output.
    """
    if raw is None or raw.shape[0] == 0:
        raise ValueError("Empty dataframe returned from yfinance")

    if isinstance(raw.columns, pd.MultiIndex):
        if "Adj Close" in raw.columns.get_level_values(0):
            return raw["Adj Close"].copy()
        elif "Close" in raw.columns.get_level_values(0):
            return raw["Close"].copy()
        else:
            raise ValueError("Missing expected price column")
    else:
        if "Adj Close" in raw.columns:
            return raw[["Adj Close"]].copy()
        elif "Close" in raw.columns:
            return raw[["Close"]].copy()
        else:
            raise ValueError("Missing expected price column")


def download_etf_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """
    Download ETF price data and return clean price panel.
    """
    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False
    )

    prices = extract_price_panel(raw)
    prices = prices.dropna(how="all")

    if prices.shape[0] == 0:
        raise ValueError("No valid price data after cleaning")

    return prices