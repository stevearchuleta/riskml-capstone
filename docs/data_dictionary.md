# Data Dictionary - MScFE 692 Capstone

## Data Freeze Status

PENDING (target: end of Week 2)

## Datasets

### ETF Adjusted Close Prices

- File: data/raw/etf_prices.parquet
- Source: yfinance (fallback: Stooq via pandas_datareader)
- Frequency: Daily
- Start date: 2015-01-01 (adjustable)
- Tickers: SPY, QQQ, IWM, EFA, EEM, XLK, XLF, XLE, XLV, TLT, LQD, HYG, GLD, DBC
- Columns: Date (index), one column per ticker (adjusted close)
- Missing data rule: drop if more than 5 percent missing; forward-fill up to 3 days otherwise

### FRED Macro Series

- File: data/raw/macro_fred.parquet
- Source: FRED API
- Series: DTB3, VIXCLS, T10Y2Y
- Frequency: Daily (weekends/holidays missing)
- Missing data rule: forward-fill

### Fama-French Daily Factors

- File: data/raw/ff_factors.parquet
- Source: Fama-French Data Library via pandas_datareader
- Columns: Mkt-RF, SMB, HML, RF
- Units: percent to decimal (divide by 100)

### Sentiment (placeholder; Week 2)

- File: data/raw/sentiment_daily.parquet
- Source: RSS headlines via feedparser (Week 3-4)
- Columns: Date, Sentiment_Market

## Master Panel

- File: data/processed/master_panel.parquet
- Construction: trading-day calendar from ETF returns; left-join macro + factors
- Row count target: approximately 2,500 trading days (2015-2025)
