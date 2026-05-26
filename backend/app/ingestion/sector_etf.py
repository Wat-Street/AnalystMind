# Sector ETF price data ingestion pipeline.
# Source: yfinance — fetches OHLCV for all 11 GICS sector ETFs (XLK, XLV, XLF, ...) and SPY.
# Computes: sector_rotation, sector_momentum, cross_sector_rank.
# Used by: macro_topdown persona.
