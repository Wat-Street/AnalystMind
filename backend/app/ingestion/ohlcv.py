"""OHLCV ingestion pipeline.

Source: yfinance (free, no API key).
Fetches daily Open, High, Low, Close, Volume bars and writes them to the
``ohlcv`` TimescaleDB hypertable.

Consumed by:
- ``quant_momentum`` persona   (momentum, vol regime, beta — 1M horizon)
- ``technical_analyst`` persona (RSI, MACD, breakout, volume — 3M horizon)
- ``options_flow_trader`` persona (cross-ref with price action — 2W horizon)
- ``technical_indicators.compute()`` (requires ≥60 daily rows)
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import yfinance as yf
from sqlalchemy.orm import Session

# ── Constants ────────────────────────────────────────────────────────────────
DEFAULT_PERIOD = "6mo"          # yfinance period string — ~126 trading days
DEFAULT_INTERVAL = "1d"         # daily bars
MIN_ROWS = 60                   # minimum rows for technical_indicators

_REQUIRED_YF_COLUMNS = ("Open", "High", "Low", "Close", "Volume")
_FINAL_COLUMNS = ("time", "open", "high", "low", "close", "volume", "adj_close")


# ── Pure helpers (no I/O — fully unit-tested) ────────────────────────────────

def _normalize_yf_history(raw: pd.DataFrame, ticker: str = "<unknown>") -> pd.DataFrame:
    """Take a raw yfinance ``history()`` DataFrame and return our standard shape.

    Steps:
      1. Validate required columns are present.
      2. Select only the columns we need.
      3. Drop rows where ``close`` is NaN (half-trading days).
      4. Fill ``volume`` NaN with 0 and coerce to int64.
      5. Fill ``adj_close`` NaN with ``close``.
      6. Reset the DatetimeIndex into a ``time`` column (keeps timezone).
      7. Sort ascending by time.

    Parameters
    ----------
    raw : pd.DataFrame
        Direct return value of ``yf.Ticker.history(auto_adjust=False)``.
    ticker : str
        Only used for the error message.

    Returns
    -------
    pd.DataFrame
        Columns: ``time, open, high, low, close, volume, adj_close``.
    """
    if not isinstance(raw, pd.DataFrame) or raw.empty:
        raise ValueError(f"yfinance returned empty history for {ticker!r}")

    missing = [c for c in _REQUIRED_YF_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(
            f"OHLCV dataframe missing required columns: {missing}"
        )

    # Work on a copy so we don't mutate the caller's frame.
    df = raw.copy()

    # Ensure adj_close exists — some yfinance versions omit it.
    if "Adj Close" not in df.columns:
        df["Adj Close"] = df["Close"]

    df = df[["Open", "High", "Low", "Close", "Volume", "Adj Close"]]

    # Drop rows where close is NaN (half-trading days, data gaps).
    df = df.dropna(subset=["Close"])

    # Fill remaining NaN: volume → 0, adj_close → close.
    df["Volume"] = df["Volume"].fillna(0).astype(np.int64)
    df["Adj Close"] = df["Adj Close"].fillna(df["Close"])

    # Reset the DatetimeIndex into a column named ``time``.
    df = df.reset_index(names=["time"])

    # Rename to lowercase to match our schema.
    df = df.rename(columns={
        "Open":      "open",
        "High":      "high",
        "Low":       "low",
        "Close":     "close",
        "Volume":    "volume",
        "Adj Close": "adj_close",
    })

    # Sort ascending by time (yfinance sometimes returns descending).
    df = df.sort_values("time").reset_index(drop=True)

    # Reorder columns to match the schema.
    df = df[list(_FINAL_COLUMNS)]

    return df


def _to_records(df: pd.DataFrame, ticker: str) -> list[dict]:
    """Convert a normalized OHLCV DataFrame into a list of row dicts for bulk insert."""
    records = []
    for _, row in df.iterrows():
        records.append({
            "ticker":    ticker,
            "time":      row["time"],
            "open":      float(row["open"]),
            "high":      float(row["high"]),
            "low":       float(row["low"]),
            "close":     float(row["close"]),
            "volume":    int(row["volume"]),
            "adj_close": float(row["adj_close"]),
        })
    return records


# ── Public API ───────────────────────────────────────────────────────────────

def fetch_history(
    ticker: str,
    period: str = DEFAULT_PERIOD,
    interval: str = DEFAULT_INTERVAL,
) -> pd.DataFrame:
    """Fetch OHLCV bars for *ticker* from yfinance and return a normalized DataFrame.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g. ``"AAPL"``).  Case-insensitive.
    period : str
        yfinance period string.  Default ``"6mo"`` (~126 trading days).
    interval : str
        yfinance interval string.  Default ``"1d"`` (daily).

    Returns
    -------
    pd.DataFrame
        Columns: ``time, open, high, low, close, volume, adj_close``.
        Sorted ascending by time.  Timezone-aware timestamps (e.g. ``America/New_York``).

    Raises
    ------
    ValueError
        If *ticker* is empty or yfinance returns no data.
    """
    if not ticker or not ticker.strip():
        raise ValueError("ticker is required")

    ticker = ticker.strip().upper()

    raw = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=False)
    if raw.empty:
        raise ValueError(f"no OHLCV data returned for ticker {ticker!r}")

    return _normalize_yf_history(raw, ticker=ticker)


def upsert_history(session: Session, ticker: str, df: pd.DataFrame) -> int:
    """Upsert rows from *df* into the ``ohlcv`` table keyed on ``(ticker, time)``.

    Commits the session.  Returns the number of rows upserted.
    """
    if df.empty:
        return 0

    from sqlalchemy.dialects.postgresql import insert as pg_insert
    from app.models.db import OHLCV

    records = _to_records(df, ticker)
    stmt = pg_insert(OHLCV).values(records)
    stmt = stmt.on_conflict_do_update(
        index_elements=["ticker", "time"],
        set_={
            "open":      stmt.excluded.open,
            "high":      stmt.excluded.high,
            "low":       stmt.excluded.low,
            "close":     stmt.excluded.close,
            "volume":    stmt.excluded.volume,
            "adj_close": stmt.excluded.adj_close,
        },
    )
    session.execute(stmt)
    session.commit()
    return len(records)


# ── Smoke test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    t = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    df = fetch_history(t)
    print(f"Ticker: {t}  —  {len(df)} rows fetched")
    print(df.tail())
