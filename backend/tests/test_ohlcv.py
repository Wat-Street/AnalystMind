"""Tests for the OHLCV ingestion pipeline.

All tests use synthetic data — no network, no DB, no yfinance.
Tests exercise the pure helpers directly: ``_normalize_yf_history`` and ``_to_records``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.ingestion.ohlcv import (
    DEFAULT_INTERVAL,
    DEFAULT_PERIOD,
    MIN_ROWS,
    _normalize_yf_history,
    _to_records,
    fetch_history,
)


# ── Test fixtures ────────────────────────────────────────────────────────────

def _yf_frame(
    close: np.ndarray,
    volume: np.ndarray | None = None,
    index: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """Build a yfinance-shaped DataFrame for tests.

    Mimics ``yf.Ticker.history(auto_adjust=False)`` output:
      - Index: tz-aware DatetimeIndex named ``Date``
      - Columns: Open, High, Low, Close, Adj Close, Volume, Dividends, Stock Splits
    """
    n = len(close)
    if volume is None:
        volume = np.full(n, 1_000_000, dtype=np.int64)
    if index is None:
        index = pd.date_range("2024-01-01", periods=n, freq="D", tz="America/New_York")

    return pd.DataFrame(
        {
            "Open":         close,
            "High":         close * 1.005,
            "Low":          close * 0.995,
            "Close":        close,
            "Adj Close":    close,
            "Volume":       volume,
            "Dividends":    np.zeros(n),
            "Stock Splits": np.zeros(n),
        },
        index=index,
    )


# ── _normalize_yf_history tests ──────────────────────────────────────────────

class TestNormalizeYfHistory:
    """Unit tests for the yfinance→schema normalization function."""

    def test_renames_columns_to_lowercase(self):
        df = _normalize_yf_history(_yf_frame(np.linspace(100, 110, MIN_ROWS)))
        assert list(df.columns) == [
            "time", "open", "high", "low", "close", "volume", "adj_close"
        ]

    def test_drops_nan_close_rows(self):
        """Half-trading days produce NaN close — those rows must go."""
        close = np.linspace(100, 110, MIN_ROWS)
        close[5] = np.nan
        close[10] = np.nan
        df = _normalize_yf_history(_yf_frame(close))
        assert len(df) == MIN_ROWS - 2
        assert df["close"].isna().sum() == 0

    def test_sorts_ascending_by_time(self):
        """yfinance sometimes returns newest-first; we always sort ascending."""
        raw = _yf_frame(np.linspace(100, 110, MIN_ROWS))
        raw = raw.iloc[::-1]  # reverse order
        df = _normalize_yf_history(raw)
        assert df["time"].is_monotonic_increasing

    def test_resets_index_to_range(self):
        """After reset, index should be a simple 0..N-1 range."""
        df = _normalize_yf_history(_yf_frame(np.linspace(100, 110, MIN_ROWS)))
        assert list(df.index) == list(range(len(df)))

    def test_coerces_volume_to_int64(self):
        """yfinance sometimes returns volume as float — we coerce to int64."""
        raw = _yf_frame(np.linspace(100, 110, MIN_ROWS))
        raw["Volume"] = raw["Volume"].astype(float)
        df = _normalize_yf_history(raw)
        assert pd.api.types.is_integer_dtype(df["volume"])

    def test_fills_missing_adj_close_with_close(self):
        raw = _yf_frame(np.linspace(100, 110, MIN_ROWS))
        raw["Adj Close"] = np.nan
        df = _normalize_yf_history(raw)
        assert (df["adj_close"] == df["close"]).all()

    def test_preserves_timezone(self):
        """Timestamps should remain timezone-aware for the TIMESTAMPTZ column."""
        df = _normalize_yf_history(_yf_frame(np.linspace(100, 110, MIN_ROWS)))
        assert df["time"].dt.tz is not None

    def test_drops_dividends_and_stock_splits_columns(self):
        raw = _yf_frame(np.linspace(100, 110, MIN_ROWS))
        assert "Dividends" in raw.columns
        df = _normalize_yf_history(raw)
        assert "Dividends" not in df.columns
        assert "Stock Splits" not in df.columns

    def test_raises_on_missing_required_columns(self):
        bad = pd.DataFrame({"Open": [1.0], "Close": [1.0]})
        with pytest.raises(ValueError, match="missing required columns"):
            _normalize_yf_history(bad)

    def test_raises_on_empty_dataframe(self):
        empty = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        with pytest.raises(ValueError, match="empty"):
            _normalize_yf_history(empty)


# ── _to_records tests ────────────────────────────────────────────────────────

class TestToRecords:
    """Unit tests for the DataFrame→row-dict conversion."""

    def test_returns_correct_number_of_records(self):
        df = _normalize_yf_history(_yf_frame(np.linspace(100, 110, MIN_ROWS)))
        records = _to_records(df, "AAPL")
        assert len(records) == len(df)

    def test_record_has_expected_keys(self):
        df = _normalize_yf_history(_yf_frame(np.linspace(100, 110, MIN_ROWS)))
        sample = _to_records(df, "MSFT")[0]
        expected = {"ticker", "time", "open", "high", "low", "close", "volume", "adj_close"}
        assert set(sample.keys()) == expected

    def test_ticker_propagated_to_all_records(self):
        df = _normalize_yf_history(_yf_frame(np.linspace(100, 110, MIN_ROWS)))
        records = _to_records(df, "TSLA")
        assert all(r["ticker"] == "TSLA" for r in records)

    def test_price_values_are_native_float(self):
        df = _normalize_yf_history(_yf_frame(np.linspace(100, 110, MIN_ROWS)))
        sample = _to_records(df, "AAPL")[0]
        for key in ("open", "high", "low", "close", "adj_close"):
            assert isinstance(sample[key], float), f"{key} should be float"

    def test_volume_is_native_int(self):
        df = _normalize_yf_history(_yf_frame(np.linspace(100, 110, MIN_ROWS)))
        sample = _to_records(df, "AAPL")[0]
        assert isinstance(sample["volume"], int)


# ── fetch_history error-path tests ───────────────────────────────────────────

class TestFetchHistoryErrors:
    """Test the validation layer (no network call — uses monkeypatch)."""

    def test_raises_on_empty_ticker(self):
        with pytest.raises(ValueError, match="ticker"):
            fetch_history("")

    def test_raises_on_none_ticker(self):
        with pytest.raises(ValueError, match="ticker"):
            fetch_history(None)  # type: ignore[arg-type]

    def test_raises_on_whitespace_ticker(self):
        with pytest.raises(ValueError, match="ticker"):
            fetch_history("   ")
