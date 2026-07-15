"""Tests for forward-return label computation.

All tests use synthetic data — no network, no DB.
Tests exercise the pure helpers directly: ``compute_forward_returns`` and
``normalize_trading_date``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.ml.features import HORIZONS
from app.ml.labels import (
    HORIZON_LABELS,
    LABEL_COLUMNS,
    compute_forward_returns,
    label_column,
    normalize_trading_date,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

def make_prices(ticker: str, n: int, start_price: float = 100.0, step: float = 1.0) -> pd.DataFrame:
    """Linear price series: adj_close[t] = start_price + step * t."""
    dates = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame({
        "ticker": ticker,
        "date": dates,
        "adj_close": start_price + step * np.arange(n),
    })


# ── label naming ─────────────────────────────────────────────────────────────

def test_label_columns_track_horizons():
    assert LABEL_COLUMNS == [f"fwd_return_{h}d" for h in HORIZONS]
    assert HORIZON_LABELS == {h: label_column(h) for h in HORIZONS}


# ── compute_forward_returns ──────────────────────────────────────────────────

def test_forward_return_values():
    n = max(HORIZONS) + 50
    prices = make_prices("AAPL", n)
    labels = compute_forward_returns(prices)

    for h in HORIZONS:
        expected = (100.0 + h) / 100.0 - 1.0  # price 100 → 100 + h after h steps
        got = labels.loc[("AAPL", prices["date"].iloc[0]), label_column(h)]
        assert got == pytest.approx(expected)


def test_tail_rows_have_nan_labels():
    n = max(HORIZONS) + 10
    labels = compute_forward_returns(make_prices("AAPL", n))
    for h in HORIZONS:
        col = labels[label_column(h)]
        assert col.tail(h).isna().all()
        assert col.head(n - h).notna().all()


def test_no_cross_ticker_bleed():
    # B's history is far shorter than the max horizon — every B label at the
    # longest horizon must be NaN, never computed from A's rows.
    a = make_prices("AAA", max(HORIZONS) + 10, start_price=100.0)
    b = make_prices("BBB", 30, start_price=1000.0)
    labels = compute_forward_returns(pd.concat([a, b], ignore_index=True))

    assert labels.loc["BBB", label_column(max(HORIZONS))].isna().all()
    # 10-day label for B computed from B's own prices
    got = labels.loc[("BBB", b["date"].iloc[0]), label_column(10)]
    assert got == pytest.approx((1010.0 - 1000.0) / 1000.0)


def test_non_positive_prices_become_nan():
    prices = make_prices("AAPL", 40)
    prices.loc[15, "adj_close"] = 0.0        # bad tick
    labels = compute_forward_returns(prices, horizons=[10])

    col = labels[label_column(10)]
    assert np.isnan(col.iloc[15])            # base price invalid
    assert np.isnan(col.iloc[5])             # future price (t=15) invalid
    assert col.iloc[0] == pytest.approx(110.0 / 100.0 - 1.0)


def test_duplicate_dates_keep_last():
    prices = make_prices("AAPL", 20)
    dup = prices.iloc[[0]].assign(adj_close=200.0)
    labels = compute_forward_returns(pd.concat([prices, dup], ignore_index=True), horizons=[10])

    got = labels.loc[("AAPL", prices["date"].iloc[0]), label_column(10)]
    assert got == pytest.approx(110.0 / 200.0 - 1.0)


def test_unsorted_input_is_handled():
    prices = make_prices("AAPL", 40).sample(frac=1.0, random_state=7)
    labels = compute_forward_returns(prices, horizons=[10])
    got = labels.loc[("AAPL", pd.Timestamp("2020-01-01")), label_column(10)]
    assert got == pytest.approx(110.0 / 100.0 - 1.0)


def test_missing_columns_raise():
    with pytest.raises(ValueError, match="missing required columns"):
        compute_forward_returns(pd.DataFrame({"ticker": [], "close": []}))


def test_empty_input_returns_empty_frame():
    empty = pd.DataFrame(columns=["ticker", "date", "adj_close"])
    labels = compute_forward_returns(empty)
    assert labels.empty
    assert list(labels.columns) == LABEL_COLUMNS


# ── normalize_trading_date ───────────────────────────────────────────────────

def test_normalize_naive_timestamps():
    ts = pd.Series(pd.to_datetime(["2024-01-02 09:30:00", "2024-01-03 16:00:00"]))
    out = normalize_trading_date(ts)
    assert list(out) == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]


def test_normalize_tz_aware_uses_market_calendar():
    # Midnight NY stored as 05:00 UTC must stay on the NY calendar date.
    ts = pd.Series(pd.to_datetime(["2024-01-02 05:00:00"]).tz_localize("UTC"))
    out = normalize_trading_date(ts)
    assert out.iloc[0] == pd.Timestamp("2024-01-02")
    assert out.dt.tz is None
