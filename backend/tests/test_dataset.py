"""Tests for ML dataset assembly.

All tests use synthetic data — no network, no DB.
Tests exercise the pure builders directly: price/macro/sentiment feature
derivation, joins, imputation, time split, and the torch Dataset wrapper.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from app.ml.dataset import (
    BENCHMARK_TICKER,
    MACRO_CHANGE_WINDOW,
    TRAIN_END,
    TRANSCRIPT_AVAILABILITY_LAG_DAYS,
    VAL_END,
    AnalystMindDataset,
    _join_fundamentals,
    _join_sentiment,
    _macro_features,
    _max_drawdown,
    _price_features,
    _quarter_availability_date,
    _sentiment_features,
    impute_features,
    split_by_time,
)
from app.ml.features import FEATURE_ORDER, HORIZONS
from app.ml.labels import LABEL_COLUMNS


# ── Fixtures ─────────────────────────────────────────────────────────────────

N_DAYS = 200


def make_bars(ticker: str, n: int = N_DAYS, start_price: float = 100.0, step: float = 0.5) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=n)
    price = start_price + step * np.arange(n)
    return pd.DataFrame({
        "ticker": ticker,
        "date": dates,
        "close": price,
        "volume": np.full(n, 1_000_000.0),
        "adj_close": price,
    })


def make_panel(tickers: list[str], n: int = N_DAYS) -> pd.DataFrame:
    """Minimal (ticker, date) panel for join tests."""
    frames = [pd.DataFrame({"ticker": t, "date": pd.bdate_range("2020-01-01", periods=n)}) for t in tickers]
    return pd.concat(frames, ignore_index=True)


# ── price-derived features ───────────────────────────────────────────────────

def test_price_features_momentum_and_bounds():
    bars = pd.concat([make_bars("AAPL"), make_bars(BENCHMARK_TICKER, step=0.1)], ignore_index=True)
    out = _price_features(bars)

    assert set(out["ticker"]) == {"AAPL"}  # benchmark excluded from panel

    aapl = out.sort_values("date").reset_index(drop=True)
    # return_21d on adj_close: price[21]=110.5, price[0]=100
    assert aapl["return_21d"].iloc[21] == pytest.approx(110.5 / 100.0 - 1.0)
    assert aapl["return_21d"].iloc[:21].isna().all()  # warmup stays NaN

    for col in ("rsi_signal", "macd_signal", "breakout_score", "volume_surge"):
        valid = aapl[col].dropna()
        assert not valid.empty
        assert valid.between(0.0, 1.0).all()

    # monotonically rising series → drawdown is 0 once the window is full
    assert aapl["max_drawdown_63d"].iloc[-1] == pytest.approx(0.0)
    assert aapl["realized_volatility_21d"].dropna().ge(0).all()


def test_price_features_relative_strength_and_beta():
    bars = pd.concat([make_bars("AAPL"), make_bars(BENCHMARK_TICKER)], ignore_index=True)
    out = _price_features(bars).sort_values("date").reset_index(drop=True)

    # AAPL and SPY are identical series → relative strength 0, beta ~1
    np.testing.assert_allclose(out["market_relative_strength_63d"].dropna(), 0.0, atol=1e-12)
    assert out["beta_126d"].iloc[-1] == pytest.approx(1.0)


def test_price_features_without_benchmark_are_nan():
    out = _price_features(make_bars("AAPL"))
    assert out["market_relative_strength_63d"].isna().all()
    assert out["beta_126d"].isna().all()
    assert out["return_21d"].notna().sum() > 0  # everything else still computed


def test_max_drawdown_helper():
    window = np.array([100.0, 120.0, 90.0, 110.0])
    assert _max_drawdown(window) == pytest.approx(90.0 / 120.0 - 1.0)


# ── fundamentals join ────────────────────────────────────────────────────────

def test_join_fundamentals_asof_and_backfill():
    panel = make_panel(["AAPL"], n=10)
    snap_date = panel["date"].iloc[5]
    fundamentals = pd.DataFrame({
        "ticker": ["AAPL"],
        "fetched_at": [snap_date],
        "fcf_yield": [0.05], "trailing_pe": [30.0], "forward_pe": [25.0],
        "ev_ebitda": [20.0], "revenue_cagr_3y": [0.1], "gross_margin": [0.4],
    })
    merged = _join_fundamentals(panel, fundamentals)

    # as-of after the snapshot and backfilled before it
    assert (merged["fcf_yield"] == 0.05).all()


def test_join_fundamentals_empty_is_noop():
    panel = make_panel(["AAPL"], n=5)
    merged = _join_fundamentals(panel, pd.DataFrame())
    assert "fcf_yield" not in merged.columns


# ── sentiment features ───────────────────────────────────────────────────────

def make_sentiment(ticker: str = "AAPL") -> pd.DataFrame:
    return pd.DataFrame({
        "ticker": ticker,
        "quarter": ["2020Q1"] * 3 + ["2020Q2"] * 2,
        "segment_role": ["management", "management", "qa", "management", "qa"],
        "label": ["positive", "negative", "positive", "positive", "neutral"],
        "score": [0.9, 0.5, 0.8, 0.6, 0.7],
    })


def test_sentiment_signing_gap_and_qoq():
    features = _sentiment_features(make_sentiment())

    q1 = features.iloc[0]
    assert q1["mgmt_sentiment_score"] == pytest.approx((0.9 - 0.5) / 2)   # signed mean
    assert q1["qa_sentiment_score"] == pytest.approx(0.8)
    assert q1["mgmt_qa_sentiment_gap"] == pytest.approx(0.2 - 0.8)
    assert np.isnan(q1["transcript_sentiment_change_qoq"])                # first quarter

    q2 = features.iloc[1]
    assert q2["mgmt_sentiment_score"] == pytest.approx(0.6)
    assert q2["qa_sentiment_score"] == pytest.approx(0.0)                 # neutral → 0
    assert q2["transcript_sentiment_change_qoq"] == pytest.approx(0.6 - 0.2)


def test_quarter_availability_date():
    available = _quarter_availability_date("2020Q1")
    assert available == pd.Timestamp("2020-03-31") + pd.Timedelta(days=TRANSCRIPT_AVAILABILITY_LAG_DAYS)
    assert _quarter_availability_date("garbage") is None


def test_join_sentiment_is_point_in_time():
    panel = make_panel(["AAPL"], n=N_DAYS)  # Jan–Oct 2020
    merged = _join_sentiment(panel, make_sentiment())

    q1_available = _quarter_availability_date("2020Q1")
    before = merged[merged["date"] < q1_available]
    after = merged[merged["date"] >= q1_available]
    assert before["mgmt_sentiment_score"].isna().all()  # no lookahead
    assert after["mgmt_sentiment_score"].notna().all()


# ── macro features ───────────────────────────────────────────────────────────

def test_macro_features_asof_and_derived():
    dates = pd.bdate_range("2020-01-01", periods=MACRO_CHANGE_WINDOW + 20)
    macro = pd.DataFrame({
        "series_id": ["FEDFUNDS", "FEDFUNDS", "DGS10", "CPI_YOY"],
        "observation_date": [dates[0], dates[MACRO_CHANGE_WINDOW + 5], dates[0], dates[0]],
        "value": [1.5, 2.0, 4.0, 3.0],
    })
    out = _macro_features(macro, dates)

    assert out.loc[dates[1], "fed_funds_rate"] == 1.5                       # ffill between obs
    assert out.loc[dates[MACRO_CHANGE_WINDOW + 6], "fed_funds_rate"] == 2.0
    # change over 63 trading days after the hike: 2.0 - 1.5
    assert out.loc[dates[MACRO_CHANGE_WINDOW + 6], "fed_funds_change_3m"] == pytest.approx(0.5)
    assert out.loc[dates[0], "real_treasury_10y"] == pytest.approx(4.0 - 3.0)
    assert out["unemployment"].isna().all()                                 # series absent


def test_macro_features_empty_table():
    dates = pd.bdate_range("2020-01-01", periods=5)
    out = _macro_features(pd.DataFrame(columns=["series_id", "observation_date", "value"]), dates)
    assert out.isna().all().all()
    assert len(out) == 5


# ── imputation and split ─────────────────────────────────────────────────────

def test_impute_features_uses_given_medians_and_flags():
    features = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [np.nan, np.nan, np.nan]})
    medians = pd.Series({"a": 2.0, "b": np.nan})
    imputed, mask = impute_features(features, medians)

    assert imputed["a"].tolist() == [1.0, 2.0, 3.0]
    assert (imputed["b"] == 0.0).all()                  # all-NaN column → 0.0
    assert mask["a"].tolist() == [0.0, 1.0, 0.0]
    assert (mask["b"] == 1.0).all()


def test_split_by_time_boundaries():
    dates = pd.to_datetime(["2022-12-30", "2022-12-31", "2023-06-01", "2023-12-31", "2024-01-02"])
    frame = pd.DataFrame(
        {"x": range(5)},
        index=pd.MultiIndex.from_product([["AAPL"], dates], names=["ticker", "date"]),
    )
    splits = split_by_time(frame)

    assert len(splits["train"]) == 2
    assert len(splits["val"]) == 2
    assert len(splits["test"]) == 1
    assert splits["train"].index.get_level_values("date").max() <= TRAIN_END
    assert splits["val"].index.get_level_values("date").max() <= VAL_END


# ── AnalystMindDataset ───────────────────────────────────────────────────────

def make_aligned_frames(n: int = 4):
    index = pd.MultiIndex.from_product(
        [["AAPL"], pd.bdate_range("2020-01-01", periods=n)], names=["ticker", "date"]
    )
    features = pd.DataFrame(0.5, index=index, columns=list(FEATURE_ORDER))
    labels = pd.DataFrame(0.1, index=index, columns=list(LABEL_COLUMNS))
    return features, labels


def test_dataset_shapes_and_dtypes():
    features, labels = make_aligned_frames()
    ds = AnalystMindDataset(features, labels)

    assert len(ds) == 4
    x, y = ds[0]
    assert x.shape == (len(FEATURE_ORDER),)
    assert y.shape == (len(HORIZONS),)
    assert x.dtype == torch.float32 and y.dtype == torch.float32
    assert ds.missing_mask.shape == (4, len(FEATURE_ORDER))
    assert (ds.missing_mask == 0).all()


def test_dataset_validates_contracts():
    features, labels = make_aligned_frames()

    with pytest.raises(ValueError, match="FEATURE_ORDER"):
        AnalystMindDataset(features[list(FEATURE_ORDER[:5])], labels)

    shifted = labels.copy()
    shifted.index = pd.MultiIndex.from_product(
        [["MSFT"], pd.bdate_range("2020-01-01", periods=4)], names=["ticker", "date"]
    )
    with pytest.raises(ValueError, match="same"):
        AnalystMindDataset(features, shifted)
