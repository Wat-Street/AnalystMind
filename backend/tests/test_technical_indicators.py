from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.ingestion.technical_indicators import MIN_ROWS, compute


def _frame(close: np.ndarray, volume: np.ndarray | None = None) -> pd.DataFrame:
    n = len(close)
    if volume is None:
        volume = np.full(n, 1_000_000, dtype=np.int64)
    return pd.DataFrame({
        "time": pd.date_range("2024-01-01", periods=n, freq="D"),
        "open":  close,
        "high":  close * 1.005,
        "low":   close * 0.995,
        "close": close,
        "volume": volume,
        "adj_close": close,
    })


def test_uptrend_bullish_signals():
    # Compounding (slightly accelerating) growth so MACD histogram is positive.
    close = 100.0 * (1.01 ** np.arange(120))
    out = compute(_frame(close))
    assert out["rsi_signal"] > 0.7
    assert out["macd_signal"] > 0.5
    assert out["breakout_score"] == pytest.approx(1.0, abs=1e-6)


def test_downtrend_bearish_signals():
    close = 200.0 * (0.99 ** np.arange(120))
    out = compute(_frame(close))
    assert out["rsi_signal"] < 0.3
    assert out["macd_signal"] < 0.5
    assert out["breakout_score"] == pytest.approx(0.0, abs=1e-6)


def test_volume_spike_saturates():
    close = np.full(120, 100.0)
    volume = np.full(120, 1_000_000, dtype=np.int64)
    volume[-1] = 10_000_000  # 10x last-day spike, well past 3x saturation
    out = compute(_frame(close, volume))
    assert out["volume_surge"] == pytest.approx(1.0, abs=1e-6)


def test_output_contract():
    close = 100 + np.cumsum(np.random.default_rng(0).normal(0, 1, 120))
    out = compute(_frame(close))
    assert set(out.keys()) == {"rsi_signal", "macd_signal", "breakout_score", "volume_surge"}
    for k, v in out.items():
        assert isinstance(v, float), f"{k} is {type(v).__name__}, not float"
        assert not np.isnan(v), f"{k} is NaN"
        assert 0.0 <= v <= 1.0, f"{k}={v} outside [0,1]"


def test_validation_too_few_rows():
    close = np.linspace(100.0, 110.0, MIN_ROWS - 1)
    with pytest.raises(ValueError, match="rows"):
        compute(_frame(close))


def test_validation_missing_column():
    close = np.linspace(100.0, 110.0, 120)
    df = _frame(close).drop(columns=["volume"])
    with pytest.raises(ValueError, match="missing required columns"):
        compute(df)
