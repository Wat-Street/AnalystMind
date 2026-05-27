# Technical indicator computation module.
# Source: no external API — computes derived signals from OHLCV data fetched by ohlcv.py.
# Computes: rsi_signal, macd_signal, breakout_score, volume_surge.
# Used by: technical_analyst persona.

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pandas_ta as ta

RSI_LENGTH = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BREAKOUT_WINDOW = 20
VOLUME_WINDOW = 20
VOLUME_SATURATION = 3.0
MACD_SCALE = 50.0
MIN_ROWS = 60

_REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")


def _rsi_signal(df: pd.DataFrame) -> float:
    rsi = ta.rsi(df["close"], length=RSI_LENGTH)
    return float(rsi.iloc[-1] / 100.0)


def _macd_signal(df: pd.DataFrame) -> float:
    # Use the MACD line (fast EMA - slow EMA) for trend direction over a 3M horizon,
    # not the histogram (which captures acceleration and flips sign on decelerating trends).
    macd = ta.macd(df["close"], fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL)
    line_col = f"MACD_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}"
    line = float(macd[line_col].iloc[-1])
    close = float(df["close"].iloc[-1])
    return 0.5 + 0.5 * math.tanh(line / close * MACD_SCALE)


def _breakout_score(df: pd.DataFrame) -> float:
    window = df["close"].iloc[-BREAKOUT_WINDOW:]
    high = float(window.max())
    low = float(window.min())
    close = float(df["close"].iloc[-1])
    if high == low:
        return 0.5
    return float(np.clip((close - low) / (high - low), 0.0, 1.0))


def _volume_surge(df: pd.DataFrame) -> float:
    avg = float(df["volume"].iloc[-VOLUME_WINDOW:].mean())
    if avg <= 0:
        return 0.0
    ratio = float(df["volume"].iloc[-1]) / avg
    return float(np.clip(ratio / VOLUME_SATURATION, 0.0, 1.0))


def compute(df: pd.DataFrame) -> dict[str, float]:
    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"OHLCV dataframe missing required columns: {missing}")
    if len(df) < MIN_ROWS:
        raise ValueError(f"need >= {MIN_ROWS} rows for technical indicators, got {len(df)}")

    return {
        "rsi_signal": _rsi_signal(df),
        "macd_signal": _macd_signal(df),
        "breakout_score": _breakout_score(df),
        "volume_surge": _volume_surge(df),
    }
