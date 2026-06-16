"""Options flow ingestion pipeline.

Source: yfinance (free, no API key).
Fetches a listed option chain for one expiration and writes a daily snapshot to
the ``options_flow`` table.

Computes:
  - ``unusual_options_score`` — normalized option volume intensity vs open interest
  - ``put_call_ratio``        — put volume divided by call volume
  - ``gamma_exposure``        — approximate signed GEX per 1% underlying move

Consumed by:
  - ``options_flow_trader`` persona
"""
from __future__ import annotations

import math
import sys
from datetime import date, datetime, timezone
from typing import Any

import pandas as pd
import yfinance as yf
from sqlalchemy.orm import Session

CONTRACT_MULTIPLIER = 100
RISK_FREE_RATE = 0.05
UNUSUAL_RATIO_SATURATION = 5.0
UNUSUAL_VOLUME_SATURATION = 50_000.0

OPTION_COLUMNS = (
    "contractSymbol",
    "lastTradeDate",
    "strike",
    "lastPrice",
    "bid",
    "ask",
    "volume",
    "openInterest",
    "impliedVolatility",
)

OUTPUT_COLUMNS = (
    "underlying_price",
    "call_volume",
    "put_volume",
    "call_open_interest",
    "put_open_interest",
    "put_call_ratio",
    "unusual_options_score",
    "gamma_exposure",
)


def _today() -> date:
    return datetime.now(timezone.utc).date()


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def _as_non_negative_float(value: Any) -> float:
    out = _as_float(value)
    if out is None or out < 0:
        return 0.0
    return out


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _normalize_option_frame(raw: pd.DataFrame | None) -> pd.DataFrame:
    """Return a numeric, schema-stable option chain side."""
    if raw is None or not isinstance(raw, pd.DataFrame) or raw.empty:
        return pd.DataFrame(columns=list(OPTION_COLUMNS))

    df = raw.copy()
    for column in OPTION_COLUMNS:
        if column not in df.columns:
            df[column] = None

    for column in ("strike", "lastPrice", "bid", "ask", "volume", "openInterest", "impliedVolatility"):
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["volume"] = df["volume"].fillna(0).clip(lower=0)
    df["openInterest"] = df["openInterest"].fillna(0).clip(lower=0)
    return df[list(OPTION_COLUMNS)]


def _sum_int(df: pd.DataFrame, column: str) -> int:
    if df.empty or column not in df.columns:
        return 0
    return int(df[column].fillna(0).clip(lower=0).sum())


def _put_call_ratio(call_volume: int, put_volume: int) -> float | None:
    if call_volume <= 0:
        return None
    return put_volume / call_volume


def _unusual_options_score(calls: pd.DataFrame, puts: pd.DataFrame) -> float:
    """Score option-volume intensity relative to open interest in [0, 1]."""
    chain = pd.concat([calls, puts], ignore_index=True)
    if chain.empty:
        return 0.0

    volume = chain["volume"].fillna(0).clip(lower=0)
    total_volume = float(volume.sum())
    if total_volume <= 0:
        return 0.0

    open_interest = chain["openInterest"].fillna(0).clip(lower=0)
    volume_to_oi = volume / (open_interest + 1.0)
    weighted_intensity = float((volume_to_oi * volume).sum() / total_volume)

    intensity_score = _clip01(weighted_intensity / UNUSUAL_RATIO_SATURATION)
    volume_score = _clip01(total_volume / UNUSUAL_VOLUME_SATURATION)
    return _clip01(0.70 * intensity_score + 0.30 * volume_score)


def _years_to_expiration(expiration_date: date, as_of_date: date) -> float:
    days = max((expiration_date - as_of_date).days, 1)
    return days / 365.0


def _black_scholes_gamma(
    underlying_price: float | None,
    strike: float | None,
    implied_volatility: float | None,
    years_to_expiration: float,
    risk_free_rate: float = RISK_FREE_RATE,
) -> float | None:
    if (
        underlying_price is None
        or strike is None
        or implied_volatility is None
        or underlying_price <= 0
        or strike <= 0
        or implied_volatility <= 0
        or years_to_expiration <= 0
    ):
        return None

    sqrt_t = math.sqrt(years_to_expiration)
    d1 = (
        math.log(underlying_price / strike)
        + (risk_free_rate + 0.5 * implied_volatility**2) * years_to_expiration
    ) / (implied_volatility * sqrt_t)
    normal_pdf = math.exp(-0.5 * d1**2) / math.sqrt(2.0 * math.pi)
    return normal_pdf / (underlying_price * implied_volatility * sqrt_t)


def _gamma_exposure_for_side(
    df: pd.DataFrame,
    underlying_price: float | None,
    expiration_date: date,
    as_of_date: date,
    sign: int,
) -> float | None:
    if underlying_price is None or df.empty:
        return None

    years = _years_to_expiration(expiration_date, as_of_date)
    exposure = 0.0
    used_rows = 0

    for _, row in df.iterrows():
        open_interest = _as_non_negative_float(row.get("openInterest"))
        if open_interest <= 0:
            continue

        gamma = _black_scholes_gamma(
            underlying_price=underlying_price,
            strike=_as_float(row.get("strike")),
            implied_volatility=_as_float(row.get("impliedVolatility")),
            years_to_expiration=years,
        )
        if gamma is None:
            continue

        exposure += sign * open_interest * CONTRACT_MULTIPLIER * gamma * underlying_price**2 * 0.01
        used_rows += 1

    if used_rows == 0:
        return None
    return exposure


def _compute_gamma_exposure(
    calls: pd.DataFrame,
    puts: pd.DataFrame,
    underlying_price: float | None,
    expiration_date: date,
    as_of_date: date,
) -> float | None:
    call_gex = _gamma_exposure_for_side(calls, underlying_price, expiration_date, as_of_date, sign=1)
    put_gex = _gamma_exposure_for_side(puts, underlying_price, expiration_date, as_of_date, sign=-1)

    if call_gex is None and put_gex is None:
        return None
    return (call_gex or 0.0) + (put_gex or 0.0)


def _underlying_price(underlying: dict | None) -> float | None:
    if not underlying:
        return None
    for key in ("regularMarketPrice", "currentPrice", "postMarketPrice", "bid", "ask"):
        value = _as_float(underlying.get(key))
        if value is not None and value > 0:
            return value
    return None


def _parse_expiration(expiration: str | date) -> date:
    if isinstance(expiration, date):
        return expiration
    try:
        return datetime.strptime(expiration, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError("expiration must be YYYY-MM-DD") from exc


def _build_snapshot(
    ticker: str,
    calls: pd.DataFrame | None,
    puts: pd.DataFrame | None,
    underlying_price: float | None,
    expiration_date: str | date,
    as_of_date: date | None = None,
) -> pd.DataFrame:
    ticker = ticker.strip().upper()
    as_of_date = as_of_date or _today()
    expiration_date = _parse_expiration(expiration_date)

    calls_df = _normalize_option_frame(calls)
    puts_df = _normalize_option_frame(puts)

    call_volume = _sum_int(calls_df, "volume")
    put_volume = _sum_int(puts_df, "volume")

    record = {
        "underlying_price": underlying_price,
        "call_volume": call_volume,
        "put_volume": put_volume,
        "call_open_interest": _sum_int(calls_df, "openInterest"),
        "put_open_interest": _sum_int(puts_df, "openInterest"),
        "put_call_ratio": _put_call_ratio(call_volume, put_volume),
        "unusual_options_score": _unusual_options_score(calls_df, puts_df),
        "gamma_exposure": _compute_gamma_exposure(
            calls_df,
            puts_df,
            underlying_price,
            expiration_date,
            as_of_date,
        ),
    }

    df = pd.DataFrame([record], columns=list(OUTPUT_COLUMNS))
    df.index = pd.MultiIndex.from_tuples(
        [(ticker, as_of_date, expiration_date)],
        names=["ticker", "as_of_date", "expiration_date"],
    )
    return df


def _record_value(value: Any) -> float | int | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, int):
        return value
    return float(value)


def _to_record(df: pd.DataFrame) -> dict:
    row = df.iloc[0]
    ticker = df.index.get_level_values("ticker")[0]
    as_of_date = df.index.get_level_values("as_of_date")[0]
    expiration_date = df.index.get_level_values("expiration_date")[0]

    return {
        "ticker": ticker,
        "as_of_date": as_of_date,
        "expiration_date": expiration_date,
        "underlying_price": _record_value(row["underlying_price"]),
        "call_volume": int(row["call_volume"]),
        "put_volume": int(row["put_volume"]),
        "call_open_interest": int(row["call_open_interest"]),
        "put_open_interest": int(row["put_open_interest"]),
        "put_call_ratio": _record_value(row["put_call_ratio"]),
        "unusual_options_score": float(row["unusual_options_score"]),
        "gamma_exposure": _record_value(row["gamma_exposure"]),
    }


def fetch_options_flow(
    ticker: str,
    expiration: str | None = None,
    as_of_date: date | None = None,
) -> pd.DataFrame:
    """Fetch an options-flow snapshot for *ticker* and one expiration."""
    if not ticker or not ticker.strip():
        raise ValueError("ticker is required")

    ticker = ticker.strip().upper()
    t = yf.Ticker(ticker)
    expirations = list(t.options)
    if not expirations:
        raise ValueError(f"no option expirations returned for ticker {ticker!r}")

    expiration = expiration or expirations[0]
    if expiration not in expirations:
        raise ValueError(
            f"expiration {expiration!r} not available for {ticker}; "
            f"available expirations: {', '.join(expirations)}"
        )

    chain = t.option_chain(expiration)
    return _build_snapshot(
        ticker=ticker,
        calls=chain.calls,
        puts=chain.puts,
        underlying_price=_underlying_price(chain.underlying),
        expiration_date=expiration,
        as_of_date=as_of_date,
    )


def upsert_options_flow(session: Session, df: pd.DataFrame) -> int:
    """Upsert one options-flow snapshot keyed on ticker/date/expiration."""
    if df.empty:
        return 0

    from sqlalchemy.dialects.postgresql import insert as pg_insert
    from app.models.db import OptionsFlow

    record = _to_record(df)
    stmt = pg_insert(OptionsFlow).values([record])
    stmt = stmt.on_conflict_do_update(
        index_elements=["ticker", "as_of_date", "expiration_date"],
        set_={
            "underlying_price": stmt.excluded.underlying_price,
            "call_volume": stmt.excluded.call_volume,
            "put_volume": stmt.excluded.put_volume,
            "call_open_interest": stmt.excluded.call_open_interest,
            "put_open_interest": stmt.excluded.put_open_interest,
            "put_call_ratio": stmt.excluded.put_call_ratio,
            "unusual_options_score": stmt.excluded.unusual_options_score,
            "gamma_exposure": stmt.excluded.gamma_exposure,
            "fetched_at": stmt.excluded.fetched_at,
        },
    )
    session.execute(stmt)
    session.commit()
    return 1


if __name__ == "__main__":
    t = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    exp = sys.argv[2] if len(sys.argv) > 2 else None
    df = fetch_options_flow(t, exp)
    print(f"Ticker: {t}")
    print(df.to_string())
