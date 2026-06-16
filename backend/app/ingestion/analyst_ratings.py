"""Analyst ratings and price-target ingestion pipeline.

Source: yfinance (free, no API key).
Fetches current analyst recommendation counts, price target consensus, and
recent upgrade/downgrade activity. Writes one daily snapshot per ticker to the
``analyst_ratings`` table.

Consumed by:
  - ``sentiment_trader`` persona via ``analyst_ratings`` data source
"""
from __future__ import annotations

import math
import sys
from datetime import date, datetime, timedelta, timezone
from typing import Any

import pandas as pd
import yfinance as yf
from sqlalchemy.orm import Session

RECOMMENDATION_COLUMNS = ("strongBuy", "buy", "hold", "sell", "strongSell")
OUTPUT_COLUMNS = (
    "current_price",
    "consensus_pt",
    "target_low",
    "target_high",
    "target_median",
    "price_target_upside",
    "recommendation_score",
    "coverage_volume",
    "recent_rating_delta",
    "strong_buy",
    "buy",
    "hold",
    "sell",
    "strong_sell",
)

COVERAGE_SATURATION = 30.0
RATING_CHANGE_LOOKBACK_DAYS = 90

_RECOMMENDATION_WEIGHTS = {
    "strongBuy": 1.0,
    "buy": 0.75,
    "hold": 0.50,
    "sell": 0.25,
    "strongSell": 0.0,
}


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


def _as_non_negative_int(value: Any) -> int:
    if value is None or pd.isna(value):
        return 0
    try:
        out = int(value)
    except (TypeError, ValueError):
        return 0
    return max(out, 0)


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _normalize_price_targets(raw: dict | pd.Series | None) -> dict[str, float | None]:
    """Normalize yfinance price-target keys into our DB column names."""
    if raw is None:
        data: dict[str, Any] = {}
    elif isinstance(raw, pd.Series):
        data = raw.to_dict()
    else:
        data = dict(raw)

    return {
        "current_price": _as_float(data.get("current")),
        "consensus_pt": _as_float(data.get("mean")),
        "target_low": _as_float(data.get("low")),
        "target_high": _as_float(data.get("high")),
        "target_median": _as_float(data.get("median")),
    }


def _normalize_recommendations(raw: pd.DataFrame | dict | None) -> dict[str, int]:
    """Return the current recommendation-count row from yfinance output."""
    counts = {column: 0 for column in RECOMMENDATION_COLUMNS}

    if raw is None:
        return counts
    if isinstance(raw, dict):
        raw = pd.DataFrame(raw)
    if not isinstance(raw, pd.DataFrame) or raw.empty:
        return counts

    df = raw.copy()
    if "period" in df.columns:
        current = df[df["period"].astype(str).str.lower() == "0m"]
        row = current.iloc[0] if not current.empty else df.iloc[0]
    else:
        row = df.iloc[0]

    for column in RECOMMENDATION_COLUMNS:
        counts[column] = _as_non_negative_int(row.get(column))
    return counts


def _recommendation_score(counts: dict[str, int]) -> float | None:
    total = sum(counts.get(column, 0) for column in RECOMMENDATION_COLUMNS)
    if total <= 0:
        return None

    weighted = sum(
        counts.get(column, 0) * _RECOMMENDATION_WEIGHTS[column]
        for column in RECOMMENDATION_COLUMNS
    )
    return weighted / total


def _coverage_volume(counts: dict[str, int]) -> float:
    total = sum(counts.get(column, 0) for column in RECOMMENDATION_COLUMNS)
    return _clip01(total / COVERAGE_SATURATION)


def _price_target_upside(current_price: float | None, consensus_pt: float | None) -> float | None:
    if current_price is None or consensus_pt is None or current_price <= 0:
        return None
    return (consensus_pt - current_price) / current_price


def _grade_score(grade: Any) -> float | None:
    if grade is None or pd.isna(grade):
        return None
    text = str(grade).strip().lower()
    if not text:
        return None

    if "strong buy" in text:
        return 1.0
    if any(term in text for term in ("buy", "outperform", "overweight", "accumulate")):
        return 0.75
    if any(term in text for term in ("hold", "neutral", "market perform", "equal-weight", "sector perform")):
        return 0.50
    if any(term in text for term in ("sell", "underperform", "underweight", "reduce")):
        return 0.25
    return None


def _recent_rating_delta(
    raw: pd.DataFrame | None,
    as_of_date: date,
    lookback_days: int = RATING_CHANGE_LOOKBACK_DAYS,
) -> float | None:
    """Score recent analyst changes as bearish=0, neutral=0.5, bullish=1."""
    if raw is None or not isinstance(raw, pd.DataFrame) or raw.empty:
        return None

    df = raw.copy()
    # yfinance returns capitalized columns (ToGrade/FromGrade/Action) and indexes
    # rows by GradeDate; older versions used lowercase. Normalize the column names
    # so this reads either schema instead of silently scoring every row as None.
    df.columns = [str(column).lower() for column in df.columns]
    if "date" in df.columns:
        dates = pd.to_datetime(df["date"], errors="coerce").dt.date
    else:
        dates = pd.to_datetime(df.index, errors="coerce").date

    cutoff = as_of_date - timedelta(days=lookback_days)
    df = df.assign(_date=dates)
    df = df[(df["_date"].notna()) & (df["_date"] >= cutoff) & (df["_date"] <= as_of_date)]
    if df.empty:
        return None

    deltas: list[float] = []
    for _, row in df.iterrows():
        action = str(row.get("action", "")).lower()
        if "up" in action:
            deltas.append(1.0)
            continue
        if "down" in action:
            deltas.append(-1.0)
            continue

        to_score = _grade_score(row.get("tograde"))
        from_score = _grade_score(row.get("fromgrade"))
        if to_score is None or from_score is None:
            continue
        diff = to_score - from_score
        if abs(diff) < 0.05:
            deltas.append(0.0)
        else:
            deltas.append(1.0 if diff > 0 else -1.0)

    if not deltas:
        return None

    return _clip01((sum(deltas) / len(deltas) + 1.0) / 2.0)


def _build_snapshot(
    ticker: str,
    price_targets: dict | pd.Series | None,
    recommendations: pd.DataFrame | dict | None,
    rating_changes: pd.DataFrame | None,
    as_of_date: date | None = None,
) -> pd.DataFrame:
    ticker = ticker.strip().upper()
    as_of_date = as_of_date or _today()

    targets = _normalize_price_targets(price_targets)
    counts = _normalize_recommendations(recommendations)

    record = {
        **targets,
        "price_target_upside": _price_target_upside(targets["current_price"], targets["consensus_pt"]),
        "recommendation_score": _recommendation_score(counts),
        "coverage_volume": _coverage_volume(counts),
        "recent_rating_delta": _recent_rating_delta(rating_changes, as_of_date),
        "strong_buy": counts["strongBuy"],
        "buy": counts["buy"],
        "hold": counts["hold"],
        "sell": counts["sell"],
        "strong_sell": counts["strongSell"],
    }

    df = pd.DataFrame([record], columns=list(OUTPUT_COLUMNS))
    df.index = pd.MultiIndex.from_tuples(
        [(ticker, as_of_date)],
        names=["ticker", "as_of_date"],
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

    return {
        "ticker": ticker,
        "as_of_date": as_of_date,
        "current_price": _record_value(row["current_price"]),
        "consensus_pt": _record_value(row["consensus_pt"]),
        "target_low": _record_value(row["target_low"]),
        "target_high": _record_value(row["target_high"]),
        "target_median": _record_value(row["target_median"]),
        "price_target_upside": _record_value(row["price_target_upside"]),
        "recommendation_score": _record_value(row["recommendation_score"]),
        "coverage_volume": _record_value(row["coverage_volume"]),
        "recent_rating_delta": _record_value(row["recent_rating_delta"]),
        "strong_buy": int(row["strong_buy"]),
        "buy": int(row["buy"]),
        "hold": int(row["hold"]),
        "sell": int(row["sell"]),
        "strong_sell": int(row["strong_sell"]),
    }


def fetch_analyst_ratings(ticker: str, as_of_date: date | None = None) -> pd.DataFrame:
    """Fetch analyst ratings and price targets for *ticker* from yfinance."""
    if not ticker or not ticker.strip():
        raise ValueError("ticker is required")

    ticker = ticker.strip().upper()
    t = yf.Ticker(ticker)
    return _build_snapshot(
        ticker=ticker,
        price_targets=t.get_analyst_price_targets(),
        recommendations=t.get_recommendations(),
        rating_changes=t.get_upgrades_downgrades(),
        as_of_date=as_of_date,
    )


def upsert_analyst_ratings(session: Session, df: pd.DataFrame) -> int:
    """Upsert one analyst-ratings snapshot keyed on ``(ticker, as_of_date)``."""
    if df.empty:
        return 0

    from sqlalchemy.dialects.postgresql import insert as pg_insert
    from app.models.db import AnalystRating

    record = _to_record(df)
    stmt = pg_insert(AnalystRating).values([record])
    stmt = stmt.on_conflict_do_update(
        index_elements=["ticker", "as_of_date"],
        set_={
            "current_price": stmt.excluded.current_price,
            "consensus_pt": stmt.excluded.consensus_pt,
            "target_low": stmt.excluded.target_low,
            "target_high": stmt.excluded.target_high,
            "target_median": stmt.excluded.target_median,
            "price_target_upside": stmt.excluded.price_target_upside,
            "recommendation_score": stmt.excluded.recommendation_score,
            "coverage_volume": stmt.excluded.coverage_volume,
            "recent_rating_delta": stmt.excluded.recent_rating_delta,
            "strong_buy": stmt.excluded.strong_buy,
            "buy": stmt.excluded.buy,
            "hold": stmt.excluded.hold,
            "sell": stmt.excluded.sell,
            "strong_sell": stmt.excluded.strong_sell,
            "fetched_at": stmt.excluded.fetched_at,
        },
    )
    session.execute(stmt)
    session.commit()
    return 1


if __name__ == "__main__":
    t = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    df = fetch_analyst_ratings(t)
    print(f"Ticker: {t}")
    print(df.to_string())
