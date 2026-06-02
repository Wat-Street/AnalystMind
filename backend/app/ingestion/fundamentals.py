"""Fundamentals ingestion pipeline.

Source: yfinance (free, no API key).
Fetches TTM (trailing twelve months) financial metrics and writes a single row
per ticker to the ``fundamentals`` table.

Metrics computed:
  - ``fcf_yield``        — Free Cash Flow ÷ Market Cap
  - ``trailing_pe``      — Price ÷ last 12 months of earnings
  - ``forward_pe``       — Price ÷ forecasted next-12-month earnings
  - ``ev_ebitda``        — Enterprise Value ÷ EBITDA
  - ``revenue_cagr_3y``  — 3-year compound annual growth rate of revenue
  - ``gross_margin``     — (Revenue − COGS) ÷ Revenue
  - ``net_debt``         — Total Debt − Cash

Consumed by:
  - ``value_fundamentalist`` persona  (fcf_yield 40%, trailing_pe 30%)
  - ``growth_visionary`` persona      (forward_pe 45%, revenue_cagr_3y 20%)
  - ``dividend_income`` persona       (fcf_yield for FCF coverage)
  - ``insider_institutional`` persona  (net_debt, ev_ebitda)
  - ``macro_topdown`` persona         (ev_ebitda for sector valuation)
"""
from __future__ import annotations

import math
import sys

import numpy as np
import pandas as pd
import yfinance as yf
from sqlalchemy.orm import Session

# ── Constants ────────────────────────────────────────────────────────────────
PERIOD_LABEL = "TTM"            # trailing twelve months — one row per ticker
CAGR_LOOKBACK_YEARS = 3        # years of revenue history needed for CAGR
_REQUIRED_REVENUE_POINTS = CAGR_LOOKBACK_YEARS + 1   # 4 data points needed

_OUTPUT_COLUMNS = (
    "fcf_yield", "trailing_pe", "forward_pe", "ev_ebitda",
    "revenue_cagr_3y", "gross_margin", "net_debt",
)

# yfinance ``info`` dict keys we read (for documentation / validation).
_INFO_KEYS = (
    "trailingPE", "forwardPE", "enterpriseToEbitda",
    "marketCap", "freeCashflow", "grossMargins",
    "totalDebt", "totalCash",
)


# ── Pure helpers (no I/O — fully unit-tested) ────────────────────────────────

def _positive_or_none(x) -> float | None:
    """Return *x* as a positive float, or ``None`` if missing / non-positive / NaN."""
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(v) or math.isinf(v) or v <= 0:
        return None
    return v


def _normalize_margin(x) -> float | None:
    """Return *x* as a float in [0, 1], or ``None`` if missing / out of range."""
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(v) or math.isinf(v) or v < 0 or v > 1:
        return None
    return v


def _compute_fcf_yield(
    free_cash_flow: float | None,
    market_cap: float | None,
) -> float | None:
    """Free Cash Flow ÷ Market Cap.

    Returns ``None`` if either input is missing or ``market_cap <= 0``.
    """
    if free_cash_flow is None or market_cap is None:
        return None
    try:
        fcf = float(free_cash_flow)
        mcap = float(market_cap)
    except (TypeError, ValueError):
        return None
    if math.isnan(fcf) or math.isnan(mcap) or mcap <= 0:
        return None
    return fcf / mcap


def _compute_net_debt(
    total_debt: float | None,
    total_cash: float | None,
) -> float | None:
    """Total Debt − Cash.  Treats a missing side as 0.

    Returns ``None`` only if both inputs are ``None``.
    """
    if total_debt is None and total_cash is None:
        return None
    debt = float(total_debt) if total_debt is not None else 0.0
    cash = float(total_cash) if total_cash is not None else 0.0
    return debt - cash


def _compute_revenue_cagr_3y(revenue_series: list[float | None]) -> float | None:
    """3-year compound annual growth rate of revenue.

    Parameters
    ----------
    revenue_series : list[float | None]
        Annual "Total Revenue" values, **oldest → newest**.
        Must contain at least 4 non-None positive values spanning 3 years.

    Returns
    -------
    float or None
        CAGR as a decimal (e.g. 0.12 = 12%), or ``None`` if the data is
        insufficient or the inputs are non-positive.
    """
    # Filter to only the positive, non-None values, keeping their original positions.
    indexed = [(i, v) for i, v in enumerate(revenue_series)
               if v is not None and not math.isnan(v) and v > 0]

    if len(indexed) < _REQUIRED_REVENUE_POINTS:
        return None

    # Use the first and last valid values.
    i_first, first = indexed[0]
    i_last, last = indexed[-1]

    n_years = i_last - i_first  # positional distance in years
    if n_years < CAGR_LOOKBACK_YEARS:
        return None

    if first <= 0 or last <= 0:
        return None

    return (last / first) ** (1.0 / n_years) - 1.0


def _annual_revenue_series(ticker_obj: yf.Ticker) -> list[float | None]:
    """Extract annual "Total Revenue" from the ticker's ``income_stmt``, oldest → newest.

    Returns an empty list if no revenue row is found.
    """
    try:
        stmt = ticker_obj.income_stmt
    except Exception:
        return []

    if stmt is None or stmt.empty:
        return []

    # yfinance uses "Total Revenue" as the row label for most companies.
    # Some companies (e.g. banks) use "Operating Revenue" instead.
    candidates = ["Total Revenue", "Operating Revenue", "Revenue"]
    label = None
    for candidate in candidates:
        matches = [idx for idx in stmt.index if idx.lower() == candidate.lower()]
        if matches:
            label = matches[0]
            break

    if label is None:
        return []

    row = stmt.loc[label]
    # Columns are fiscal period end dates (Timestamps), newest first.
    # Convert to list, oldest → newest.
    values = [row.iloc[i] for i in range(len(row) - 1, -1, -1)]
    return [float(v) if pd.notna(v) else None for v in values]


def _extract_metrics(
    info: dict,
    revenue_series: list[float | None],
) -> dict[str, float | None]:
    """Map a yfinance ``info`` dict + revenue history to our schema columns.

    This is a pure function — no I/O, no side effects.
    """
    return {
        "fcf_yield":       _compute_fcf_yield(info.get("freeCashflow"), info.get("marketCap")),
        "trailing_pe":     _positive_or_none(info.get("trailingPE")),
        "forward_pe":      _positive_or_none(info.get("forwardPE")),
        "ev_ebitda":       _positive_or_none(info.get("enterpriseToEbitda")),
        "revenue_cagr_3y": _compute_revenue_cagr_3y(revenue_series),
        "gross_margin":    _normalize_margin(info.get("grossMargins")),
        "net_debt":        _compute_net_debt(info.get("totalDebt"), info.get("totalCash")),
    }


def _to_record(df: pd.DataFrame, ticker: str) -> dict:
    """Convert the single-row fundamentals DataFrame into a dict for DB insert."""
    row = df.iloc[0]
    return {
        "ticker":          ticker,
        "period":          PERIOD_LABEL,
        "fcf_yield":       float(row["fcf_yield"]) if pd.notna(row["fcf_yield"]) else None,
        "trailing_pe":     float(row["trailing_pe"]) if pd.notna(row["trailing_pe"]) else None,
        "forward_pe":      float(row["forward_pe"]) if pd.notna(row["forward_pe"]) else None,
        "ev_ebitda":       float(row["ev_ebitda"]) if pd.notna(row["ev_ebitda"]) else None,
        "revenue_cagr_3y": float(row["revenue_cagr_3y"]) if pd.notna(row["revenue_cagr_3y"]) else None,
        "gross_margin":    float(row["gross_margin"]) if pd.notna(row["gross_margin"]) else None,
        "net_debt":        float(row["net_debt"]) if pd.notna(row["net_debt"]) else None,
    }


# ── Public API ───────────────────────────────────────────────────────────────

def fetch_fundamentals(ticker: str) -> pd.DataFrame:
    """Fetch TTM fundamentals for *ticker* from yfinance.

    Returns a single-row DataFrame indexed by ``(ticker, "TTM")`` with columns:
    ``fcf_yield, trailing_pe, forward_pe, ev_ebitda, revenue_cagr_3y,
    gross_margin, net_debt``.

    Any metric may be ``NaN`` if yfinance does not provide the underlying data.

    Raises
    ------
    ValueError
        If *ticker* is empty.
    """
    if not ticker or not ticker.strip():
        raise ValueError("ticker is required")

    ticker = ticker.strip().upper()
    t = yf.Ticker(ticker)

    info = t.info
    revenue_series = _annual_revenue_series(t)
    metrics = _extract_metrics(info, revenue_series)

    df = pd.DataFrame([metrics], columns=list(_OUTPUT_COLUMNS))
    df.index = pd.MultiIndex.from_tuples(
        [(ticker, PERIOD_LABEL)],
        names=["ticker", "period"],
    )
    return df


def upsert_fundamentals(session: Session, df: pd.DataFrame) -> int:
    """Upsert rows from *df* into the ``fundamentals`` table keyed on ``(ticker, period)``.

    Commits the session.  Returns the number of rows upserted.
    """
    if df.empty:
        return 0

    from sqlalchemy.dialects.postgresql import insert as pg_insert
    from app.models.db import Fundamentals

    ticker = df.index.get_level_values("ticker")[0]
    record = _to_record(df, ticker)

    stmt = pg_insert(Fundamentals).values([record])
    stmt = stmt.on_conflict_do_update(
        index_elements=["ticker", "period"],
        set_={
            "fcf_yield":       stmt.excluded.fcf_yield,
            "trailing_pe":     stmt.excluded.trailing_pe,
            "forward_pe":      stmt.excluded.forward_pe,
            "ev_ebitda":       stmt.excluded.ev_ebitda,
            "revenue_cagr_3y": stmt.excluded.revenue_cagr_3y,
            "gross_margin":    stmt.excluded.gross_margin,
            "net_debt":        stmt.excluded.net_debt,
            "fetched_at":      stmt.excluded.fetched_at,
        },
    )
    session.execute(stmt)
    session.commit()
    return 1


# ── Smoke test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    t = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    df = fetch_fundamentals(t)
    print(f"Ticker: {t}")
    print(df.to_string())
