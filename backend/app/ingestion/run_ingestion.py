"""End-to-end ingestion runner.

Populates ohlcv, fundamentals, transcripts, and macro_series for the Week-1
ticker set by calling the existing per-source fetch_*/upsert_* functions
against a live DATABASE_URL. Each source is independently try/excepted and
logged, so one bad ticker or an unavailable dependency (e.g. defeatbeta-api
outside Linux, or a missing FRED_API_KEY) doesn't abort the whole run.

Usage (from backend/):
    python -m app.ingestion.run_ingestion [TICKER ...]

With no arguments, ingests the fixed Week-1 set: AAPL, MSFT, NVDA, TSLA, AMZN.
"""
from __future__ import annotations

import logging
import sys

from sqlalchemy.orm import Session

from . import fundamentals, macro, ohlcv, transcripts

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

WEEK1_TICKERS = ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN"]

# ohlcv.fetch_history defaults to period="6mo" — nowhere near enough history
# for dataset.py's train <=2022 / val 2023 / test >=2024 split.
OHLCV_PERIOD = "max"


def _run_step(name: str, ticker: str, fn) -> int:
    try:
        rows = fn()
        logger.info("%-12s %-6s ok — %d rows", name, ticker, rows)
        return rows
    except Exception as exc:  # noqa: BLE001 - best-effort per source, log and move on
        logger.warning("%-12s %-6s skipped — %s: %s", name, ticker, type(exc).__name__, exc)
        return 0


def ingest_ticker(session: Session, ticker: str) -> dict[str, int]:
    totals = {}

    totals["ohlcv"] = _run_step(
        "ohlcv", ticker,
        lambda: ohlcv.upsert_history(session, ticker, ohlcv.fetch_history(ticker, period=OHLCV_PERIOD)),
    )
    totals["fundamentals"] = _run_step(
        "fundamentals", ticker,
        lambda: fundamentals.upsert_fundamentals(session, fundamentals.fetch_fundamentals(ticker)),
    )
    totals["transcripts"] = _run_step(
        "transcripts", ticker,
        lambda: transcripts.upsert_transcripts(session, transcripts.ingest(ticker)),
    )

    return totals


def ingest_macro(session: Session) -> int:
    return _run_step("macro", "-", lambda: macro.upsert_macro(session, macro.fetch_all_macro()))


def run(tickers: list[str]) -> None:
    from app.models.db import engine

    grand_totals = {"ohlcv": 0, "fundamentals": 0, "transcripts": 0, "macro": 0}

    with Session(engine) as session:
        for ticker in tickers:
            per_ticker = ingest_ticker(session, ticker)
            for source, rows in per_ticker.items():
                grand_totals[source] += rows

        grand_totals["macro"] += ingest_macro(session)

    logger.info("=" * 60)
    logger.info(
        "totals — ohlcv=%d fundamentals=%d transcripts=%d macro=%d",
        grand_totals["ohlcv"], grand_totals["fundamentals"],
        grand_totals["transcripts"], grand_totals["macro"],
    )


if __name__ == "__main__":
    run(sys.argv[1:] or WEEK1_TICKERS)
