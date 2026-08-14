"""Macro series ingestion pipeline.

Source: FRED (Federal Reserve Economic Data) REST API. Requires a free API key
(https://fred.stlouisfed.org/docs/api/api_key.html) passed as *api_key* or read
from the ``FRED_API_KEY`` env var.

Series fetched, matching ``app.ml.dataset.MACRO_SERIES_TO_FEATURE``:
  - ``FEDFUNDS``  -> fed_funds_rate
  - ``UNRATE``    -> unemployment
  - ``DGS10``     -> treasury_10y
  - ``T10Y2Y``    -> yield_spread
  - ``CPI_YOY``   -> cpi_yoy (derived: CPIAUCSL year-over-year % change; FRED has
                     no single series id for this, so it is computed here and
                     stored under the synthetic id ``CPI_YOY``)

Rows are ticker-independent (macro_series has no ticker column) — fetch once,
not per ticker.
"""
from __future__ import annotations

import os
import sys

import pandas as pd
import requests
from sqlalchemy.orm import Session

FRED_OBSERVATIONS_URL = "https://api.stlouisfed.org/fred/series/observations"
DEFAULT_START = "1990-01-01"

DIRECT_SERIES_IDS = ("FEDFUNDS", "UNRATE", "DGS10", "T10Y2Y")
CPI_SOURCE_SERIES_ID = "CPIAUCSL"
CPI_DERIVED_SERIES_ID = "CPI_YOY"


def fetch_series(series_id: str, api_key: str, start: str = DEFAULT_START) -> pd.DataFrame:
    """Fetch one FRED series and return an ``observation_date, value`` DataFrame.

    Drops FRED's ``"."`` missing-value sentinel rows.
    """
    resp = requests.get(
        FRED_OBSERVATIONS_URL,
        params={
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
            "observation_start": start,
        },
        timeout=30,
    )
    resp.raise_for_status()
    observations = resp.json()["observations"]

    df = pd.DataFrame(observations)[["date", "value"]]
    df = df[df["value"] != "."]
    df["value"] = df["value"].astype(float)
    df = df.rename(columns={"date": "observation_date"})
    df["observation_date"] = pd.to_datetime(df["observation_date"]).dt.date
    return df[["observation_date", "value"]]


def fetch_all_macro(api_key: str | None = None, start: str = DEFAULT_START) -> pd.DataFrame:
    """Fetch every macro series the model consumes into one long DataFrame.

    Returns columns ``series_id, observation_date, value``.

    Raises
    ------
    RuntimeError
        If no FRED API key is supplied or set via ``FRED_API_KEY``.
    """
    api_key = api_key or os.environ.get("FRED_API_KEY")
    if not api_key:
        raise RuntimeError(
            "FRED_API_KEY not set — get a free key at "
            "https://fred.stlouisfed.org/docs/api/api_key.html and set it as "
            "an env var to ingest macro series."
        )

    frames: list[pd.DataFrame] = []

    for series_id in DIRECT_SERIES_IDS:
        df = fetch_series(series_id, api_key, start=start)
        df.insert(0, "series_id", series_id)
        frames.append(df)

    cpi = fetch_series(CPI_SOURCE_SERIES_ID, api_key, start=start).sort_values("observation_date")
    cpi["value"] = cpi["value"].pct_change(12) * 100
    cpi = cpi.dropna(subset=["value"])
    cpi.insert(0, "series_id", CPI_DERIVED_SERIES_ID)
    frames.append(cpi)

    return pd.concat(frames, ignore_index=True)


def upsert_macro(session: Session, df: pd.DataFrame) -> int:
    """Upsert rows from *df* into the ``macro_series`` table keyed on
    ``(series_id, observation_date)``.

    Commits the session. Returns the number of rows upserted.
    """
    if df.empty:
        return 0

    from sqlalchemy.dialects.postgresql import insert as pg_insert
    from app.models.db import MacroSeries

    records = df[["series_id", "observation_date", "value"]].to_dict("records")

    stmt = pg_insert(MacroSeries).values(records)
    stmt = stmt.on_conflict_do_update(
        index_elements=["series_id", "observation_date"],
        set_={"value": stmt.excluded.value},
    )
    session.execute(stmt)
    session.commit()
    return len(records)


# ── Smoke test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    series = sys.argv[1] if len(sys.argv) > 1 else None
    if series:
        df = fetch_series(series, os.environ["FRED_API_KEY"])
        print(f"Series: {series}  —  {len(df)} rows fetched")
        print(df.tail())
    else:
        df = fetch_all_macro()
        print(f"{len(df)} total rows fetched across {df['series_id'].nunique()} series")
        print(df.groupby("series_id").size())
