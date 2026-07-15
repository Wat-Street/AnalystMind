# Forward return label computation for FT-Transformer training.
# Source: no external API — derived from the ohlcv table (ingested by ohlcv.py).
# Computes: fwd_return_{N}d for every N in features.HORIZONS.
# Used by: dataset.py (training labels), evaluate.py (metrics per horizon).

# Kiana

"""Forward return labels.

For each (ticker, trading date) the label at horizon ``N`` is the simple
forward return on the *adjusted* close (split/dividend safe):

    fwd_return_Nd[t] = (adj_close[t + N] - adj_close[t]) / adj_close[t]

``t + N`` counts *trading days* on the ticker's own price history (row
offsets), not calendar days. Column names are derived from
``features.HORIZONS`` so adding/removing a horizon there automatically
propagates here — the stub's fixed ``return_1m..return_12m`` naming predates
the 5-horizon config and does not scale with it.

Rows whose future price does not exist yet (the tail of history) keep NaN
labels in :func:`compute_forward_returns` (pure, no dropping) and are dropped
in :func:`build_labels` according to ``require_all_horizons``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from .features import HORIZONS

MARKET_TZ = "America/New_York"

_REQUIRED_COLUMNS = ("ticker", "date", "adj_close")


def label_column(horizon: int) -> str:
    """Canonical label column name for a horizon in trading days."""
    return f"fwd_return_{int(horizon)}d"


# Shared contract with dataset.py / train.py / evaluate.py — one column per
# horizon, in HORIZONS order.
LABEL_COLUMNS: list[str] = [label_column(h) for h in HORIZONS]
HORIZON_LABELS: dict[int, str] = {h: label_column(h) for h in HORIZONS}


# ── Pure helpers (no I/O — fully unit-tested) ────────────────────────────────

def normalize_trading_date(timestamps: pd.Series) -> pd.Series:
    """Collapse bar timestamps to naive market-calendar dates.

    tz-aware timestamps are interpreted in America/New_York before the time
    component is dropped; naive timestamps are assumed already market-local.
    """
    try:
        ts = pd.to_datetime(timestamps)
    except (ValueError, TypeError):
        # Mixed UTC offsets (e.g. EST/EDT across DST) need explicit utc=True.
        ts = pd.to_datetime(timestamps, utc=True)
    if getattr(ts.dtype, "tz", None) is not None:
        ts = ts.dt.tz_convert(MARKET_TZ).dt.tz_localize(None)
    return ts.dt.normalize()


def compute_forward_returns(
    prices: pd.DataFrame,
    horizons: list[int] | None = None,
) -> pd.DataFrame:
    """Compute forward returns per (ticker, date) at every horizon.

    Parameters
    ----------
    prices : pd.DataFrame
        Columns ``ticker, date, adj_close``. Need not be sorted or
        deduplicated — duplicates on (ticker, date) keep the last row.
    horizons : list[int] | None
        Trading-day horizons. Defaults to ``features.HORIZONS``.

    Returns
    -------
    pd.DataFrame
        Indexed by (ticker, date), one ``fwd_return_{N}d`` column per horizon.
        NaN where the future price is unavailable (end of history) or where
        the base/future adjusted close is missing or non-positive. No rows
        are dropped here — this function is pure; dropping is a policy
        decision made in :func:`build_labels`.
    """
    missing = [c for c in _REQUIRED_COLUMNS if c not in prices.columns]
    if missing:
        raise ValueError(f"prices dataframe missing required columns: {missing}")

    horizons = list(HORIZONS if horizons is None else horizons)

    df = prices.loc[:, list(_REQUIRED_COLUMNS)].copy()
    df["adj_close"] = pd.to_numeric(df["adj_close"], errors="coerce")
    # Non-positive prices are data errors — poison both the base return and
    # any label whose future lands on them, rather than emitting inf/garbage.
    df.loc[df["adj_close"] <= 0, "adj_close"] = np.nan

    df = (
        df.sort_values(["ticker", "date"], kind="stable")
          .drop_duplicates(["ticker", "date"], keep="last")
          .reset_index(drop=True)
    )

    grouped = df.groupby("ticker", sort=False)["adj_close"]
    out = df[["ticker", "date"]].copy()
    for horizon in horizons:
        future = grouped.shift(-horizon)
        out[label_column(horizon)] = (future - df["adj_close"]) / df["adj_close"]

    return out.set_index(["ticker", "date"]).sort_index()


# ── Public API ───────────────────────────────────────────────────────────────

def load_price_history(session: Session) -> pd.DataFrame:
    """Load (ticker, date, adj_close) for all tickers from the ohlcv table."""
    from ..models.db import OHLCV

    rows = session.execute(select(OHLCV.ticker, OHLCV.time, OHLCV.adj_close)).all()
    if not rows:
        return pd.DataFrame(columns=["ticker", "date", "adj_close"])

    df = pd.DataFrame(rows, columns=["ticker", "time", "adj_close"])
    df["date"] = normalize_trading_date(df["time"])
    return (
        df.sort_values(["ticker", "time"], kind="stable")
          .drop_duplicates(["ticker", "date"], keep="last")
          .loc[:, ["ticker", "date", "adj_close"]]
          .reset_index(drop=True)
    )


def build_labels(session: Session, require_all_horizons: bool = False) -> pd.DataFrame:
    """Load prices from the DB and return the forward-return label frame.

    Parameters
    ----------
    require_all_horizons : bool
        If True, keep only rows where *every* horizon label exists — clean
        tensors, but drops the trailing ``max(HORIZONS)`` days of history.
        If False (default), drop only rows with *no* label at all; longer
        horizons may then be NaN and the training loss must mask them.
    """
    labels = compute_forward_returns(load_price_history(session))
    return labels.dropna(how="any" if require_all_horizons else "all")
