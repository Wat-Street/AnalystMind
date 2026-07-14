# ML dataset assembly: DB tables → per-(ticker, date) snapshots → torch Dataset.
# Source: no external API — joins the ohlcv, fundamentals, sentiment_scores
# (via transcripts) and macro_series tables ingested by the ingestion modules.
# Used by: train.py (training data), inference.py (feature_medians for serving).

# Kiana

"""Training dataset for the FT-Transformer.

Pipeline
--------
1. Load daily bars from ``ohlcv`` and derive the price-based features that
   ``features.FEATURE_ORDER`` documents as "derived in dataset.py":
   momentum, market-relative strength/beta (vs ``BENCHMARK_TICKER``),
   realized volatility, max drawdown, dollar volume, and the four technical
   signals (vectorised over history with the same formulas and constants as
   ``ingestion.technical_indicators``, which only scores the latest bar).
2. As-of join ``fundamentals`` (point-in-time on ``fetched_at``),
   transcript sentiment (quarter end + availability lag), and ``macro_series``
   (last observation on or before each trading date, plus derived trends).
3. Feature groups whose ingestors have no historical store yet (options,
   analyst ratings, insider history, ebitda for net_debt_to_ebitda) surface
   as all-NaN columns → imputed with a missing indicator, never a crash.
4. Time split: train ≤ 2022, val = 2023, test ≥ 2024.
5. Impute NaNs with **train-split** column medians (reused for val/test and
   exposed on the bundle for inference — no leakage from future splits).
   All-NaN columns fall back to 0.0 with indicator 1.
6. Wrap each split in :class:`AnalystMindDataset` yielding
   ``(feature_tensor, label_tensor)``.

Contracts
---------
- Feature tensor column order is exactly ``features.FEATURE_ORDER`` — the
  contract shared with ``tokenizer.FeatureTokenizer`` (agreed with Ken).
  Missing-value indicators therefore live in a separate
  ``AnalystMindDataset.missing_mask`` tensor, not extra feature columns,
  so the tensor stays ``N_FEATURES`` wide.
- Label column order is ``labels.LABEL_COLUMNS`` (one per ``HORIZONS`` entry).

Known caveats (documented, revisit as ingestion matures)
--------------------------------------------------------
- fundamentals stores a single TTM snapshot per ticker: dates after
  ``fetched_at`` get a true point-in-time as-of join, dates before it are
  backfilled from the earliest snapshot (mild lookahead that disappears as
  snapshot history accumulates).
- transcripts do not persist ``report_date``, so a transcript is dated
  quarter end + ``TRANSCRIPT_AVAILABILITY_LAG_DAYS``; fiscal-vs-calendar
  quarter drift (e.g. AAPL) is a known approximation.
- macro series ids: coordinate with Kunal (macro.py) — both FRED ids in
  ``MACRO_SERIES_TO_FEATURE`` and the feature names themselves are accepted.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import pandas_ta_classic as ta
import torch
from sqlalchemy import select
from sqlalchemy.orm import Session
from torch.utils.data import Dataset

from ..ingestion.technical_indicators import (
    BREAKOUT_WINDOW,
    MACD_FAST,
    MACD_SCALE,
    MACD_SIGNAL,
    MACD_SLOW,
    RSI_LENGTH,
    VOLUME_SATURATION,
    VOLUME_WINDOW,
)
from .features import FEATURE_ORDER, HORIZONS
from .labels import LABEL_COLUMNS, build_labels, normalize_trading_date

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

BENCHMARK_TICKER = "SPY"            # market reference — needs SPY rows in ohlcv
TRADING_DAYS_PER_YEAR = 252

MOMENTUM_WINDOWS = (21, 63, 126)    # return_{w}d features
RELATIVE_STRENGTH_WINDOW = 63       # market_relative_strength_63d
VOLATILITY_WINDOW = 21              # realized_volatility_21d
BETA_WINDOW = 126                   # beta_126d
DRAWDOWN_WINDOW = 63                # max_drawdown_63d
DOLLAR_VOLUME_WINDOW = 20           # dollar_volume_20d_log

MACRO_CHANGE_WINDOW = 63            # trading days ≈ 3 months, for *_3m features
TRANSCRIPT_AVAILABILITY_LAG_DAYS = 45  # earnings call ≈ 2-6 weeks after quarter end

# Time-based split boundaries (train ≤ 2022, val = 2023, test ≥ 2024).
TRAIN_END = pd.Timestamp("2022-12-31")
VAL_END = pd.Timestamp("2023-12-31")
SPLIT_NAMES = ("train", "val", "test")

SENTIMENT_LABEL_SIGN = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}

# FRED series id → feature name. The feature names themselves are also
# accepted as series ids, so macro.py may store either. cpi_yoy has no single
# FRED id (computed from CPIAUCSL) — store it under "cpi_yoy" or "CPI_YOY".
MACRO_SERIES_TO_FEATURE = {
    "FEDFUNDS": "fed_funds_rate",
    "UNRATE": "unemployment",
    "DGS10": "treasury_10y",
    "T10Y2Y": "yield_spread",
    "CPI_YOY": "cpi_yoy",
}
_MACRO_BASE_FEATURES = ("fed_funds_rate", "cpi_yoy", "unemployment", "treasury_10y", "yield_spread")
_MACRO_DERIVED_FEATURES = ("fed_funds_change_3m", "cpi_trend_3m", "unemployment_change_3m", "real_treasury_10y")

_FUNDAMENTAL_COLUMNS = ("fcf_yield", "trailing_pe", "forward_pe", "ev_ebitda", "revenue_cagr_3y", "gross_margin")
_SENTIMENT_COLUMNS = ("mgmt_sentiment_score", "qa_sentiment_score", "mgmt_qa_sentiment_gap", "transcript_sentiment_change_qoq")


# ── DB loaders (thin I/O — models imported lazily, DATABASE_URL not needed
#    to import this module) ───────────────────────────────────────────────────

def _load_ohlcv(session: Session) -> pd.DataFrame:
    from ..models.db import OHLCV

    rows = session.execute(
        select(OHLCV.ticker, OHLCV.time, OHLCV.close, OHLCV.volume, OHLCV.adj_close)
    ).all()
    if not rows:
        return pd.DataFrame(columns=["ticker", "date", "close", "volume", "adj_close"])

    df = pd.DataFrame(rows, columns=["ticker", "time", "close", "volume", "adj_close"])
    df["date"] = normalize_trading_date(df["time"])
    return (
        df.sort_values(["ticker", "time"], kind="stable")
          .drop_duplicates(["ticker", "date"], keep="last")
          .loc[:, ["ticker", "date", "close", "volume", "adj_close"]]
          .reset_index(drop=True)
    )


def _load_fundamentals(session: Session) -> pd.DataFrame:
    from ..models.db import Fundamentals

    rows = session.execute(
        select(Fundamentals.ticker, Fundamentals.fetched_at,
               *[getattr(Fundamentals, c) for c in _FUNDAMENTAL_COLUMNS])
    ).all()
    return pd.DataFrame(rows, columns=["ticker", "fetched_at", *_FUNDAMENTAL_COLUMNS])


def _load_sentiment(session: Session) -> pd.DataFrame:
    from ..models.db import SentimentScore, Transcript

    rows = session.execute(
        select(SentimentScore.ticker, Transcript.quarter,
               SentimentScore.segment_role, SentimentScore.label, SentimentScore.score)
        .join(Transcript, SentimentScore.transcript_id == Transcript.id)
    ).all()
    return pd.DataFrame(rows, columns=["ticker", "quarter", "segment_role", "label", "score"])


def _load_macro(session: Session) -> pd.DataFrame:
    from ..models.db import MacroSeries

    rows = session.execute(
        select(MacroSeries.series_id, MacroSeries.observation_date, MacroSeries.value)
    ).all()
    return pd.DataFrame(rows, columns=["series_id", "observation_date", "value"])


# ── Price-derived features (pure — fully unit-tested) ────────────────────────

def _max_drawdown(window: np.ndarray) -> float:
    """Max peak-to-trough decline within a price window (≤ 0)."""
    peaks = np.maximum.accumulate(window)
    return float(np.min(window / peaks - 1.0))


def _market_frame(benchmark_bars: pd.DataFrame) -> pd.DataFrame:
    """Per-date benchmark returns used for relative strength and beta."""
    g = benchmark_bars.sort_values("date", kind="stable")
    adj = pd.to_numeric(g["adj_close"], errors="coerce")
    adj = adj.where(adj > 0)
    out = pd.DataFrame(
        {
            "log_return_1d": np.log(adj).diff().to_numpy(),
            "return_63d": adj.pct_change(RELATIVE_STRENGTH_WINDOW, fill_method=None).to_numpy(),
        },
        index=pd.Index(g["date"].to_numpy(), name="date"),
    )
    return out[~out.index.duplicated(keep="last")]


def _single_ticker_price_features(bars: pd.DataFrame, market: pd.DataFrame | None) -> pd.DataFrame:
    """All ohlcv-derived features for one ticker's history, sorted by date.

    Warmup rows (insufficient window) stay NaN and get imputed downstream —
    they are not silently filled with defaults.
    """
    g = bars.sort_values("date", kind="stable")
    close = pd.to_numeric(g["close"], errors="coerce")
    volume = pd.to_numeric(g["volume"], errors="coerce")
    adj = pd.to_numeric(g["adj_close"], errors="coerce")
    adj = adj.where(adj > 0)
    log_ret = np.log(adj).diff()

    out = pd.DataFrame(index=g.index)
    out["ticker"] = g["ticker"]
    out["date"] = g["date"]

    # Technical signals — same formulas/constants as technical_indicators.py,
    # vectorised over the full history instead of scoring only the last bar.
    rsi = ta.rsi(close, length=RSI_LENGTH)
    out["rsi_signal"] = rsi / 100.0 if rsi is not None else np.nan

    macd = ta.macd(close, fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL)
    line_col = f"MACD_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}"
    if macd is not None and line_col in macd:
        out["macd_signal"] = 0.5 + 0.5 * np.tanh(macd[line_col] / close * MACD_SCALE)
    else:
        out["macd_signal"] = np.nan

    window_low = close.rolling(BREAKOUT_WINDOW).min()
    window_span = close.rolling(BREAKOUT_WINDOW).max() - window_low
    out["breakout_score"] = ((close - window_low) / window_span).clip(0.0, 1.0).mask(window_span == 0, 0.5)

    volume_avg = volume.rolling(VOLUME_WINDOW).mean()
    out["volume_surge"] = ((volume / volume_avg) / VOLUME_SATURATION).mask(volume_avg == 0, 0.0).clip(0.0, 1.0)

    # Momentum on adjusted close (split/dividend safe — features.py sketches
    # these on raw close, but that breaks across split days and would be
    # inconsistent with the adj_close-based labels).
    for w in MOMENTUM_WINDOWS:
        out[f"return_{w}d"] = adj.pct_change(w, fill_method=None)

    if market is not None:
        out["market_relative_strength_63d"] = (
            out[f"return_{RELATIVE_STRENGTH_WINDOW}d"] - g["date"].map(market["return_63d"])
        )
        market_ret = g["date"].map(market["log_return_1d"])
        out["beta_126d"] = (
            log_ret.rolling(BETA_WINDOW).cov(market_ret)
            / market_ret.rolling(BETA_WINDOW).var()
        )
    else:
        out["market_relative_strength_63d"] = np.nan
        out["beta_126d"] = np.nan

    out["realized_volatility_21d"] = (
        log_ret.rolling(VOLATILITY_WINDOW).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    )
    out["max_drawdown_63d"] = adj.rolling(DRAWDOWN_WINDOW).apply(_max_drawdown, raw=True)

    dollar_volume = (close * volume).rolling(DOLLAR_VOLUME_WINDOW).mean()
    out["dollar_volume_20d_log"] = np.log(dollar_volume.where(dollar_volume > 0))

    return out.reset_index(drop=True)


def _price_features(ohlcv: pd.DataFrame, benchmark: str = BENCHMARK_TICKER) -> pd.DataFrame:
    """Price-derived feature panel for every non-benchmark ticker."""
    benchmark_bars = ohlcv[ohlcv["ticker"] == benchmark]
    market = _market_frame(benchmark_bars) if not benchmark_bars.empty else None
    if market is None:
        logger.warning(
            "benchmark %s not in ohlcv — market_relative_strength_63d and "
            "beta_126d will be missing (backfill it via ohlcv.py)", benchmark,
        )

    frames = [
        _single_ticker_price_features(bars, market)
        for ticker, bars in ohlcv.groupby("ticker", sort=False)
        if ticker != benchmark
    ]
    if not frames:
        raise ValueError("ohlcv has no non-benchmark tickers — run ingestion first")
    return pd.concat(frames, ignore_index=True)


# ── Table joins (pure — fully unit-tested) ───────────────────────────────────

def _join_fundamentals(panel: pd.DataFrame, fundamentals: pd.DataFrame) -> pd.DataFrame:
    """Point-in-time as-of join of fundamental snapshots on fetched_at.

    Dates before a ticker's first snapshot are backfilled from it — see the
    module docstring for the lookahead caveat. net_debt_to_ebitda stays
    missing until fundamentals.py ingests ebitda (noted in features.py).
    """
    if fundamentals.empty:
        return panel

    snapshots = fundamentals.copy()
    snapshots["date"] = normalize_trading_date(snapshots["fetched_at"])
    for column in _FUNDAMENTAL_COLUMNS:
        snapshots[column] = pd.to_numeric(snapshots[column], errors="coerce")
    snapshots = (
        snapshots.sort_values("date", kind="stable")
                 .drop_duplicates(["ticker", "date"], keep="last")
                 .loc[:, ["ticker", "date", *_FUNDAMENTAL_COLUMNS]]
    )

    merged = pd.merge_asof(
        panel.sort_values("date", kind="stable"),
        snapshots,
        on="date", by="ticker", direction="backward",
    )
    columns = list(_FUNDAMENTAL_COLUMNS)
    merged[columns] = merged.groupby("ticker")[columns].bfill()
    return merged


def _quarter_availability_date(quarter: str) -> pd.Timestamp | None:
    """'2024Q1' → calendar quarter end + availability lag (NaT-safe)."""
    try:
        period = pd.Period(str(quarter), freq="Q")
    except Exception:
        return None
    return period.end_time.normalize() + pd.Timedelta(days=TRANSCRIPT_AVAILABILITY_LAG_DAYS)


def _sentiment_features(sentiment: pd.DataFrame) -> pd.DataFrame:
    """Per-(ticker, availability date) transcript sentiment features.

    FinBERT segment scores are signed by label (positive → +score,
    negative → −score, neutral → 0) and averaged per (ticker, quarter, role),
    giving scores in [-1, 1]. QoQ change diffs consecutive *ingested*
    quarters — a gap in transcript coverage spans the gap.
    """
    s = sentiment.copy()
    s["signed"] = pd.to_numeric(s["score"], errors="coerce") * s["label"].map(SENTIMENT_LABEL_SIGN)
    s = s.dropna(subset=["signed"])
    if s.empty:
        return pd.DataFrame(columns=["ticker", "date", *_SENTIMENT_COLUMNS])

    agg = (
        s.groupby(["ticker", "quarter", "segment_role"])["signed"].mean()
         .unstack("segment_role")
         .rename(columns={"management": "mgmt_sentiment_score", "qa": "qa_sentiment_score"})
         .reset_index()
    )
    for column in ("mgmt_sentiment_score", "qa_sentiment_score"):
        if column not in agg.columns:
            agg[column] = np.nan

    agg["date"] = pd.to_datetime(agg["quarter"].map(_quarter_availability_date))
    agg = agg.dropna(subset=["date"]).sort_values(["ticker", "date"], kind="stable")

    agg["mgmt_qa_sentiment_gap"] = agg["mgmt_sentiment_score"] - agg["qa_sentiment_score"]
    agg["transcript_sentiment_change_qoq"] = agg.groupby("ticker")["mgmt_sentiment_score"].diff()
    return agg.loc[:, ["ticker", "date", *_SENTIMENT_COLUMNS]].reset_index(drop=True)


def _join_sentiment(panel: pd.DataFrame, sentiment: pd.DataFrame) -> pd.DataFrame:
    """As-of join transcript sentiment: each trading date sees the most
    recent transcript already available on that date (no lookahead)."""
    if sentiment.empty:
        return panel
    features = _sentiment_features(sentiment)
    if features.empty:
        return panel
    return pd.merge_asof(
        panel.sort_values("date", kind="stable"),
        features.sort_values("date", kind="stable"),
        on="date", by="ticker", direction="backward",
    )


def _macro_features(macro: pd.DataFrame, trading_dates: Iterable[pd.Timestamp]) -> pd.DataFrame:
    """Macro features aligned to trading dates (same values for all tickers).

    Each trading date gets the last observation on or before it (as-of,
    forward-filled), then trend features are derived on the trading-date
    axis so `_3m` means 63 trading days, consistent with features.py.
    """
    index = pd.DatetimeIndex(sorted(set(trading_dates)), name="date")
    columns = [*_MACRO_BASE_FEATURES, *_MACRO_DERIVED_FEATURES]
    if macro.empty:
        return pd.DataFrame(np.nan, index=index, columns=columns)

    m = macro.copy()
    m["feature"] = m["series_id"].map(
        lambda sid: MACRO_SERIES_TO_FEATURE.get(sid, sid if sid in _MACRO_BASE_FEATURES else None)
    )
    unknown = m.loc[m["feature"].isna(), "series_id"].unique()
    if len(unknown):
        logger.warning("ignoring unmapped macro series ids: %s", sorted(unknown))
    m = m.dropna(subset=["feature"])
    if m.empty:
        return pd.DataFrame(np.nan, index=index, columns=columns)

    wide = m.pivot_table(index="observation_date", columns="feature", values="value", aggfunc="last")
    wide.index = pd.to_datetime(wide.index)
    wide = wide.sort_index().reindex(index, method="ffill")

    for column in _MACRO_BASE_FEATURES:
        if column not in wide.columns:
            wide[column] = np.nan

    wide["fed_funds_change_3m"] = wide["fed_funds_rate"] - wide["fed_funds_rate"].shift(MACRO_CHANGE_WINDOW)
    wide["cpi_trend_3m"] = wide["cpi_yoy"] - wide["cpi_yoy"].shift(MACRO_CHANGE_WINDOW)
    wide["unemployment_change_3m"] = wide["unemployment"] - wide["unemployment"].shift(MACRO_CHANGE_WINDOW)
    wide["real_treasury_10y"] = wide["treasury_10y"] - wide["cpi_yoy"]
    return wide.loc[:, columns]


def _join_macro(panel: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    features = _macro_features(macro, panel["date"].unique())
    for column in features.columns:
        panel[column] = panel["date"].map(features[column])
    return panel


# ── Assembly ─────────────────────────────────────────────────────────────────

def build_feature_frame(session: Session) -> pd.DataFrame:
    """Assemble the full feature panel: (ticker, date) index × FEATURE_ORDER.

    Every FEATURE_ORDER column is present; features whose source has no
    historical store yet (options_flow.py / analyst_ratings.py pending,
    insider Form 4 scores computed live and not persisted, ebitda missing
    from fundamentals) are all-NaN and handled by imputation + indicators.
    """
    ohlcv = _load_ohlcv(session)
    if ohlcv.empty:
        raise ValueError("ohlcv table is empty — run OHLCV ingestion before building the dataset")

    panel = _price_features(ohlcv)
    panel = _join_fundamentals(panel, _load_fundamentals(session))
    panel = _join_sentiment(panel, _load_sentiment(session))
    panel = _join_macro(panel, _load_macro(session))

    for column in FEATURE_ORDER:
        if column not in panel.columns:
            panel[column] = np.nan

    frame = panel.set_index(["ticker", "date"]).loc[:, list(FEATURE_ORDER)].sort_index()

    empty_features = [c for c in FEATURE_ORDER if frame[c].isna().all()]
    if empty_features:
        logger.warning(
            "features with no data yet (imputed to 0.0, missing indicator = 1): %s",
            empty_features,
        )
    return frame


def impute_features(
    features: pd.DataFrame,
    medians: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fill NaNs with the given (train-fit) medians; 0.0 where the median
    itself is NaN (all-missing column). Returns (imputed, missing_indicators)
    where the indicator frame is 1.0 exactly where a value was imputed.
    """
    missing = features.isna()
    imputed = features.fillna(medians).fillna(0.0)
    return imputed, missing.astype(np.float32)


def split_by_time(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Time-based split on the date index level: train ≤ 2022, val 2023, test 2024+.

    Note: forward-return labels near a boundary realise inside the next
    split's window (a 2022-12 row's 12M label uses 2023 prices). This overlap
    is inherent to forward labels with a simple time split; purging it would
    cost up to max(HORIZONS) days of each split, so it is documented rather
    than removed.
    """
    dates = frame.index.get_level_values("date")
    return {
        "train": frame.loc[dates <= TRAIN_END],
        "val": frame.loc[(dates > TRAIN_END) & (dates <= VAL_END)],
        "test": frame.loc[dates > VAL_END],
    }


# ── torch Dataset ────────────────────────────────────────────────────────────

class AnalystMindDataset(Dataset):
    """Tabular dataset yielding ``(feature_tensor, label_tensor)`` per row.

    - ``feature_tensor``: float32, shape ``(N_FEATURES,)`` in FEATURE_ORDER —
      matches tokenizer.FeatureTokenizer input.
    - ``label_tensor``: float32, shape ``(len(HORIZONS),)`` in LABEL_COLUMNS
      order. NaN-free when built with ``require_all_horizons=True`` (default);
      otherwise longer horizons may be NaN and the loss must mask them.
    - ``missing_mask``: float32 ``(len, N_FEATURES)`` attribute, 1.0 where the
      feature value was imputed — kept out of the feature tensor to preserve
      the N_FEATURES width contract with the tokenizer.
    - ``index``: the (ticker, date) MultiIndex, for traceability/debugging.
    """

    def __init__(
        self,
        features: pd.DataFrame,
        labels: pd.DataFrame,
        missing_mask: pd.DataFrame | None = None,
    ) -> None:
        if list(features.columns) != list(FEATURE_ORDER):
            raise ValueError("features columns must be exactly FEATURE_ORDER")
        if list(labels.columns) != list(LABEL_COLUMNS):
            raise ValueError("labels columns must be exactly LABEL_COLUMNS")
        if not features.index.equals(labels.index):
            raise ValueError("features and labels must share the same (ticker, date) index")

        self.index = features.index
        # copy=True: to_numpy may hand back a read-only view, which torch
        # cannot safely wrap.
        self.features = torch.from_numpy(features.to_numpy(dtype=np.float32, copy=True))
        self.labels = torch.from_numpy(labels.to_numpy(dtype=np.float32, copy=True))
        if missing_mask is None:
            self.missing_mask = torch.zeros_like(self.features)
        else:
            if not missing_mask.index.equals(features.index):
                raise ValueError("missing_mask must share the features index")
            self.missing_mask = torch.from_numpy(missing_mask.to_numpy(dtype=np.float32, copy=True))

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]

    def __repr__(self) -> str:  # pragma: no cover — debugging aid
        return (
            f"{type(self).__name__}(rows={len(self)}, "
            f"features={self.features.shape[1]}, horizons={self.labels.shape[1]})"
        )


@dataclass(frozen=True)
class DatasetBundle:
    """Train/val/test datasets plus the imputation medians fit on train.

    ``feature_medians`` must be reused when building feature vectors at
    inference time (inference.py) so serving matches training exactly.
    """
    train: AnalystMindDataset
    val: AnalystMindDataset
    test: AnalystMindDataset
    feature_medians: pd.Series


# ── Public API ───────────────────────────────────────────────────────────────

def build_datasets(session: Session, require_all_horizons: bool = True) -> DatasetBundle:
    """Build the full train/val/test bundle from the database.

    Parameters
    ----------
    require_all_horizons : bool
        True (default): only rows with every horizon label — NaN-free label
        tensors. Requires ≥ max(HORIZONS) trading days of ohlcv history past
        each training date; with a short backfill this can empty the dataset
        (the error below says so). False: rows need ≥ 1 label and train.py
        must mask NaN labels in the loss.
    """
    features = build_feature_frame(session)
    labels = build_labels(session, require_all_horizons=require_all_horizons)

    common = features.index.intersection(labels.index)
    if common.empty:
        raise ValueError(
            "no (ticker, date) rows have both features and labels — "
            "likely too little ohlcv history for the label horizons "
            f"(max horizon = {max(HORIZONS)} trading days); backfill ohlcv "
            "or relax require_all_horizons"
        )
    features = features.loc[common].sort_index()
    labels = labels.loc[common].sort_index()

    feature_splits = split_by_time(features)
    label_splits = split_by_time(labels)
    if feature_splits["train"].empty:
        raise ValueError(
            f"train split is empty (no rows on or before {TRAIN_END.date()}) — "
            "backfill ohlcv history before training"
        )

    # Medians fit on train only, applied everywhere — no leakage from
    # val/test into imputation.
    medians = feature_splits["train"].median()

    datasets: dict[str, AnalystMindDataset] = {}
    for name in SPLIT_NAMES:
        split_features, missing_mask = impute_features(feature_splits[name], medians)
        datasets[name] = AnalystMindDataset(split_features, label_splits[name], missing_mask)
        logger.info("%s split: %d rows", name, len(datasets[name]))

    return DatasetBundle(feature_medians=medians, **datasets)
