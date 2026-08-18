"""Run persona-masked model inference and aggregate a consensus price target.

The training pipeline stores feature vectors in the order defined by
``features.FEATURE_ORDER``.  Serving must rebuild that same vector, apply the
train-split imputation medians, and only then call the model.  This module keeps
all of that data preparation together with the model call so API callers do
not accidentally create a differently ordered feature tensor.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import yaml
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..schemas.persona import ConsensusPT, PersonaOutput
from .backbone import FTTransformerModel
from .dataset import TRAIN_END, build_feature_frame, impute_features
from .features import FEATURE_ORDER, HORIZONS, PERSONA_MODALITIES
from .train import CHECKPOINT_PATH, MODEL_CONFIG

PERSONAS_DIR = Path(__file__).resolve().parents[3] / "personas"

_H1M = HORIZONS.index(21)
_H3M = HORIZONS.index(63)
_H6M = HORIZONS.index(126)

OUTLIER_SIGMA = 1.5
OUTLIER_WEIGHT = 0.30
ALIGNED_SIGMA = 0.50

_SNAPSHOT_CACHE_KEY = "analystmind.inference.snapshots"
_OUTPUT_CACHE_KEY = "analystmind.inference.persona_outputs"


def load_model(
    checkpoint_path: str | Path,
    device: torch.device | None = None,
) -> FTTransformerModel:
    """Rebuild a model from a training checkpoint and put it in eval mode.

    Training checkpoints contain both ``state_dict`` and the model ``config``.
    A bare state dictionary is also accepted for compatibility with manually
    saved checkpoints; in that case the training defaults are used.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"no checkpoint at {checkpoint_path} — run `python -m app.ml.train` first"
        )

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint: Any = torch.load(checkpoint_path, map_location=device)

    if not isinstance(checkpoint, dict):
        raise ValueError(f"checkpoint at {checkpoint_path} must contain a state dictionary")

    state_dict = checkpoint.get("state_dict", checkpoint)
    config = checkpoint.get("config", MODEL_CONFIG)
    if not isinstance(state_dict, dict):
        raise ValueError(f"checkpoint at {checkpoint_path} has no valid state_dict")
    if not isinstance(config, dict):
        raise ValueError(f"checkpoint at {checkpoint_path} has no valid model config")

    model = FTTransformerModel(**config).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _persona_display_name(persona_name: str) -> str:
    return persona_name.replace("_", " ").title()


def _top_weight_factor(persona_name: str) -> str | None:
    """Return the highest-weighted display factor from a persona YAML file."""
    path = PERSONAS_DIR / f"{persona_name}.yaml"
    if not path.exists():
        return None

    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(config, dict):
        return None
    weights = config.get("weights") or {}
    if not isinstance(weights, dict) or not weights:
        return None
    return max(weights, key=weights.get)


def _session_info(session: Session) -> dict[str, Any] | None:
    """Return SQLAlchemy's per-session cache, if the object provides one."""
    info = getattr(session, "info", None)
    return info if isinstance(info, dict) else None


class PersonaInference:
    """Run the trained model through each persona's modality mask."""

    def __init__(
        self,
        checkpoint_path: str | Path = CHECKPOINT_PATH,
        device: torch.device | None = None,
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model(checkpoint_path, self.device)

    @staticmethod
    def load_model(
        checkpoint_path: str | Path,
        device: torch.device | None = None,
    ) -> FTTransformerModel:
        """Load a checkpoint; exposed on the class for service callers/tests."""
        return load_model(checkpoint_path, device)

    @staticmethod
    def _current_price(session: Session, ticker: str) -> float:
        """Return the most recent unadjusted close for *ticker*."""
        from ..models.db import OHLCV

        row = session.execute(
            select(OHLCV.close)
            .where(OHLCV.ticker == ticker)
            .order_by(OHLCV.time.desc())
            .limit(1)
        ).first()
        if row is None or row[0] is None:
            raise ValueError(f"no OHLCV close price for ticker {ticker!r}")

        price = float(row[0])
        if not math.isfinite(price) or price <= 0:
            raise ValueError(f"latest close price for ticker {ticker!r} is invalid: {price!r}")
        return price

    def _latest_snapshot(self, session: Session, ticker: str) -> tuple[float, torch.Tensor]:
        """Build the latest canonical feature vector and current price.

        The SQLAlchemy session cache makes the nine persona calls in one
        request reuse the same feature frame and imputation vector.  A fresh
        API request gets a fresh session, so a later ingestion run is visible.
        """
        cache = _session_info(session)
        if cache is not None:
            snapshots = cache.setdefault(_SNAPSHOT_CACHE_KEY, {})
            if ticker in snapshots:
                return snapshots[ticker]
        else:
            snapshots = None

        frame = build_feature_frame(session)
        if frame.empty or "ticker" not in frame.index.names or "date" not in frame.index.names:
            raise ValueError(f"no ingested feature data for ticker {ticker!r}")

        tickers = frame.index.get_level_values("ticker")
        if ticker not in tickers:
            raise ValueError(f"no ingested feature data for ticker {ticker!r}")

        missing_columns = [name for name in FEATURE_ORDER if name not in frame.columns]
        if missing_columns:
            raise ValueError(f"feature frame is missing required columns: {missing_columns}")

        ticker_frame = frame.loc[tickers == ticker, list(FEATURE_ORDER)]
        latest_date = ticker_frame.index.get_level_values("date").max()
        latest_row = ticker_frame.loc[[(ticker, latest_date)]]

        train_dates = frame.index.get_level_values("date") <= TRAIN_END
        train_frame = frame.loc[train_dates, list(FEATURE_ORDER)]
        medians = train_frame.median(numeric_only=True).reindex(FEATURE_ORDER)
        imputed, _ = impute_features(latest_row, medians)
        values = imputed.to_numpy(dtype=np.float32, copy=True)
        if values.shape != (1, len(FEATURE_ORDER)) or not np.isfinite(values).all():
            raise ValueError(f"could not build a finite feature vector for ticker {ticker!r}")

        snapshot = (self._current_price(session, ticker), torch.from_numpy(values[0]))
        if snapshots is not None:
            snapshots[ticker] = snapshot
        return snapshot

    @staticmethod
    def _confidence(predictions: Sequence[float]) -> float:
        """Use 1M/3M/6M sign agreement as an inference-time confidence proxy."""
        anchor_is_up = predictions[_H3M] > 0
        agreements = sum(
            (predictions[index] > 0) == anchor_is_up
            for index in (_H1M, _H3M, _H6M)
        )
        return agreements / 3.0

    @torch.no_grad()
    def run_persona(
        self,
        ticker: str,
        persona_name: str,
        session: Session,
    ) -> PersonaOutput:
        """Run one persona against the latest feature snapshot for *ticker*."""
        ticker = ticker.strip().upper()
        if not ticker:
            raise ValueError("ticker is required")
        if persona_name not in PERSONA_MODALITIES:
            raise KeyError(f"unknown persona {persona_name!r}")

        current_price, features = self._latest_snapshot(session, ticker)
        predictions = self.model(
            features.unsqueeze(0).to(self.device),
            persona_name=persona_name,
        )
        if not isinstance(predictions, torch.Tensor):
            predictions = torch.as_tensor(predictions)
        if predictions.shape != (1, len(HORIZONS)):
            raise RuntimeError(
                f"model returned shape {tuple(predictions.shape)}, "
                f"expected (1, {len(HORIZONS)})"
            )

        values = [float(value) for value in predictions[0].detach().cpu()]
        if not all(math.isfinite(value) for value in values):
            raise RuntimeError(f"model returned a non-finite prediction for {ticker!r}")

        three_month_return = values[_H3M]
        return PersonaOutput(
            ticker=ticker,
            persona_name=persona_name,
            pt_base=current_price * (1.0 + three_month_return),
            pt_bull=current_price * (1.0 + max(values)),
            pt_bear=current_price * (1.0 + min(values)),
            confidence=self._confidence(values),
            rationale=(
                f"{_persona_display_name(persona_name)} projects a "
                f"{three_month_return:+.1%} 3M return on {ticker}, using "
                f"{', '.join(PERSONA_MODALITIES[persona_name])} signals."
            ),
        )

    def run_all_personas(self, ticker: str, session: Session) -> list[PersonaOutput]:
        """Run all nine configured model personas for *ticker*."""
        # Resolve the mapping at call time so a deployment that extends the
        # shared persona registry cannot silently use a stale import-time list.
        return [self.run_persona(ticker, name, session) for name in PERSONA_MODALITIES]

    @staticmethod
    def _dominant_thesis(
        aligned: Sequence[PersonaOutput],
        total_personas: int,
    ) -> str | None:
        if not aligned:
            return None

        factors = []
        for output in aligned:
            factor = _top_weight_factor(output.persona_name)
            label = _persona_display_name(output.persona_name)
            factors.append(f"{label} ({factor})" if factor else label)
        return (
            f"Aligned cluster ({len(aligned)}/{total_personas} personas): "
            f"{', '.join(factors)}."
        )

    def aggregate_consensus(
        self,
        ticker: str,
        outputs: Sequence[PersonaOutput],
    ) -> ConsensusPT:
        """Compute the confidence-weighted, outlier-adjusted consensus."""
        if not outputs:
            raise ValueError("at least one persona output is required")

        pts = np.asarray([float(output.pt_base) for output in outputs], dtype=np.float64)
        confidences = np.asarray(
            [float(output.confidence) for output in outputs],
            dtype=np.float64,
        )
        if not np.isfinite(pts).all() or not np.isfinite(confidences).all():
            raise ValueError("persona outputs must contain finite price targets and confidences")
        if (confidences < 0).any():
            raise ValueError("persona confidences must be non-negative")

        if confidences.sum() > 0:
            initial_consensus = float(np.average(pts, weights=confidences))
        else:
            initial_consensus = float(pts.mean())
        sigma = float(pts.std(ddof=0))

        is_outlier = np.zeros(len(outputs), dtype=bool)
        if sigma > 0:
            is_outlier = np.abs(pts - initial_consensus) > OUTLIER_SIGMA * sigma
            adjusted_weights = np.where(
                is_outlier,
                confidences * OUTLIER_WEIGHT,
                confidences,
            )
            if adjusted_weights.sum() > 0:
                consensus = float(np.average(pts, weights=adjusted_weights))
            else:
                consensus = float(pts.mean())
        else:
            consensus = initial_consensus

        conviction = 1.0 - (sigma / consensus) if consensus != 0 else 0.0
        conviction_score = max(0.0, min(1.0, conviction))

        outlier_persona: str | None = None
        outlier_pt: float | None = None
        if is_outlier.any():
            distances = np.where(is_outlier, np.abs(pts - initial_consensus), -np.inf)
            outlier_index = int(np.argmax(distances))
            outlier_persona = outputs[outlier_index].persona_name
            outlier_pt = float(outputs[outlier_index].pt_base)

        if sigma > 0:
            aligned = [
                output
                for output in outputs
                if abs(float(output.pt_base) - consensus) <= ALIGNED_SIGMA * sigma
            ]
        else:
            aligned = list(outputs)

        return ConsensusPT(
            ticker=ticker.strip().upper(),
            consensus_pt=consensus,
            band_low=consensus - sigma,
            band_high=consensus + sigma,
            conviction_score=conviction_score,
            dominant_thesis=self._dominant_thesis(aligned, len(outputs)),
            outlier_persona=outlier_persona,
            outlier_pt=outlier_pt,
        )

    def _store_outputs(
        self,
        session: Session,
        ticker: str,
        outputs: list[PersonaOutput],
    ) -> None:
        info = _session_info(session)
        if info is not None:
            output_cache = info.setdefault(_OUTPUT_CACHE_KEY, {})
            output_cache[ticker] = outputs

    @staticmethod
    def get_last_persona_outputs(
        session: Session,
        ticker: str,
    ) -> list[PersonaOutput]:
        """Return outputs generated by the current session's last consensus run."""
        info = _session_info(session)
        if info is None:
            return []
        output_cache = info.get(_OUTPUT_CACHE_KEY, {})
        return output_cache.pop(ticker.strip().upper(), [])

    def run_consensus(self, ticker: str, session: Session) -> ConsensusPT:
        """Run every persona and aggregate the resulting price targets."""
        ticker = ticker.strip().upper()
        if not ticker:
            raise ValueError("ticker is required")
        outputs = self.run_all_personas(ticker, session)
        self._store_outputs(session, ticker, outputs)
        return self.aggregate_consensus(ticker, outputs)
