"""Persona-level and consensus inference against a trained checkpoint.

Loads the FT-Transformer checkpoint once, runs each of the 9 personas'
modality-masked forward pass for a ticker's latest feature snapshot, and
aggregates them into a consensus price target per the algorithm described in
ARCHITECTURE.md / CLAUDE.md's "Consensus aggregation" section.

Design notes (decisions not otherwise specified in the codebase):
  - confidence: horizon-agreement heuristic — the fraction of {1M, 3M, 6M}
    predicted-return signs that match the 3M sign (the anchor horizon used
    for pt_base). No ground truth exists at inference time, so this is a
    self-consistency signal, not a calibrated probability.
  - pt_bull / pt_bear: current_price * (1 + max/min predicted return) across
    all HORIZONS for that persona's mask.
  - dominant_thesis: a deterministic template (persona names + each aligned
    persona's top-weighted YAML factor) — no LLM call, to avoid an unconfigured
    OpenAI dependency. Swap for a real LLM call once an API key is wired up.

Depends on: backbone.py (FTTransformerModel), dataset.py (feature assembly),
evaluate.py (checkpoint loading), features.py, schemas/persona.py.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import yaml
from sqlalchemy.orm import Session

from ..schemas.persona import ConsensusPT, PersonaOutput
from .backbone import FTTransformerModel
from .dataset import TRAIN_END, build_feature_frame, impute_features
from .evaluate import load_model
from .features import HORIZONS, PERSONA_MODALITIES
from .train import CHECKPOINT_PATH

PERSONAS_DIR = Path(__file__).resolve().parents[3] / "personas"

_H1M = HORIZONS.index(21)
_H3M = HORIZONS.index(63)
_H6M = HORIZONS.index(126)

OUTLIER_SIGMA = 1.5
OUTLIER_WEIGHT = 0.30
ALIGNED_SIGMA = 0.5


def _persona_display_name(persona_name: str) -> str:
    return persona_name.replace("_", " ").title()


def _top_weight_factor(persona_name: str) -> str | None:
    """The most heavily-weighted key in a persona's YAML `weights` block.

    These are display-oriented labels (CLAUDE.md's persona table), not
    FEATURE_ORDER names — used only for the dominant_thesis string.
    """
    path = PERSONAS_DIR / f"{persona_name}.yaml"
    if not path.exists():
        return None
    config = yaml.safe_load(path.read_text())
    weights = config.get("weights") or {}
    if not weights:
        return None
    return max(weights, key=weights.get)


class PersonaInference:
    """Runs the trained backbone through each persona's modality mask."""

    def __init__(
        self,
        checkpoint_path: Path = CHECKPOINT_PATH,
        device: torch.device | None = None,
    ) -> None:
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"no checkpoint at {checkpoint_path} — run `python -m app.ml.train` first"
            )
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: FTTransformerModel = load_model(checkpoint_path, self.device)

    def _latest_snapshot(self, session: Session, ticker: str) -> tuple[float, torch.Tensor]:
        """Return (current_price, feature_tensor) for *ticker*'s most recent trading date.

        Medians are fit fresh on the train split (<= TRAIN_END) of the full
        feature panel, matching how dataset.py fits imputation medians for
        training — the checkpoint itself doesn't persist them.
        """
        frame = build_feature_frame(session)
        if ticker not in frame.index.get_level_values("ticker"):
            raise ValueError(f"no ingested data for ticker {ticker!r}")

        ticker_frame = frame.xs(ticker, level="ticker", drop_level=False)
        latest_date = ticker_frame.index.get_level_values("date").max()
        latest_row = ticker_frame.loc[[(ticker, latest_date)]]

        train_frame = frame.loc[frame.index.get_level_values("date") <= TRAIN_END]
        medians = train_frame.median(numeric_only=True)

        imputed, _ = impute_features(latest_row, medians)
        x = torch.from_numpy(imputed.to_numpy(dtype=np.float32)[0])

        current_price = self._current_price(session, ticker)
        return current_price, x

    @staticmethod
    def _current_price(session: Session, ticker: str) -> float:
        from sqlalchemy import select

        from ..models.db import OHLCV

        row = session.execute(
            select(OHLCV.close)
            .where(OHLCV.ticker == ticker)
            .order_by(OHLCV.time.desc())
            .limit(1)
        ).first()
        if row is None or row[0] is None:
            raise ValueError(f"no ohlcv close price for ticker {ticker!r}")
        return float(row[0])

    @staticmethod
    def _confidence(preds: list[float]) -> float:
        """Fraction of {1M, 3M, 6M} predicted-return signs matching the 3M sign."""
        sign_3m = preds[_H3M] > 0
        agree = sum(1 for i in (_H1M, _H3M, _H6M) if (preds[i] > 0) == sign_3m)
        return agree / 3

    @torch.no_grad()
    def run_persona(self, ticker: str, persona_name: str, session: Session) -> PersonaOutput:
        """Run one persona's modality-masked forward pass for *ticker*."""
        if persona_name not in PERSONA_MODALITIES:
            raise KeyError(f"unknown persona {persona_name!r}")

        current_price, x = self._latest_snapshot(session, ticker)
        preds = self.model(x.unsqueeze(0).to(self.device), persona_name=persona_name)
        preds = preds.squeeze(0).cpu().tolist()

        r_3m = preds[_H3M]
        pt_base = current_price * (1 + r_3m)
        pt_bull = current_price * (1 + max(preds))
        pt_bear = current_price * (1 + min(preds))
        confidence = self._confidence(preds)

        rationale = (
            f"{_persona_display_name(persona_name)} projects a {r_3m:+.1%} 3M return on "
            f"{ticker}, viewing it through {', '.join(PERSONA_MODALITIES[persona_name])} signals."
        )

        return PersonaOutput(
            ticker=ticker,
            persona_name=persona_name,
            pt_base=pt_base,
            pt_bull=pt_bull,
            pt_bear=pt_bear,
            confidence=confidence,
            rationale=rationale,
        )

    def _dominant_thesis(self, aligned: list[PersonaOutput], total: int) -> str | None:
        if not aligned:
            return None
        parts = []
        for output in aligned:
            factor = _top_weight_factor(output.persona_name)
            label = _persona_display_name(output.persona_name)
            parts.append(f"{label} ({factor})" if factor else label)
        return f"Aligned cluster ({len(aligned)}/{total} personas): {', '.join(parts)}."

    def run_all_personas(self, ticker: str, session: Session) -> list[PersonaOutput]:
        """Run every persona's modality-masked forward pass for *ticker*."""
        return [self.run_persona(ticker, persona, session) for persona in PERSONA_MODALITIES]

    def aggregate_consensus(self, ticker: str, outputs: list[PersonaOutput]) -> ConsensusPT:
        """Aggregate already-computed persona outputs into a consensus price target.

        Split out from run_consensus so callers that also need the individual
        PersonaOutputs (e.g. to persist them) don't have to run personas twice.
        """
        pts = np.array([o.pt_base for o in outputs], dtype=np.float64)
        confidences = np.array([o.confidence for o in outputs], dtype=np.float64)

        mean = float(np.average(pts, weights=confidences)) if confidences.sum() > 0 else float(pts.mean())
        sigma = float(pts.std(ddof=0))

        is_outlier = np.zeros(len(outputs), dtype=bool)
        if sigma > 0:
            is_outlier = np.abs(pts - mean) > OUTLIER_SIGMA * sigma
            weights = np.where(is_outlier, confidences * OUTLIER_WEIGHT, confidences)
            if weights.sum() == 0:
                weights = np.ones_like(weights)
            consensus_final = float(np.average(pts, weights=weights))
        else:
            consensus_final = mean

        conviction = 1.0 - (sigma / consensus_final) if consensus_final else 0.0
        conviction_score = max(0.0, min(1.0, conviction))

        band_low = consensus_final - sigma
        band_high = consensus_final + sigma

        outlier_persona = outlier_pt = None
        if is_outlier.any():
            idx = int(np.argmax(np.where(is_outlier, np.abs(pts - mean), -np.inf)))
            outlier_persona = outputs[idx].persona_name
            outlier_pt = outputs[idx].pt_base

        if sigma > 0:
            aligned = [o for o in outputs if abs(o.pt_base - consensus_final) <= ALIGNED_SIGMA * sigma]
        else:
            aligned = outputs
        dominant_thesis = self._dominant_thesis(aligned, len(outputs))

        return ConsensusPT(
            ticker=ticker,
            consensus_pt=consensus_final,
            band_low=band_low,
            band_high=band_high,
            conviction_score=conviction_score,
            dominant_thesis=dominant_thesis,
            outlier_persona=outlier_persona,
            outlier_pt=outlier_pt,
        )

    def run_consensus(self, ticker: str, session: Session) -> ConsensusPT:
        """Run every persona and aggregate into a single consensus price target."""
        outputs = self.run_all_personas(ticker, session)
        return self.aggregate_consensus(ticker, outputs)
