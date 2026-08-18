"""Pydantic response models for persona and consensus inference."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class PersonaOutput(BaseModel):
    """One model persona's price-target view for a ticker."""

    model_config = ConfigDict(from_attributes=True)

    ticker: str
    persona_name: str
    pt_base: float
    pt_bull: float | None = None
    pt_bear: float | None = None
    confidence: float = Field(ge=0, le=1)
    rationale: str | None = None


class ConsensusPT(BaseModel):
    """The confidence-weighted consensus price target for a ticker."""

    model_config = ConfigDict(from_attributes=True)

    ticker: str
    consensus_pt: float
    band_low: float
    band_high: float
    conviction_score: float = Field(ge=0, le=1)
    dominant_thesis: str | None = None
    outlier_persona: str | None = None
    outlier_pt: float | None = None
