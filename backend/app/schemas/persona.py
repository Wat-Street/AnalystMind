"""Pydantic models returned by inference.py and consumed by routes.py POST /analyze.

Field shapes mirror the ORM models in app.models.db (PersonaOutput, ConsensusPT);
the ge/le bounds mirror the matching CheckConstraints on those tables.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class PersonaOutput(BaseModel):
    ticker: str
    persona_name: str
    pt_base: float
    pt_bull: float | None = None
    pt_bear: float | None = None
    confidence: float = Field(ge=0, le=1)
    rationale: str | None = None


class ConsensusPT(BaseModel):
    ticker: str
    consensus_pt: float
    band_low: float
    band_high: float
    conviction_score: float = Field(ge=0, le=1)
    dominant_thesis: str | None = None
    outlier_persona: str | None = None
    outlier_pt: float | None = None
