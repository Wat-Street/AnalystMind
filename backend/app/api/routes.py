"""All API route handlers.

GET endpoints are read-only cache hits against the DB — they never trigger
persona/consensus computation. Only POST /analyze runs the model.
"""
from __future__ import annotations

import logging
from pathlib import Path

import yaml
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..ml.inference import PERSONAS_DIR, PersonaInference
from ..models.db import ConsensusPT as ConsensusPTRow
from ..models.db import PersonaOutput as PersonaOutputRow
from ..models.db import engine
from ..schemas.persona import ConsensusPT

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Dependencies ─────────────────────────────────────────────────────────────

def get_session():
    with Session(engine) as session:
        yield session


_inference: PersonaInference | None = None


def get_inference() -> PersonaInference:
    """Lazily load the checkpoint once and reuse it across requests."""
    global _inference
    if _inference is None:
        try:
            _inference = PersonaInference()
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
    return _inference


def load_personas() -> list[dict]:
    """Load every personas/*.yaml config, sorted by name."""
    configs = []
    for path in sorted(PERSONAS_DIR.glob("*.yaml")):
        configs.append(yaml.safe_load(path.read_text()))
    return configs


# ── POST /analyze ────────────────────────────────────────────────────────────

@router.post("/analyze", response_model=ConsensusPT)
def analyze(
    ticker: str,
    session: Session = Depends(get_session),
    inference: PersonaInference = Depends(get_inference),
) -> ConsensusPT:
    ticker = ticker.strip().upper()
    if not ticker:
        raise HTTPException(status_code=400, detail="ticker is required")

    try:
        persona_outputs = inference.run_all_personas(ticker, session)
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    consensus = inference.aggregate_consensus(ticker, persona_outputs)

    for output in persona_outputs:
        session.add(PersonaOutputRow(**output.model_dump()))
    session.add(ConsensusPTRow(**consensus.model_dump()))
    session.commit()

    return consensus


# ── GET /api/stock/:ticker ───────────────────────────────────────────────────

@router.get("/api/stock/{ticker}", response_model=ConsensusPT)
def get_stock(ticker: str, session: Session = Depends(get_session)) -> ConsensusPT:
    ticker = ticker.strip().upper()
    row = session.execute(
        select(ConsensusPTRow)
        .where(ConsensusPTRow.ticker == ticker)
        .order_by(ConsensusPTRow.computed_at.desc())
        .limit(1)
    ).scalar_one_or_none()

    if row is None:
        raise HTTPException(
            status_code=404,
            detail=f"no cached consensus for {ticker!r} — run POST /analyze?ticker={ticker} first",
        )

    return ConsensusPT(
        ticker=row.ticker,
        consensus_pt=float(row.consensus_pt),
        band_low=float(row.band_low),
        band_high=float(row.band_high),
        conviction_score=row.conviction_score,
        dominant_thesis=row.dominant_thesis,
        outlier_persona=row.outlier_persona,
        outlier_pt=float(row.outlier_pt) if row.outlier_pt is not None else None,
    )


# ── GET /api/personas ────────────────────────────────────────────────────────

@router.get("/api/personas")
def get_personas() -> list[dict]:
    return load_personas()


# ── GET /api/health ──────────────────────────────────────────────────────────

@router.get("/api/health")
def health(session: Session = Depends(get_session)) -> dict:
    try:
        last_run = session.execute(select(func.max(ConsensusPTRow.computed_at))).scalar_one()
        db_status = "ok"
    except Exception:  # noqa: BLE001 - health check must not raise
        logger.exception("health check DB query failed")
        last_run = None
        db_status = "unreachable"

    return {
        "status": "ok" if db_status == "ok" else "degraded",
        "database": db_status,
        "last_job_run": last_run.isoformat() if last_run else None,
    }
