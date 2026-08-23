"""FastAPI routes for analysis, cached results, and persona configuration."""

from __future__ import annotations

import logging

import yaml
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..ml.inference import PERSONAS_DIR, PersonaInference
from ..schemas.persona import ConsensusPT

logger = logging.getLogger(__name__)
router = APIRouter()


def get_session():
    """Yield one database session per request."""
    from ..models.db import engine

    with Session(engine) as session:
        yield session


_inference: PersonaInference | None = None


def get_inference() -> PersonaInference:
    """Lazily load and reuse the checkpoint-backed inference service."""
    global _inference
    if _inference is None:
        try:
            _inference = PersonaInference()
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
    return _inference


def load_personas() -> list[dict]:
    """Load persona YAML files in a stable name order for the API."""
    configs: list[dict] = []
    for path in sorted(PERSONAS_DIR.glob("*.yaml")):
        config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(config, dict):
            raise RuntimeError(f"persona config {path} must contain a YAML mapping")
        configs.append(config)
    return configs


@router.post("/analyze", response_model=ConsensusPT)
def analyze(
    ticker: str,
    session: Session = Depends(get_session),
    inference: PersonaInference = Depends(get_inference),
) -> ConsensusPT:
    """Run all personas, persist their views, and return the consensus."""
    ticker = ticker.strip().upper()
    if not ticker:
        raise HTTPException(status_code=400, detail="ticker is required")

    try:
        consensus = inference.run_consensus(ticker, session)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    from ..models.db import ConsensusPT as ConsensusPTRow
    from ..models.db import PersonaOutput as PersonaOutputRow

    # run_consensus stores the individual outputs on this request's session so
    # they can be persisted without running the nine model passes twice.
    get_outputs = getattr(inference, "get_last_persona_outputs", None)
    outputs = get_outputs(session, ticker) if get_outputs is not None else []
    try:
        for output in outputs:
            session.add(PersonaOutputRow(**output.model_dump()))
        session.add(ConsensusPTRow(**consensus.model_dump()))
        session.commit()
    except Exception as exc:  # noqa: BLE001 - convert DB failures to API errors
        session.rollback()
        logger.exception("failed to persist analysis for %s", ticker)
        raise HTTPException(status_code=500, detail="could not persist analysis") from exc

    return consensus


@router.get("/api/stock/{ticker}", response_model=ConsensusPT)
def get_stock(ticker: str, session: Session = Depends(get_session)) -> ConsensusPT:
    """Return the latest cached consensus for a ticker."""
    from ..models.db import ConsensusPT as ConsensusPTRow

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

    return ConsensusPT.model_validate(row)


@router.get("/api/personas")
def get_personas() -> list[dict]:
    return load_personas()


@router.get("/api/health")
def health(session: Session = Depends(get_session)) -> dict:
    """Report API and database reachability without raising DB exceptions."""
    from ..models.db import ConsensusPT as ConsensusPTRow

    try:
        last_run = session.execute(select(func.max(ConsensusPTRow.computed_at))).scalar_one()
        database = "ok"
    except Exception:  # noqa: BLE001 - health checks should return degraded status
        logger.exception("health check DB query failed")
        last_run = None
        database = "unreachable"

    return {
        "status": "ok" if database == "ok" else "degraded",
        "database": database,
        "last_job_run": last_run.isoformat() if last_run else None,
    }
