"""FastAPI app entrypoint.

Validates every personas/*.yaml config at startup — weights must sum to 1.0,
per CLAUDE.md's "Persona YAML weights must sum to 1.0" rule — before the app
starts serving requests.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.routes import load_personas, router

logger = logging.getLogger(__name__)

WEIGHT_SUM_TOLERANCE = 1e-6


def validate_personas() -> None:
    for config in load_personas():
        name = config.get("name", "<unnamed>")
        weights = config.get("weights") or {}
        total = sum(weights.values())
        if abs(total - 1.0) > WEIGHT_SUM_TOLERANCE:
            raise RuntimeError(
                f"persona {name!r} weights sum to {total}, expected 1.0 — fix its YAML config"
            )
        logger.info("persona %s: weights sum to %.4f (ok)", name, total)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO)
    validate_personas()
    yield


app = FastAPI(title="AnalystMind", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
