"""FastAPI application entrypoint."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.routes import load_personas, router

logger = logging.getLogger(__name__)
WEIGHT_SUM_TOLERANCE = 1e-6


def validate_personas() -> None:
    """Fail fast when a persona YAML has invalid factor weights."""
    configs = load_personas()
    if not configs:
        raise RuntimeError("no persona YAML configurations found")

    for config in configs:
        name = config.get("name", "<unnamed>")
        weights = config.get("weights") or {}
        if not isinstance(weights, dict) or not weights:
            raise RuntimeError(f"persona {name!r} must define non-empty weights")
        total = sum(float(weight) for weight in weights.values())
        if abs(total - 1.0) > WEIGHT_SUM_TOLERANCE:
            raise RuntimeError(
                f"persona {name!r} weights sum to {total}, expected 1.0"
            )
        logger.info("persona %s: weights sum to %.4f (ok)", name, total)


@asynccontextmanager
async def lifespan(_app: FastAPI):
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
