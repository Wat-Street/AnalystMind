"""Initial schema (applied via db/init.sql on first container start)

Revision ID: 0001
Revises:
Create Date: 2026-05-25

This migration is a no-op. The initial schema — all 7 tables, indexes,
and the ohlcv TimescaleDB hypertable — is created by db/init.sql when
the Postgres container first starts.

This file exists so Alembic's version table is in sync with reality.
After running `docker compose up` for the first time, stamp the DB:

    docker compose exec backend alembic stamp 0001

All future schema changes must be Alembic migrations (see README).
"""
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Schema already created by db/init.sql. Nothing to do.
    pass


def downgrade() -> None:
    # To fully reset: docker compose down -v && docker compose up --build
    pass
