# AnalystMind

Multi-agent LLM system that simulates how different classes of institutional investors analyze a stock and arrive at a price target. A swarm of analyst personas each applies its own weighting philosophy to a shared data lake, then a consensus desk aggregates their outputs into a single defensible price target with a confidence score and dispersion band.

---

## How it works

1. A `POST /analyze?ticker=AAPL` request triggers the orchestrator.
2. All 9 (to begin with) persona agents run in parallel. Each loads its YAML config, fetches only the data sources it needs, and produces a price target + confidence score.
3. The consensus desk computes a dispersion-weighted mean, flags outlier personas (> 1.5σ from mean), re-weights them at 30%, and fires a single LLM call to generate the `dominant_thesis`.
4. The result is written to the `consensus_pt` table and returned.

---

## Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| API | FastAPI |
| Database | PostgreSQL 15 + TimescaleDB |
| ORM | SQLAlchemy 2.0 |
| Agent framework | LangChain |
| Sentiment model | FinBERT (HuggingFace, CPU local) |
| Financials | Financial Modeling Prep (FMP) API |
| Price / options | yfinance + Polygon.io |
| Macro data | FRED API |
| Containerization | Docker + Docker Compose |
| Frontend | React + Next.js + Tailwind (Week 4+) |

---

## Repo layout

```
AnalystMind/
├── docker-compose.yml
├── personas/                        # YAML config per analyst persona
│   ├── value_fundamentalist.yaml
│   ├── growth_visionary.yaml
│   ├── quant_momentum.yaml
│   ├── macro_topdown.yaml
│   ├── sentiment_trader.yaml
│   ├── technical_analyst.yaml
│   ├── dividend_income.yaml
│   ├── options_flow_trader.yaml
│   └── insider_institutional.yaml
├── db/
│   ├── init.sql                     # DB schema + TimescaleDB hypertable setup
│   └── migrations/                  # Alembic migration files
└── backend/
    ├── Dockerfile
    ├── pyproject.toml              # project metadata + dependencies (uv)
    ├── uv.lock                     # pinned dep graph — committed, reproducible builds
    ├── .python-version             # pinned to 3.11
    └── app/
        ├── main.py                  # FastAPI entrypoint
        ├── api/
        │   └── routes.py
        ├── agents/
        │   ├── orchestrator.py      # Consensus desk
        │   └── persona_agent.py     # Per-persona scoring + PT computation
        ├── ingestion/               # One file per data source (all stubs)
        │   ├── fundamentals.py      # FMP — P/E, FCF, EV, revenue
        │   ├── ohlcv.py             # yfinance / Polygon — price + volume
        │   ├── macro.py             # FRED — rates, CPI, unemployment
        │   ├── news.py              # Finnhub — news feed
        │   ├── transcripts.py       # EDGAR — earnings call text
        │   ├── earnings.py          # FMP — EPS estimates + surprises
        │   ├── analyst_ratings.py   # FMP — price targets + upgrades
        │   ├── dividends.py         # FMP — dividend history
        │   ├── options_flow.py      # Polygon — options chain + GEX
        │   ├── insider_activity.py  # EDGAR Form 4 + FMP 13F
        │   ├── sector_etf.py        # yfinance — GICS sector ETFs
        │   ├── short_interest.py    # yfinance + FINRA REGSHO
        │   └── technical_indicators.py  # Computed from OHLCV (no API)
        ├── models/
        │   └── db.py                # SQLAlchemy table models
        └── schemas/
            └── persona.py           # Pydantic: PersonaOutput, ConsensusPT
```

---

## Local setup

**Prerequisites:** Docker Desktop running.

```bash
git clone <repo-url>
cd AnalystMind
cp .env.example .env        # fill in API keys (see Environment variables below)
docker compose up --build
```

### Backend development (outside Docker)

Dependencies are managed with [`uv`](https://docs.astral.sh/uv/). Install uv first
(`curl -LsSf https://astral.sh/uv/install.sh | sh`), then:

```bash
cd backend
uv sync                          # create .venv, install prod + dev deps from uv.lock
uv run pytest                    # run the test suite
uv run uvicorn app.main:app      # run the API locally (DB still via docker compose)
uv add <pkg>                     # add a prod dep — updates pyproject.toml + uv.lock
uv add --dev <pkg>               # add a dev-only dep
uv lock --upgrade                # bump all deps within declared ranges
uv lock --check                  # verify lockfile matches pyproject.toml (CI gate)
```

The lockfile (`backend/uv.lock`) is committed and is the source of truth for builds.
The Python version is pinned via `backend/.python-version`; `uv` provisions it
automatically.

The database connection is configured in `docker-compose.yml` and injected directly into the backend container — no `DATABASE_URL` in `.env` needed. The schema is created automatically on first start via `db/init.sql`.

| Service | URL |
|---|---|
| Backend API | http://localhost:8000 |
| API docs | http://localhost:8000/docs |
| PostgreSQL | localhost:5432 (user: `analyst`, db: `analystmind`) |

To bring up the frontend as well:
```bash
docker compose --profile frontend up --build
```

---

## API

| Method | Path | Description |
|---|---|---|
| `POST` | `/analyze?ticker=AAPL` | Run all personas + consensus desk. Triggers real computation. |
| `GET` | `/api/stock/:ticker` | Return latest cached `ConsensusPT` from DB. Read-only. |
| `GET` | `/api/personas` | List loaded persona configs with weight vectors. |
| `GET` | `/api/health` | Service status and last job run timestamp. |

**Rule: GET endpoints never trigger computation.** Only `POST /analyze` runs persona agents.

---

## Personas

Each persona is a YAML file in `/personas/`. The orchestrator loads all files at startup — adding a new persona requires only a new YAML file, zero code changes.

| Persona | Key weights | Data sources | Horizon |
|---|---|---|---|
| Value Fundamentalist | FCF 40%, P/E 30%, moat 30% | fundamentals, transcripts | 3Y |
| Growth Visionary | Fwd P/E 45%, TAM 35%, rev CAGR 20% | fundamentals, transcripts, earnings | 5Y |
| Quant Momentum | Momentum 50%, vol regime 30%, beta 20% | ohlcv, options_flow | 1M |
| Macro Top-Down | Macro 50%, sector rotation 30%, rates 20% | macro, fundamentals, sector_etf | 6M |
| Sentiment Trader | Tone 50%, coverage vol 30%, news 20% | transcripts, news, analyst_ratings | 1M |
| Technical Analyst | RSI 30%, MACD 30%, breakout 25%, volume 15% | ohlcv | 3M |
| Dividend & Income | Yield 35%, payout 30%, FCF coverage 25%, CAGR 10% | fundamentals, dividends | 3Y |
| Options Flow Trader | Unusual flow 40%, P/C ratio 30%, GEX 30% | options_flow, ohlcv | 2W |
| Insider & Institutional | Insider buys 40%, inst. flow 35%, delta 25% | insider_activity, fundamentals | 6M |


**YAML rules:**
- Primary weights must sum to exactly `1.0`
- `sentiment_delta` is a top-level additive modifier, not counted in the weight sum
- Adding a persona = drop a new YAML file, restart the backend

---

## Database schema

| Table | Purpose |
|---|---|
| `transcripts` | Earnings call text + metadata (EDGAR) |
| `sentiment_scores` | FinBERT output per transcript segment |
| `ohlcv` | OHLCV price series — TimescaleDB hypertable, partitioned on `time` |
| `fundamentals` | P/E, FCF yield, EV/EBITDA, revenue CAGR, gross margin |
| `macro_series` | FRED series values (CPI, FEDFUNDS, UNRATE, etc.) |
| `persona_outputs` | Individual persona PT (base/bull/bear) + confidence + rationale |
| `consensus_pt` | Final aggregated PT, dispersion band, conviction score, dominant thesis |

Schema is defined in `db/init.sql` and mirrored as SQLAlchemy models in `backend/app/models/db.py`.

---

## Schema migrations

The initial schema is created by `db/init.sql` on first container start. All subsequent schema changes — new columns, new tables, dropped columns — must go through Alembic so every team member's DB stays in sync.

### First-time setup (after `docker compose up`)

Stamp the DB to tell Alembic the initial schema is already applied:

```bash
docker compose exec backend alembic stamp 0001
```

### Adding a new column (the usual case)

**Step 1 — Update the SQLAlchemy model** in `backend/app/models/db.py`:

```python
# Example: adding `moat_score` to the fundamentals table
class Fundamentals(Base):
    ...
    moat_score = Column(Float)   # ← add this line
```

**Step 2 — Generate the migration** (Alembic diffs the model against the live DB):

```bash
docker compose exec backend alembic revision --autogenerate -m "add moat_score to fundamentals"
```

This creates a file in `db/migrations/versions/`. Open it and verify the `upgrade()` and `downgrade()` functions look right before continuing.

**Step 3 — Apply it:**

```bash
docker compose exec backend alembic upgrade head
```

**Step 4 — Commit both files** — the model change and the generated migration file.

---

### Writing a migration manually

When `--autogenerate` can't detect the change (e.g. adding a TimescaleDB index, changing a constraint), write it by hand:

```bash
docker compose exec backend alembic revision -m "add composite index on persona_outputs"
```

Edit the generated file:

```python
def upgrade() -> None:
    op.create_index(
        "idx_persona_outputs_ticker_computed",
        "persona_outputs",
        ["ticker", "computed_at"],
    )

def downgrade() -> None:
    op.drop_index("idx_persona_outputs_ticker_computed", table_name="persona_outputs")
```

Then apply with `alembic upgrade head`.

---

### Other useful commands

```bash
# See current migration version applied to the DB
docker compose exec backend alembic current

# See full migration history
docker compose exec backend alembic history --verbose

# Roll back one migration
docker compose exec backend alembic downgrade -1

# Roll back to a specific revision
docker compose exec backend alembic downgrade 0001
```

### Rules

- **Never edit `db/init.sql` after the first deployment.** It only runs on a fresh DB. Changes there have no effect on existing containers.
- **Every schema change = one Alembic migration file.** No raw `ALTER TABLE` in Postgres directly.
- **Always implement `downgrade()`** so rollbacks are possible.
- Migration files live in `db/migrations/versions/` and must be committed alongside the model change.

---

## Consensus aggregation

1. Compute confidence-weighted mean across all persona price targets.
2. Compute std dev; flag outliers > 1.5σ from mean.
3. Re-weight outliers at 30% and recompute final consensus PT.
4. `conviction_score = 1 - (sigma / consensus_final)`, capped to [0, 1].
5. Personas within 0.5σ of the final PT form the "aligned cluster" — a single LLM call generates `dominant_thesis` from their names and top weight factors.

---

## Week 1 ticker set

AAPL · MSFT · NVDA · TSLA · AMZN
