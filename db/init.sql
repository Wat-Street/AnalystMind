-- Run once on first container start (mounted to /docker-entrypoint-initdb.d/).
-- Creates the TimescaleDB extension, all tables, the ohlcv hypertable, and indexes.

CREATE EXTENSION IF NOT EXISTS timescaledb;

-- ── transcripts ──────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS transcripts (
    id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    ticker      VARCHAR(10) NOT NULL,
    quarter     VARCHAR(10) NOT NULL,
    source      VARCHAR(50) NOT NULL,
    segments    JSONB       NOT NULL DEFAULT '[]',
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_transcripts_ticker         ON transcripts(ticker);
CREATE UNIQUE INDEX IF NOT EXISTS idx_transcripts_ticker_quarter ON transcripts(ticker, quarter);

-- ── sentiment_scores ─────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS sentiment_scores (
    id            UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    transcript_id UUID        NOT NULL REFERENCES transcripts(id) ON DELETE CASCADE,
    ticker        VARCHAR(10) NOT NULL,
    segment_role  VARCHAR(20) NOT NULL CHECK (segment_role IN ('management', 'qa')),
    label         VARCHAR(20) NOT NULL CHECK (label IN ('positive', 'negative', 'neutral')),
    score         FLOAT       NOT NULL CHECK (score >= 0 AND score <= 1),
    scored_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_sentiment_ticker     ON sentiment_scores(ticker);
CREATE INDEX IF NOT EXISTS idx_sentiment_transcript ON sentiment_scores(transcript_id);

-- ── ohlcv (TimescaleDB hypertable) ───────────────────────────────────────────
CREATE TABLE IF NOT EXISTS ohlcv (
    ticker    VARCHAR(10) NOT NULL,
    time      TIMESTAMPTZ NOT NULL,
    open      NUMERIC,
    high      NUMERIC,
    low       NUMERIC,
    close     NUMERIC,
    volume    BIGINT,
    adj_close NUMERIC
);

SELECT create_hypertable('ohlcv', 'time', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_ohlcv_ticker_time ON ohlcv(ticker, time DESC);

-- ── fundamentals ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS fundamentals (
    id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    ticker          VARCHAR(10) NOT NULL,
    period          VARCHAR(10) NOT NULL,
    fcf_yield       FLOAT,
    trailing_pe     FLOAT,
    forward_pe      FLOAT,
    ev_ebitda       FLOAT,
    revenue_cagr_3y FLOAT,
    gross_margin    FLOAT,
    net_debt        NUMERIC,
    fetched_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_fundamentals_ticker        ON fundamentals(ticker);
CREATE UNIQUE INDEX IF NOT EXISTS idx_fundamentals_ticker_period ON fundamentals(ticker, period);

-- ── macro_series ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS macro_series (
    id               UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    series_id        VARCHAR(50) NOT NULL,
    value            FLOAT       NOT NULL,
    observation_date DATE        NOT NULL,
    fetched_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_macro_series_id_date ON macro_series(series_id, observation_date);

-- ── persona_outputs ──────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS persona_outputs (
    id           UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    ticker       VARCHAR(10) NOT NULL,
    persona_name VARCHAR(50) NOT NULL,
    pt_base      NUMERIC,
    pt_bull      NUMERIC,
    pt_bear      NUMERIC,
    confidence   FLOAT       CHECK (confidence >= 0 AND confidence <= 1),
    rationale    TEXT,
    computed_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_persona_outputs_ticker         ON persona_outputs(ticker);
CREATE INDEX IF NOT EXISTS idx_persona_outputs_ticker_persona ON persona_outputs(ticker, persona_name);

-- ── consensus_pt ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS consensus_pt (
    id               UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    ticker           VARCHAR(10) NOT NULL,
    consensus_pt     NUMERIC,
    band_low         NUMERIC,
    band_high        NUMERIC,
    conviction_score FLOAT       CHECK (conviction_score >= 0 AND conviction_score <= 1),
    dominant_thesis  TEXT,
    outlier_persona  VARCHAR(50),
    outlier_pt       NUMERIC,
    computed_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_consensus_pt_ticker      ON consensus_pt(ticker);
CREATE INDEX IF NOT EXISTS idx_consensus_pt_computed_at ON consensus_pt(ticker, computed_at DESC);
