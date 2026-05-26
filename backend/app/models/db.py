import os
from sqlalchemy import (
    Column, String, Float, BigInteger, Numeric, Text, Date,
    DateTime, ForeignKey, CheckConstraint, UniqueConstraint, Index,
    create_engine, text,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import declarative_base, relationship

DATABASE_URL = os.environ["DATABASE_URL"]

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
Base = declarative_base()


class Transcript(Base):
    __tablename__ = "transcripts"

    id          = Column(UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()"))
    ticker      = Column(String(10), nullable=False, index=True)
    quarter     = Column(String(10), nullable=False)
    source      = Column(String(50), nullable=False)
    segments    = Column(JSONB, nullable=False, server_default=text("'[]'::jsonb"))
    ingested_at = Column(DateTime(timezone=True), nullable=False, server_default=text("NOW()"))

    sentiment_scores = relationship("SentimentScore", back_populates="transcript", cascade="all, delete-orphan")

    __table_args__ = (
        UniqueConstraint("ticker", "quarter", name="uq_transcripts_ticker_quarter"),
    )


class SentimentScore(Base):
    __tablename__ = "sentiment_scores"

    id            = Column(UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()"))
    transcript_id = Column(UUID(as_uuid=True), ForeignKey("transcripts.id", ondelete="CASCADE"), nullable=False)
    ticker        = Column(String(10), nullable=False, index=True)
    segment_role  = Column(String(20), nullable=False)
    label         = Column(String(20), nullable=False)
    score         = Column(Float, nullable=False)
    scored_at     = Column(DateTime(timezone=True), nullable=False, server_default=text("NOW()"))

    transcript = relationship("Transcript", back_populates="sentiment_scores")

    __table_args__ = (
        CheckConstraint("segment_role IN ('management', 'qa')", name="ck_sentiment_role"),
        CheckConstraint("label IN ('positive', 'negative', 'neutral')",  name="ck_sentiment_label"),
        CheckConstraint("score >= 0 AND score <= 1", name="ck_sentiment_score_range"),
    )


class OHLCV(Base):
    # TimescaleDB hypertable — partitioned on `time`.
    # Hypertable conversion is handled in db/init.sql, not here.
    __tablename__ = "ohlcv"

    ticker    = Column(String(10), primary_key=True)
    time      = Column(DateTime(timezone=True), primary_key=True)
    open      = Column(Numeric)
    high      = Column(Numeric)
    low       = Column(Numeric)
    close     = Column(Numeric)
    volume    = Column(BigInteger)
    adj_close = Column(Numeric)


class Fundamentals(Base):
    __tablename__ = "fundamentals"

    id              = Column(UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()"))
    ticker          = Column(String(10), nullable=False, index=True)
    period          = Column(String(10), nullable=False)
    fcf_yield       = Column(Float)
    trailing_pe     = Column(Float)
    forward_pe      = Column(Float)
    ev_ebitda       = Column(Float)
    revenue_cagr_3y = Column(Float)
    gross_margin    = Column(Float)
    net_debt        = Column(Numeric)
    fetched_at      = Column(DateTime(timezone=True), nullable=False, server_default=text("NOW()"))

    __table_args__ = (
        UniqueConstraint("ticker", "period", name="uq_fundamentals_ticker_period"),
    )


class MacroSeries(Base):
    __tablename__ = "macro_series"

    id               = Column(UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()"))
    series_id        = Column(String(50), nullable=False)
    value            = Column(Float, nullable=False)
    observation_date = Column(Date, nullable=False)
    fetched_at       = Column(DateTime(timezone=True), nullable=False, server_default=text("NOW()"))

    __table_args__ = (
        UniqueConstraint("series_id", "observation_date", name="uq_macro_series_id_date"),
    )


class PersonaOutput(Base):
    __tablename__ = "persona_outputs"

    id           = Column(UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()"))
    ticker       = Column(String(10), nullable=False, index=True)
    persona_name = Column(String(50), nullable=False)
    pt_base      = Column(Numeric)
    pt_bull      = Column(Numeric)
    pt_bear      = Column(Numeric)
    confidence   = Column(Float)
    rationale    = Column(Text)
    computed_at  = Column(DateTime(timezone=True), nullable=False, server_default=text("NOW()"))

    __table_args__ = (
        CheckConstraint("confidence >= 0 AND confidence <= 1", name="ck_persona_confidence_range"),
        Index("idx_persona_outputs_ticker_persona", "ticker", "persona_name"),
    )


class ConsensusPT(Base):
    __tablename__ = "consensus_pt"

    id               = Column(UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()"))
    ticker           = Column(String(10), nullable=False, index=True)
    consensus_pt     = Column(Numeric)
    band_low         = Column(Numeric)
    band_high        = Column(Numeric)
    conviction_score = Column(Float)
    dominant_thesis  = Column(Text)
    outlier_persona  = Column(String(50))
    outlier_pt       = Column(Numeric)
    computed_at      = Column(DateTime(timezone=True), nullable=False, server_default=text("NOW()"))

    __table_args__ = (
        CheckConstraint("conviction_score >= 0 AND conviction_score <= 1", name="ck_consensus_conviction_range"),
        Index("idx_consensus_pt_computed_at", "ticker", "computed_at"),
    )
