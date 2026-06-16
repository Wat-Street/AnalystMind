import os
from sqlalchemy import (
    Column, String, Float, Integer, BigInteger, Numeric, Text, Date,
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


class AnalystRating(Base):
    __tablename__ = "analyst_ratings"

    id                   = Column(UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()"))
    ticker               = Column(String(10), nullable=False, index=True)
    as_of_date           = Column(Date, nullable=False)
    current_price        = Column(Numeric)
    consensus_pt         = Column(Numeric)
    target_low           = Column(Numeric)
    target_high          = Column(Numeric)
    target_median        = Column(Numeric)
    price_target_upside  = Column(Float)
    recommendation_score = Column(Float)
    coverage_volume      = Column(Float)
    recent_rating_delta  = Column(Float)
    strong_buy           = Column(Integer, nullable=False, server_default=text("0"))
    buy                  = Column(Integer, nullable=False, server_default=text("0"))
    hold                 = Column(Integer, nullable=False, server_default=text("0"))
    sell                 = Column(Integer, nullable=False, server_default=text("0"))
    strong_sell          = Column(Integer, nullable=False, server_default=text("0"))
    fetched_at           = Column(DateTime(timezone=True), nullable=False, server_default=text("NOW()"))

    __table_args__ = (
        UniqueConstraint("ticker", "as_of_date", name="uq_analyst_ratings_ticker_date"),
        CheckConstraint(
            "recommendation_score IS NULL OR (recommendation_score >= 0 AND recommendation_score <= 1)",
            name="ck_analyst_recommendation_score_range",
        ),
        CheckConstraint(
            "coverage_volume IS NULL OR (coverage_volume >= 0 AND coverage_volume <= 1)",
            name="ck_analyst_coverage_volume_range",
        ),
        CheckConstraint(
            "recent_rating_delta IS NULL OR (recent_rating_delta >= 0 AND recent_rating_delta <= 1)",
            name="ck_analyst_recent_delta_range",
        ),
    )


class OptionsFlow(Base):
    __tablename__ = "options_flow"

    id                    = Column(UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()"))
    ticker                = Column(String(10), nullable=False, index=True)
    as_of_date            = Column(Date, nullable=False)
    expiration_date       = Column(Date, nullable=False)
    underlying_price      = Column(Numeric)
    call_volume           = Column(BigInteger, nullable=False, server_default=text("0"))
    put_volume            = Column(BigInteger, nullable=False, server_default=text("0"))
    call_open_interest    = Column(BigInteger, nullable=False, server_default=text("0"))
    put_open_interest     = Column(BigInteger, nullable=False, server_default=text("0"))
    put_call_ratio        = Column(Float)
    unusual_options_score = Column(Float, nullable=False, server_default=text("0"))
    gamma_exposure        = Column(Numeric)
    fetched_at            = Column(DateTime(timezone=True), nullable=False, server_default=text("NOW()"))

    __table_args__ = (
        UniqueConstraint("ticker", "as_of_date", "expiration_date", name="uq_options_flow_ticker_date_exp"),
        CheckConstraint("put_call_ratio IS NULL OR put_call_ratio >= 0", name="ck_options_put_call_ratio_nonnegative"),
        CheckConstraint(
            "unusual_options_score >= 0 AND unusual_options_score <= 1",
            name="ck_options_unusual_score_range",
        ),
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
