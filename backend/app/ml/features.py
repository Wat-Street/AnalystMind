# SHARED CONSTANTS — Kiana + Ken must agree on this before writing dataset.py or tokenizer.py.
# Do not reorder FEATURE_ORDER without updating MODALITY_GROUPS indices.

FEATURE_ORDER = [
    # fundamentals (valuation)
    "fcf_yield", "trailing_pe", "forward_pe", "ev_ebitda",
    # fundamentals (quality/growth)
    "revenue_cagr_3y", "gross_margin",
    "net_debt_to_ebitda",       # net_debt / ebitda — need to add ebitda to fundamentals.py (yfinance info["ebitda"])
    # technical indicators
    "rsi_signal", "macd_signal", "breakout_score", "volume_surge",
    # momentum — derived from ohlcv in dataset.py
    "return_21d",                       # (close[t] - close[t-21]) / close[t-21]
    "return_63d",                       # same, 63 days
    "return_126d",                      # same, 126 days
    "market_relative_strength_63d",     # ticker 63d return minus SPY 63d return — need SPY in ohlcv
    # market risk / liquidity — derived from ohlcv in dataset.py
    "realized_volatility_21d",          # std dev of daily log returns over 21d, annualised (* sqrt(252))
    "beta_126d",                        # OLS slope of ticker daily returns on SPY over 126d
    "max_drawdown_63d",                 # max peak-to-trough / peak over 63d
    "dollar_volume_20d_log",            # log(mean(close * volume) over 20d) — liquidity proxy
    # macro — macro.py (FRED, pending); change/trend features derived in dataset.py
    "fed_funds_rate",
    "fed_funds_change_3m",              # fed_funds_rate[t] - fed_funds_rate[t-63 trading days]
    "cpi_yoy",
    "cpi_trend_3m",                     # cpi_yoy[t] - cpi_yoy[t-3M] — accelerating vs decelerating
    "unemployment",
    "unemployment_change_3m",           # unemployment[t] - unemployment[t-3M]
    "treasury_10y",
    "real_treasury_10y",                # treasury_10y - cpi_yoy — true cost of capital
    "yield_spread",
    # options
    "put_call_ratio", "unusual_options_score", "gamma_exposure",
    # insider
    "insider_buy_score", "net_insider_delta",
    # transcript sentiment — transcripts.py → FinBERT; gap/qoq derived in dataset.py
    "mgmt_sentiment_score", "qa_sentiment_score",
    "mgmt_qa_sentiment_gap",                # mgmt_score - qa_score — evasion signal (exec sounds great on slides but gets defensive in Q&A)
    "transcript_sentiment_change_qoq",      # mgmt_sentiment_score[this Q] - mgmt_sentiment_score[last Q]
    # analyst / sell-side — analyst_ratings.py (Afreed, pending)
    "price_target_upside", "price_target_dispersion", "recommendation_score", "recent_rating_delta", "coverage_volume",
]

MODALITY_FEATURES: dict[str, list[str]] = {
    "valuation":    ["fcf_yield", "trailing_pe", "forward_pe", "ev_ebitda"],
    "quality":      ["revenue_cagr_3y", "gross_margin", "net_debt_to_ebitda"],
    "technical":    ["rsi_signal", "macd_signal", "breakout_score", "volume_surge"],
    "momentum":     ["return_21d", "return_63d", "return_126d", "market_relative_strength_63d"],
    "market_risk":  ["realized_volatility_21d", "beta_126d", "max_drawdown_63d", "dollar_volume_20d_log"],
    "macro":        ["fed_funds_rate", "fed_funds_change_3m", "cpi_yoy", "cpi_trend_3m",
                     "unemployment", "unemployment_change_3m", "treasury_10y", "real_treasury_10y", "yield_spread"],
    "options":      ["put_call_ratio", "unusual_options_score", "gamma_exposure"],
    "insider":      ["insider_buy_score", "net_insider_delta"],
    "transcript":   ["mgmt_sentiment_score", "qa_sentiment_score",
                     "mgmt_qa_sentiment_gap", "transcript_sentiment_change_qoq"],
    "analyst":      ["price_target_upside", "price_target_dispersion",
                     "recommendation_score", "recent_rating_delta", "coverage_volume"],
}

_FEATURE_INDEX: dict[str, int] = {name: i for i, name in enumerate(FEATURE_ORDER)}
MODALITY_GROUPS: dict[str, list[int]] = {
    modality: [_FEATURE_INDEX[name] for name in names]
    for modality, names in MODALITY_FEATURES.items()
}

PERSONA_MODALITIES: dict[str, list[str]] = {
    "value_fundamentalist":  ["valuation", "quality", "insider", "transcript"],
    "growth_visionary":      ["valuation", "quality", "analyst", "transcript"],
    "quant_momentum":        ["momentum", "market_risk", "options"],
    "technical_analyst":     ["technical", "momentum", "market_risk"],
    "macro_topdown":         ["macro", "valuation", "market_risk"],
    "options_flow_trader":   ["options", "technical", "momentum"],
    "sentiment_trader":      ["transcript", "analyst", "options", "momentum"],
    "insider_institutional": ["insider", "valuation", "quality", "transcript"],
    "dividend_income":       ["valuation", "quality", "macro", "transcript"],
}

N_FEATURES = len(FEATURE_ORDER)
HORIZONS = [10, 21, 63, 126, 252]  # trading days: 2W, 1M, 3M, 6M, 12M


# Features intentionally excluded from FEATURE_ORDER until their ingestors exist.
# Do not add all-null columns to training — add to FEATURE_ORDER only when the
# corresponding ingestor is merged and backfilled.
# INGESTION_GAPS: dict[str, tuple[str, ...]] = {
#    "dividends":         ("dividend_yield", "fcf_payout_ratio", "dividend_cagr", "net_shareholder_yield"),
#    "earnings":          ("eps_surprise", "revenue_surprise", "guidance_surprise", "estimate_revision_breadth"),
#    "sector_etf":        ("sector_relative_strength", "sector_breadth", "sector_rotation_score"),
#    "institutional_13f": ("institutional_flow", "institutional_ownership_pct", "institutional_crowding"),
#    "news":              ("news_event_sentiment", "news_volume_change", "news_novelty"),
#}
