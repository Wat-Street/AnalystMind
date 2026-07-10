# SHARED CONSTANTS — Kiana + Ken must agree on this before writing dataset.py or tokenizer.py.
# Do not reorder FEATURE_ORDER without updating MODALITY_GROUPS indices.

FEATURE_ORDER = [
    # fundamentals (valuation)
    "fcf_yield", "trailing_pe", "forward_pe", "ev_ebitda",

    #  fundamentals (quality)
    "revenue_cagr_3y", "gross_margin", "net_debt_to_ebita", # to do: add to fundamentals.py

    # technical
    "rsi_signal", "macd_signal", "breakout_score", "volume_surge",

    # momentum
    "return_21d", "return_63d", "return_126d", "market_relative_strength_63d",

    #market risk / liquidity
    "realized_volatility_21d", "beta_126d", "max_drawdown_63d", "dollar_volume_20d_log",

    # macro - pending the FRED ingestion (macro.py still empty)
    "fed_funds_rate", "fed_funds_change_3m", "cpi_yoy", "cpi_trend_3m", "unemployment", "unemployment_change_3m", 
    "treasury_10y", "real_treasury_10y", "yield_spread",

    # options 
    "put_call_ratio", "unusual_options_score", "gamma_exposure",

    # insider 
    "insider_buy_score", "net_insider_delta",

    # sentiment 
    "mgmt_sentiment_score", "qa_sentiment_score", "mgmt_qa_sentiment_gap", "transcript_sentiment_change_qoq",

    #analyst / sells-side
    "price_target_upside", "price_target_dispersion", "recommendation_score", "recent_rating_delta", "coverage_volume",
]

MODALITY_FEATURES: dict[str, list[int]] = {
    "valuation": ["fcf_yield", "trailing_pe", "forward_pe", "ev_ebitda"],
    "quality": ["revenue_cagr_3y", "gross_margin", "net_debt_to_ebita"],
    "technical": ["rsi_signal", "macd_signal", "breakout_score", "volume_surge"],
    "momentum": [ "return_21d", "return_63d", "return_126d", "market_relative_strength_63d"],
    "market_risk": ["realized_volatility_21d", "beta_126d", "max_drawdown_63d", "dollar_volume_20d_log"],
    "macro": ["fed_funds_rate", "fed_funds_change_3m", "cpi_yoy", "cpi_trend_3m", "unemployment", "unemployment_change_3m", 
    "treasury_10y", "real_treasury_10y", "yield_spread"],
    "options": [ "put_call_ratio", "unusual_options_score", "gamma_exposure",],
    "insider": ["insider_buy_score", "net_insider_delta"],
    "transcript": ["mgmt_sentiment_score", "qa_sentiment_score", "mgmt_qa_sentiment_gap", "transcript_sentiment_change_qoq"],
    "analyst": ["price_target_upside", "price_target_dispersion", "recommendation_score", "recent_rating_delta", "coverage_volume"]

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
    "macro_topdown":         ["macro", "valuation", "momentum"],
    "options_flow_trader":   ["options", "technical", "momentum"],
    "sentiment_trader":      ["transcript", "analyst", "options", "momentum"],
    "insider_institutional": ["insider", "valuation", "quality", "transcript"],
    "dividend_income":       ["valuation", "quality", "macro", "transcript"],
}

N_FEATURES = len(FEATURE_ORDER)
HORIZONS = [10, 21, 63, 126, 252]  # trading days: 2W, 1M, 3M, 6M, 12M

#TO DO - update personal yamls