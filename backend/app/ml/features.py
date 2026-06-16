# SHARED CONSTANTS — Kiana + Ken must agree on this before writing dataset.py or tokenizer.py.
# Do not reorder FEATURE_ORDER without updating MODALITY_GROUPS indices.

FEATURE_ORDER = [
    # fundamentals (indices 0–6)
    "fcf_yield", "trailing_pe", "forward_pe", "ev_ebitda",
    "revenue_cagr_3y", "gross_margin", "net_debt",
    # technical (indices 7–12)
    "rsi_signal", "macd_signal", "breakout_score", "volume_surge",
    "close", "volume",
    # macro (indices 13–17)
    "fed_funds_rate", "cpi_yoy", "unemployment", "treasury_10y", "yield_spread",
    # options (indices 18–20)
    "put_call_ratio", "unusual_options_score", "gamma_exposure",
    # sentiment (indices 21–25)
    "mgmt_sentiment_score", "qa_sentiment_score", "news_sentiment",
    "insider_buy_score", "net_insider_delta",
]

MODALITY_GROUPS: dict[str, list[int]] = {
    "fundamentals": [0, 1, 2, 3, 4, 5, 6],
    "technical":    [7, 8, 9, 10, 11, 12],
    "macro":        [13, 14, 15, 16, 17],
    "options":      [18, 19, 20],
    "sentiment":    [21, 22, 23, 24, 25],
}

PERSONA_MODALITIES: dict[str, list[str]] = {
    "value_fundamentalist":  ["fundamentals"],
    "growth_visionary":      ["fundamentals"],
    "quant_momentum":        ["technical"],
    "technical_analyst":     ["technical"],
    "macro_topdown":         ["macro", "fundamentals"],
    "options_flow_trader":   ["options", "technical"],
    "sentiment_trader":      ["sentiment"],
    "insider_institutional": ["sentiment", "fundamentals"],
    "dividend_income":       ["fundamentals"],
}

N_FEATURES = len(FEATURE_ORDER)
HORIZONS = [21, 63, 126, 252]  # trading days: 1M, 3M, 6M, 12M
