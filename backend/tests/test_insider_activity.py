from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock

from backend.app.ingestion.insider_activity import (
    compute,
    _parse_filing_xml,
    _insider_buy_score,
    _net_insider_delta,
    MIN_FILINGS,
    SCORE_SATURATION,
    NET_DELTA_SCALE,
)
import pandas as pd
import numpy as np


# ── Fixtures ────────────────────────────────────────────────────────────────

SELL_XML = """<?xml version="1.0"?>
<ownershipDocument>
    <nonDerivativeTable>
        <nonDerivativeTransaction>
            <transactionCoding>
                <transactionCode>S</transactionCode>
            </transactionCoding>
            <transactionAmounts>
                <transactionShares><value>1000</value></transactionShares>
                <transactionPricePerShare><value>200</value></transactionPricePerShare>
            </transactionAmounts>
        </nonDerivativeTransaction>
    </nonDerivativeTable>
</ownershipDocument>"""

BUY_XML = """<?xml version="1.0"?>
<ownershipDocument>
    <nonDerivativeTable>
        <nonDerivativeTransaction>
            <transactionCoding>
                <transactionCode>P</transactionCode>
            </transactionCoding>
            <transactionAmounts>
                <transactionShares><value>500</value></transactionShares>
                <transactionPricePerShare><value>150</value></transactionPricePerShare>
            </transactionAmounts>
        </nonDerivativeTransaction>
    </nonDerivativeTable>
</ownershipDocument>"""

MISSING_PRICE_XML = """<?xml version="1.0"?>
<ownershipDocument>
    <nonDerivativeTable>
        <nonDerivativeTransaction>
            <transactionCoding>
                <transactionCode>P</transactionCode>
            </transactionCoding>
            <transactionAmounts>
                <transactionShares><value>500</value></transactionShares>
            </transactionAmounts>
        </nonDerivativeTransaction>
    </nonDerivativeTable>
</ownershipDocument>"""

MALFORMED_XML = "this is not xml at all <<<"


# ── XML parsing ─────────────────────────────────────────────────────────────

def test_parse_sell_transaction():
    result = _parse_filing_xml(SELL_XML)
    assert result is not None
    assert result["transaction_code"] == "S"
    assert result["shares"] == 1000.0
    assert result["price_per_share"] == 200.0
    assert result["is_buy"] is False


def test_parse_buy_transaction():
    result = _parse_filing_xml(BUY_XML)
    assert result is not None
    assert result["transaction_code"] == "P"
    assert result["shares"] == 500.0
    assert result["price_per_share"] == 150.0
    assert result["is_buy"] is True


def test_parse_missing_price_returns_none():
    result = _parse_filing_xml(MISSING_PRICE_XML)
    assert result is None


def test_parse_malformed_xml_returns_none():
    result = _parse_filing_xml(MALFORMED_XML)
    assert result is None


def test_parse_empty_string_returns_none():
    result = _parse_filing_xml("")
    assert result is None


# ── Scoring ──────────────────────────────────────────────────────────────────

def test_insider_buy_score_all_buys():
    df = pd.DataFrame([{"is_buy": True}] * SCORE_SATURATION)
    assert _insider_buy_score(df) == pytest.approx(1.0, abs=1e-6)


def test_insider_buy_score_no_buys():
    df = pd.DataFrame([{"is_buy": False}] * 5)
    assert _insider_buy_score(df) == pytest.approx(0.0, abs=1e-6)


def test_insider_buy_score_saturates():
    # More buys than SCORE_SATURATION should still cap at 1.0
    df = pd.DataFrame([{"is_buy": True}] * (SCORE_SATURATION * 3))
    assert _insider_buy_score(df) == pytest.approx(1.0, abs=1e-6)


def test_insider_buy_score_partial():
    df = pd.DataFrame([{"is_buy": True}] * 5 + [{"is_buy": False}] * 5)
    score = _insider_buy_score(df)
    assert 0.0 < score < 1.0


def test_net_insider_delta_all_sells():
    df = pd.DataFrame([{
        "shares": 1000.0,
        "price_per_share": 1000.0,  # $1M sell
        "is_buy": False,
    }])
    assert _net_insider_delta(df) == pytest.approx(-1.0, abs=1e-6)


def test_net_insider_delta_all_buys():
    df = pd.DataFrame([{
        "shares": 1000.0,
        "price_per_share": 1000.0,  # $1M buy
        "is_buy": True,
    }])
    assert _net_insider_delta(df) == pytest.approx(1.0, abs=1e-6)


def test_net_insider_delta_neutral():
    df = pd.DataFrame([
        {"shares": 500.0, "price_per_share": 100.0, "is_buy": True},
        {"shares": 500.0, "price_per_share": 100.0, "is_buy": False},
    ])
    assert _net_insider_delta(df) == pytest.approx(0.0, abs=1e-6)


# ── Output contract ──────────────────────────────────────────────────────────

def _make_hit(accession: str, filename: str, cik: str) -> dict:
    return {
        "_id": f"{accession}:{filename}",
        "_source": {"ciks": [cik]},
    }


def test_output_contract():
    hits = [_make_hit("0001140361-26-020871", "form4.xml", "0000320193")]
    with patch("backend.app.ingestion.insider_activity._fetch_form4_filings", return_value=hits), \
         patch("backend.app.ingestion.insider_activity._fetch_filing_xml", return_value=BUY_XML):
        out = compute("AAPL")

    assert set(out.keys()) == {"insider_buy_score", "institutional_flow", "net_insider_delta"}
    for k, v in out.items():
        assert isinstance(v, float), f"{k} is not float"
        assert not np.isnan(v), f"{k} is NaN"
        assert -1.0 <= v <= 1.0, f"{k}={v} outside [-1, 1]"


def test_institutional_flow_is_zero():
    hits = [_make_hit("0001140361-26-020871", "form4.xml", "0000320193")]
    with patch("backend.app.ingestion.insider_activity._fetch_form4_filings", return_value=hits), \
         patch("backend.app.ingestion.insider_activity._fetch_filing_xml", return_value=BUY_XML):
        out = compute("AAPL")
    assert out["institutional_flow"] == 0.0


# ── Validation ───────────────────────────────────────────────────────────────

def test_no_filings_raises():
    with patch("backend.app.ingestion.insider_activity._fetch_form4_filings", return_value=[]):
        with pytest.raises(ValueError, match="No Form 4 filings"):
            compute("FAKE")


def test_all_filings_unparseable_raises():
    hits = [_make_hit("0001140361-26-020871", "form4.xml", "0000320193")]
    with patch("backend.app.ingestion.insider_activity._fetch_form4_filings", return_value=hits), \
         patch("backend.app.ingestion.insider_activity._fetch_filing_xml", return_value=MALFORMED_XML):
        with pytest.raises(ValueError, match="Could not parse"):
            compute("AAPL")


def test_fetch_xml_failure_skips_filing():
    hits = [
        _make_hit("0001140361-26-020871", "form4.xml", "0000320193"),
        _make_hit("0001140361-26-020872", "form4.xml", "0000320193"),
    ]
    # First returns None (network failure), second returns valid XML
    with patch("backend.app.ingestion.insider_activity._fetch_form4_filings", return_value=hits), \
         patch("backend.app.ingestion.insider_activity._fetch_filing_xml", side_effect=[None, BUY_XML]):
        out = compute("AAPL")
    assert out["insider_buy_score"] > 0.0


def test_missing_id_field_skips_hit():
    hits = [{"_source": {"ciks": ["0000320193"]}}]  # no _id field
    with patch("backend.app.ingestion.insider_activity._fetch_form4_filings", return_value=hits):
        with pytest.raises(ValueError, match="Could not parse"):
            compute("AAPL")