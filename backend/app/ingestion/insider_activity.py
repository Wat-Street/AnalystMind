# Insider and institutional activity ingestion pipeline.
# Sources: SEC EDGAR Form 4 (free)
# Computes: insider_buy_score, institutional_flow, net_insider_delta.
# Used by: insider_institutional persona.

# Kiana

from __future__ import annotations
import xml.etree.ElementTree as ET
import pandas as pd
import requests

# EDGAR endpoints
EDGAR_SEARCH_URL  = "https://efts.sec.gov/LATEST/search-index"
EDGAR_ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data"

# Transaction codes: P = open market purchase, S = open market sale
BUY_CODES  = {"P"}
SELL_CODES = {"S"}

# Scoring
SCORE_SATURATION = 10      # number of insider buys that maps to a score of 1.0
NET_DELTA_SCALE  = 1_000_000  # $1M net = scale anchor

# Validation
MIN_FILINGS      = 1
_REQUIRED_FIELDS = ("transaction_code", "shares", "price_per_share", "is_buy")

HEADERS = {"User-Agent": "WatStreet kiana@uwaterloo.ca"}


def _fetch_form4_filings(ticker: str) -> list[dict]:
    """Search EDGAR for Form 4 filings for a given ticker. Returns raw hits."""
    params = {"q": f'"{ticker}"', "forms": "4"}
    response = requests.get(EDGAR_SEARCH_URL, params=params, headers=HEADERS, timeout=10)
    response.raise_for_status()
    return response.json().get("hits", {}).get("hits", [])


def _fetch_filing_xml(cik: str, accession_no: str, filename: str) -> str | None:
    """Fetch the raw XML for a single Form 4 filing from EDGAR archives."""
    accession_clean = accession_no.replace("-", "")
    url = f"{EDGAR_ARCHIVE_URL}/{cik}/{accession_clean}/{filename}"
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        return response.text
    except requests.RequestException:
        return None


def _parse_filing_xml(xml_text: str) -> dict | None:
    """Parse a Form 4 XML document and extract transaction fields."""
    try:
        root = ET.fromstring(xml_text)
        code   = root.find(".//transactionCode")
        shares = root.find(".//transactionShares/value")
        price  = root.find(".//transactionPricePerShare/value")

        if code is None or shares is None or price is None:
            return None

        code_val   = code.text.strip()
        shares_val = float(shares.text or 0)
        price_val  = float(price.text or 0)

        return {
            "transaction_code": code_val,
            "shares":           shares_val,
            "price_per_share":  price_val,
            "is_buy":           code_val in BUY_CODES,
        }
    except (ET.ParseError, ValueError, TypeError):
        return None


def _insider_buy_score(df: pd.DataFrame) -> float:
    """Fraction of buy transactions, saturated at SCORE_SATURATION buys = 1.0."""
    import numpy as np
    buys = df["is_buy"].sum()
    return float(np.clip(buys / SCORE_SATURATION, 0.0, 1.0))


def _net_insider_delta(df: pd.DataFrame) -> float:
    """Net dollar flow normalised to [-1, 1]. Positive = net buying."""
    import numpy as np
    df = df.copy()
    df["dollar_value"] = df["shares"] * df["price_per_share"]
    df["signed"]       = df["dollar_value"] * df["is_buy"].map({True: 1, False: -1})
    net = df["signed"].sum()
    return float(np.clip(net / NET_DELTA_SCALE, -1.0, 1.0))


def _institutional_flow(df: pd.DataFrame) -> float:
    """Placeholder — Form 4 covers insiders only. 13F ingestion needed for institutions."""
    return 0.0


def compute(ticker: str) -> dict[str, float]:
    raw_hits = _fetch_form4_filings(ticker)

    if len(raw_hits) < MIN_FILINGS:
        raise ValueError(f"No Form 4 filings found for {ticker}.")

    parsed = []
    for hit in raw_hits:
        src            = hit.get("_source", {})
        ciks           = src.get("ciks", [])
        raw_id         = hit.get("_id", "")          # e.g. "0001140361-26-020871:form4.xml"
        if ":" not in raw_id or not ciks:
            continue
        accession_no, filename = raw_id.split(":", 1)
        cik = ciks[0].lstrip("0")                    # strip leading zeros

        xml_text = _fetch_filing_xml(cik, accession_no, filename)
        if xml_text is None:
            continue

        record = _parse_filing_xml(xml_text)
        if record:
            parsed.append(record)

    if not parsed:
        raise ValueError(f"Could not parse any Form 4 transactions for {ticker}.")

    df = pd.DataFrame(parsed)

    return {
        "insider_buy_score":  _insider_buy_score(df),
        "institutional_flow": _institutional_flow(df),
        "net_insider_delta":  _net_insider_delta(df),
    }