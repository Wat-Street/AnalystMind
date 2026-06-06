# Source:
#   - defeatbeta-api (https://github.com/defeat-beta/defeatbeta-api) for transcript text + speaker segmentation

from __future__ import annotations
import re
from typing import Any
import pandas as pd

TICKERS = ("AAPL", "MSFT", "NVDA", "TSLA", "AMZN")
SOURCE = "defeatbeta"
MIN_FISCAL_YEAR = 2015

# The Operator paragraph that opens the analyst q&a marks the management -> q&a transition.
_QA_OPEN_RE = re.compile(
    r"question[-\s]and[-\s]answer"
    r"|operator instructions"
    r"|first question"
    r"|floor is (?:now )?open"
    r"|open .{0,20}for questions"
    r"|(?:we(?:'ll| will)|now|like to).{0,40}\bquestions\b",
    re.IGNORECASE,
)
_OPERATOR = "operator"
_REQUIRED_COLUMNS = ("paragraph_number", "speaker", "content")

def _is_operator(speaker: str) -> bool:
    return speaker.strip().lower() == _OPERATOR


def _classify_speaker_type(speaker: str, role: str, mgmt_speakers: set[str]) -> str:
    """Resolve a speaker to 'operator', 'executive', or 'analyst' by position

    Everyone who delivers prepared remarks is company management; in Q&A any new voice is an analyst
    """
    if _is_operator(speaker):
        return "operator"
    if role == "management":
        return "executive"
    if speaker.strip().lower() in mgmt_speakers:
        return "executive"
    return "analyst"


# transform
def split_segments(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Tag each transcript paragraph with its section, speaker identity, and Q&A exchange.
    """
    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"transcript dataframe missing required columns: {missing}")
    if len(df) == 0:
        raise ValueError("transcript dataframe is empty")

    ordered = df.sort_values("paragraph_number")

    # Pass 1: section tagging + collect the speakers who delivered prepared remarks (the executives)
    tagged: list[tuple[int, str, str, str]] = []
    in_qa = False
    mgmt_speakers: set[str] = set()
    for _, row in ordered.iterrows():
        speaker = str(row["speaker"]).strip()
        text = str(row["content"]).strip()
        if not in_qa and _is_operator(speaker) and _QA_OPEN_RE.search(text):
            in_qa = True
        role = "qa" if in_qa else "management"
        tagged.append((int(row["paragraph_number"]), role, speaker, text))
        if role == "management" and not _is_operator(speaker):
            mgmt_speakers.add(speaker.lower())

    # Pass 2: classify each speaker by position and pair questions with answers
    segments: list[dict[str, Any]] = []
    exchange = 0
    prev_qa_type: str | None = None
    for paragraph, role, speaker, text in tagged:
        speaker_type = _classify_speaker_type(speaker, role, mgmt_speakers)
        if role == "qa":
            if speaker_type == "analyst" and prev_qa_type != "analyst":
                exchange += 1
            qa_exchange = exchange if (exchange and speaker_type != "operator") else None
            prev_qa_type = speaker_type
        else:
            qa_exchange = None
        segments.append({
            "paragraph": paragraph,
            "role": role,
            "speaker": speaker,
            "speaker_type": speaker_type,
            "qa_exchange": qa_exchange,
            "text": text,
        })
    return segments


def build_record(
    ticker: str,
    fiscal_year: int,
    fiscal_quarter: int,
    report_date: Any,
    transcript_df: pd.DataFrame,
    source_transcript_id: Any = None,
) -> dict[str, Any]:
    if not 1 <= int(fiscal_quarter) <= 4:
        raise ValueError(f"fiscal_quarter must be 1-4, got {fiscal_quarter}")

    segments = split_segments(transcript_df)
    report = pd.to_datetime(report_date, errors="coerce")
    return {
        "ticker": ticker,
        "quarter": f"{int(fiscal_year)}Q{int(fiscal_quarter)}",
        "source": SOURCE,
        "source_transcript_id": None if source_transcript_id is None else str(source_transcript_id),
        "report_date": None if pd.isna(report) else report.date().isoformat(),
        "segments": segments,
    }


def fetch_transcript_list(ticker: str) -> pd.DataFrame:
    """List available earnings calls for a ticker: columns symbol, fiscal_year, fiscal_quarter, report_date"""
    from defeatbeta_api.data.ticker import Ticker

    return Ticker(ticker).earning_call_transcripts().get_transcripts_list()


def fetch_transcript(ticker: str, fiscal_year: int, fiscal_quarter: int) -> pd.DataFrame:
    """Fetch one call's paragraphs: columns paragraph_number, speaker, content"""
    from defeatbeta_api.data.ticker import Ticker

    return Ticker(ticker).earning_call_transcripts().get_transcript(fiscal_year, fiscal_quarter)


#ingest
def ingest(ticker: str) -> list[dict[str, Any]]:
    """Fetch and normalize all earnings call transcripts (fiscal year >= MIN_FISCAL_YEAR) for a ticker

    Returns a list of records
    """
    listing = fetch_transcript_list(ticker)

    records: list[dict[str, Any]] = []
    for _, row in listing.iterrows():
        year = int(row["fiscal_year"])
        if year < MIN_FISCAL_YEAR:
            continue
        quarter = int(row["fiscal_quarter"])
        report_date = row.get("report_date")
        transcript_id = row.get("transcripts_id")
        if transcript_id is not None and pd.isna(transcript_id):
            transcript_id = None
        df = fetch_transcript(ticker, year, quarter)
        records.append(build_record(ticker, year, quarter, report_date, df, transcript_id))
    return records


def ingest_all() -> dict[str, list[dict[str, Any]]]:
    return {ticker: ingest(ticker) for ticker in TICKERS}
