from __future__ import annotations

import pandas as pd
import pytest

from app.ingestion import transcripts
from app.ingestion.transcripts import (
    MIN_FISCAL_YEAR,
    build_record,
    ingest,
    split_segments,
)


def _transcript_df(rows: list[tuple[str, str]]) -> pd.DataFrame:
    # rows in spoken order
    return pd.DataFrame({
        "paragraph_number": range(1, len(rows) + 1),
        "speaker": [r[0] for r in rows],
        "content": [r[1] for r in rows],
    })


_CALL = [
    ("Operator", "Good day, and welcome to the Q1 earnings call."),
    ("Jane CEO", "Thanks. We had a strong quarter with record revenue."),
    ("John CFO", "Margins expanded 200 basis points year over year."),
    ("Operator", "We will now begin the question-and-answer session."),
    ("Analyst A", "Can you talk about guidance for next quarter?"),
    ("Jane CEO", "Sure, we expect continued momentum."),
]


# split_segments
def test_split_segments_management_then_qa():
    segs = split_segments(_transcript_df(_CALL))
    roles = [s["role"] for s in segs]
    # The first three paragraphs are prepared remarks; the Operator Q&A line flips to qa
    assert roles == ["management", "management", "management", "qa", "qa", "qa"]


def test_split_segments_no_qa_marker_all_management():
    remarks_only = [
        ("Operator", "Welcome to the call."),
        ("Jane CEO", "Here are our results."),
        ("John CFO", "And the financial detail."),
    ]
    segs = split_segments(_transcript_df(remarks_only))
    assert {s["role"] for s in segs} == {"management"}


def test_split_segments_preserves_speaker_and_order():
    segs = split_segments(_transcript_df(_CALL))
    assert [s["paragraph"] for s in segs] == [1, 2, 3, 4, 5, 6]
    assert segs[1]["speaker"] == "Jane CEO"


def test_split_segments_premature_management_questions_dont_trigger():
    # A management member teeing up Q&A should not flip the role, only the Operator does
    rows = [
        ("Jane CEO", "With that, we'll open it up for questions."),
        ("Operator", "Thank you. Our first question comes from Analyst A."),
        ("Analyst A", "Great quarter."),
    ]
    roles = [s["role"] for s in split_segments(_transcript_df(rows))]
    assert roles == ["management", "qa", "qa"]


def test_split_segments_missing_column_raises():
    df = _transcript_df(_CALL).drop(columns=["speaker"])
    with pytest.raises(ValueError, match="missing required columns"):
        split_segments(df)


def test_split_segments_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        split_segments(_transcript_df([]))


def test_split_segments_pairs_questions_with_answers():
    segs = split_segments(_transcript_df(_CALL))
    by_para = {s["paragraph"]: s for s in segs}
    assert by_para[4]["qa_exchange"] is None          # operator opener
    assert by_para[5]["qa_exchange"] == 1             # Analyst A's question
    assert by_para[6]["qa_exchange"] == 1             # Jane CEO's answer, same exchange


# speaker_type + Q&A pairing
_RICH_CALL = [
    ("Operator", "Welcome to the Q2 earnings call."),
    ("Tim Cook -- Chief Executive Officer", "Revenue hit a record this quarter."),
    ("Luca Maestri -- Chief Financial Officer", "Gross margin was 46%."),
    ("Operator", "We will now begin the question-and-answer session."),
    ("Katy Huberty -- Morgan Stanley -- Analyst", "Can you discuss iPhone demand?"),
    ("Tim Cook -- Chief Executive Officer", "Demand was strong across regions."),
    ("Operator", "Our next question comes from Toni Sacconaghi."),
    ("Toni Sacconaghi -- Bernstein -- Analyst", "What about services margins?"),
    ("Luca Maestri -- Chief Financial Officer", "Services margin expanded."),
]


def test_split_segments_keeps_speaker_label_verbatim():
    segs = split_segments(_transcript_df(_RICH_CALL))
    assert segs[1]["speaker"] == "Tim Cook -- Chief Executive Officer"
    assert set(segs[1].keys()) == {"paragraph", "role", "speaker", "speaker_type", "qa_exchange", "text"}


def test_split_segments_classifies_speaker_type_by_position():
    segs = split_segments(_transcript_df(_RICH_CALL))
    types = [s["speaker_type"] for s in segs]
    assert types == [
        "operator", "executive", "executive", "operator",
        "analyst", "executive", "operator", "analyst", "executive",
    ]


def test_split_segments_executive_answering_in_qa_is_still_executive():
    segs = split_segments(_transcript_df(_RICH_CALL))
    answer = segs[5]
    assert answer["speaker"] == "Tim Cook -- Chief Executive Officer"
    assert answer["speaker_type"] == "executive"
    assert answer["role"] == "qa"


def test_split_segments_pairs_multiple_qa_exchanges():
    segs = split_segments(_transcript_df(_RICH_CALL))
    ex = [s["qa_exchange"] for s in segs]
    assert ex == [None, None, None, None, 1, 1, None, 2, 2]


# build_record
def test_build_record_contract():
    rec = build_record("AAPL", 2024, 1, _transcript_df(_CALL))
    assert set(rec.keys()) == {"ticker", "quarter", "source", "segments"}
    assert rec["ticker"] == "AAPL"
    assert rec["quarter"] == "2024Q1"
    assert rec["source"] == "defeatbeta"
    assert len(rec["segments"]) == len(_CALL)
    assert {s["role"] for s in rec["segments"]} <= {"management", "qa"}


def test_build_record_rejects_bad_quarter():
    with pytest.raises(ValueError, match="fiscal_quarter"):
        build_record("AAPL", 2024, 5, _transcript_df(_CALL))


#ingest
def test_ingest_filters_by_min_year_and_builds_records(monkeypatch):
    listing = pd.DataFrame({
        "symbol":         ["AAPL", "AAPL", "AAPL"],
        "fiscal_year":    [MIN_FISCAL_YEAR - 1, MIN_FISCAL_YEAR, MIN_FISCAL_YEAR + 1],
        "fiscal_quarter": [4, 1, 2],
    })
    monkeypatch.setattr(transcripts, "fetch_transcript_list", lambda t: listing)
    monkeypatch.setattr(transcripts, "fetch_transcript", lambda t, y, q: _transcript_df(_CALL))

    records = ingest("AAPL")

    assert [r["quarter"] for r in records] == [f"{MIN_FISCAL_YEAR}Q1", f"{MIN_FISCAL_YEAR + 1}Q2"]
    assert all(r["ticker"] == "AAPL" for r in records)
    assert all(r["segments"] for r in records)
