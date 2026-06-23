from __future__ import annotations

import pytest

from app.ml import text_tokenizer
from app.ml.text_tokenizer import MAX_TOKENS, chunk_segments, chunk_text


class _FakeTokenizer:
    """Whitespace stand-in for the FinBERT WordPiece tokenizer.
    """

    def encode(self, text: str, add_special_tokens: bool = False) -> list[str]:
        return text.split()

    def decode(self, ids: list[str]) -> str:
        return " ".join(ids)


@pytest.fixture(autouse=True)
def _fake_tokenizer(monkeypatch):
    monkeypatch.setattr(text_tokenizer, "get_tokenizer", lambda: _FakeTokenizer())


# chunk_text
def test_chunk_text_short_text_is_single_chunk():
    chunks = chunk_text("We had a strong quarter with record revenue.")
    assert len(chunks) == 1
    text, token_count = chunks[0]
    assert text == "We had a strong quarter with record revenue."
    assert token_count == 8


def test_chunk_text_window_splits_long_unbroken_text():
    # No sentence breaks, longer than the window -> window-split into ordered chunks, nothing lost.
    long_text = " ".join(["word"] * (MAX_TOKENS * 2 + 30))
    chunks = chunk_text(long_text)
    assert len(chunks) == 3
    assert all(token_count <= MAX_TOKENS for _, token_count in chunks)
    assert " ".join(text for text, _ in chunks) == long_text


def test_chunk_text_packs_sentences_under_the_limit():
    sentence = " ".join(["alpha"] * 90) + "."
    paragraph = " ".join([sentence] * 8)  # 8 * ~91 tokens ~ 728 > MAX_TOKENS
    chunks = chunk_text(paragraph)
    assert len(chunks) >= 2
    assert all(token_count <= MAX_TOKENS for _, token_count in chunks)


def test_chunk_text_preserves_raw_text_verbatim():
    text = "Margins?  We DON'T disclose that... yet."
    chunks = chunk_text(text)
    assert chunks == [(text, len(text.split()))]


def test_chunk_text_accepts_injected_tokenizer():
    chunks = chunk_text("one two three", tokenizer=_FakeTokenizer())
    assert chunks == [("one two three", 3)]


# chunk_segments
def _segment(paragraph: int, text: str, **overrides) -> dict:
    seg = {
        "paragraph": paragraph,
        "role": "qa",
        "speaker": "Jane CEO",
        "speaker_type": "executive",
        "qa_exchange": 1,
        "text": text,
    }
    seg.update(overrides)
    return seg


def test_chunk_segments_carries_tags_and_indexes_chunks():
    long_text = " ".join(["word"] * (MAX_TOKENS + 5))
    segments = [
        _segment(1, "Short answer.", qa_exchange=1),
        _segment(2, long_text, qa_exchange=2),
    ]
    chunks = chunk_segments(segments)

    # Paragraph 1 stays one chunk; paragraph 2 splits into two.
    by_para = {}
    for c in chunks:
        by_para.setdefault(c["paragraph"], []).append(c)
    assert [c["chunk"] for c in by_para[1]] == [0]
    assert [c["chunk"] for c in by_para[2]] == [0, 1]

    # Tags from the source segment ride along onto every chunk.
    assert by_para[2][0]["qa_exchange"] == 2
    assert by_para[2][0]["speaker_type"] == "executive"
    assert all(c["token_count"] <= MAX_TOKENS for c in chunks)
    assert set(chunks[0].keys()) == {
        "paragraph", "chunk", "role", "speaker", "speaker_type", "qa_exchange", "token_count", "text",
    }
