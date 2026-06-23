# FinBERT text tokenization for sentiment scoring
# NOTE: This module tokenizes transcript *text*: it chunks paragraphs 
# to fit FinBERT's 512-token window. 

#TO-DO: still need to run FinBERT on chunks to get scores for dataset - Kiana

from __future__ import annotations
import re
from typing import Any

FINBERT_MODEL = "ProsusAI/finbert"
MAX_TOKENS = 450

# Prefer to break a chunk on a sentence boundary so a split never lands mid-clause when avoidable.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

_tokenizer = None


def get_tokenizer():
    """Lazily load and cache the FinBERT WordPiece tokenizer.
    """
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer

        _tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
    return _tokenizer


def _count_tokens(text: str, tokenizer) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def _split_long_sentence(sentence: str, tokenizer, max_tokens: int) -> list[tuple[str, int]]:
    """Hard-split a single sentence that on its own exceeds max_tokens, on token windows."""
    ids = tokenizer.encode(sentence, add_special_tokens=False)
    chunks: list[tuple[str, int]] = []
    for start in range(0, len(ids), max_tokens):
        window = ids[start:start + max_tokens]
        chunks.append((tokenizer.decode(window).strip(), len(window)))
    return chunks


def chunk_text(text: str, tokenizer=None, max_tokens: int = MAX_TOKENS) -> list[tuple[str, int]]:
    """Split text into chunks of at most max_tokens FinBERT tokens, preferring sentence breaks.

    Returns (chunk_text, token_count) pairs.
    """
    if tokenizer is None:
        tokenizer = get_tokenizer()

    total = _count_tokens(text, tokenizer)
    if total <= max_tokens:
        return [(text, total)]

    chunks: list[tuple[str, int]] = []
    current: list[str] = []
    current_tokens = 0
    for sentence in _SENTENCE_RE.split(text):
        sentence = sentence.strip()
        if not sentence:
            continue
        n = _count_tokens(sentence, tokenizer)
        if n > max_tokens:
            # A single sentence larger than the window: flush what we have, then window-split it.
            if current:
                chunks.append((" ".join(current), current_tokens))
                current, current_tokens = [], 0
            chunks.extend(_split_long_sentence(sentence, tokenizer, max_tokens))
            continue
        if current_tokens + n > max_tokens:
            chunks.append((" ".join(current), current_tokens))
            current, current_tokens = [sentence], n
        else:
            current.append(sentence)
            current_tokens += n
    if current:
        chunks.append((" ".join(current), current_tokens))
    return chunks


def chunk_segments(segments: list[dict[str, Any]], tokenizer=None) -> list[dict[str, Any]]:
    """Expand transcript segments into FinBERT-sized scoring chunks.

    Each input segment (one paragraph) yields one or more chunk dicts that carry the paragraph's identity and tags (paragraph, role, speaker,
    speaker_type, qa_exchange) plus the chunk's `chunk` index within the paragraph, its `token_count`, and `text`.
    """
    if tokenizer is None:
        tokenizer = get_tokenizer()

    chunks: list[dict[str, Any]] = []
    for seg in segments:
        for chunk_index, (chunk, token_count) in enumerate(chunk_text(seg["text"], tokenizer)):
            chunks.append({
                "paragraph": seg["paragraph"],
                "chunk": chunk_index,
                "role": seg["role"],
                "speaker": seg["speaker"],
                "speaker_type": seg["speaker_type"],
                "qa_exchange": seg["qa_exchange"],
                "token_count": token_count,
                "text": chunk,
            })
    return chunks
