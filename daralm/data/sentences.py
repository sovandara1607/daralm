"""A minimal sentence splitter — Khmer- and English-aware."""

from __future__ import annotations

import re

# Khmer sentences end in ។ (khan) or ៕ (bariyoosan); English/Latin text ends in ./!/?.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[។៕.!?])\s+")


def split_sentences(text: str, min_chars: int = 20, max_chars: int = 200) -> list[str]:
    """Split `text` into sentence-shaped chunks within [min_chars, max_chars]."""
    if min_chars <= 0 or max_chars <= 0:
        raise ValueError(f"min_chars and max_chars must be positive, got {min_chars}, {max_chars}")
    if min_chars > max_chars:
        raise ValueError(f"min_chars ({min_chars}) must be <= max_chars ({max_chars})")

    raw_sentences = _SENTENCE_BOUNDARY.split(text.strip())
    return [s.strip() for s in raw_sentences if min_chars <= len(s.strip()) <= max_chars]
