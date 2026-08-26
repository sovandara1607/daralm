"""A minimal sentence splitter — Khmer- and English-aware.

Built for `scripts/prepare_grammar_dataset.py`: base-pretraining documents
(Wikipedia articles) are far too long to use whole as grammar-correction
training examples, so they're split into individual sentences first. Not a
general-purpose sentence tokenizer (no abbreviation handling, no quote-
nesting) — good enough for chunking long documents into bounded, roughly-
sentence-shaped pieces, which is all this project currently needs it for.
"""

from __future__ import annotations

import re

# Khmer sentences end in ។ (khan) or ៕ (bariyoosan); English/Latin text
# ends in ./!/?. Splitting keeps the terminator attached to its sentence
# (a lookbehind) rather than discarding it — the terminator is real content
# a grammar-correction model should reproduce, not punctuation to strip.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[។៕.!?])\s+")


def split_sentences(text: str, min_chars: int = 20, max_chars: int = 200) -> list[str]:
    """Split `text` into sentence-shaped chunks within [min_chars, max_chars].

    Chunks outside that range are dropped rather than truncated or padded —
    too-short chunks (stray fragments, list bullets) carry little grammar
    signal, and too-long ones defeat the point of chunking a long document
    down to trainable-sized pieces in the first place.
    """
    if min_chars <= 0 or max_chars <= 0:
        raise ValueError(f"min_chars and max_chars must be positive, got {min_chars}, {max_chars}")
    if min_chars > max_chars:
        raise ValueError(f"min_chars ({min_chars}) must be <= max_chars ({max_chars})")

    raw_sentences = _SENTENCE_BOUNDARY.split(text.strip())
    return [s.strip() for s in raw_sentences if min_chars <= len(s.strip()) <= max_chars]
