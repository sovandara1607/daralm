"""Exact document deduplication.

We hash a normalized form of each document's text (lowercased, whitespace-
collapsed, punctuation-stripped — for hashing purposes only, the stored
`text` field is never mutated) and drop any document whose hash we've
already seen. `\\w` in Python's `re` module is Unicode-aware by default, so
stripping non-word characters for the hash key works correctly on Khmer
text too, not just ASCII.

This catches exact and near-exact duplicates (differing only in whitespace,
case, or punctuation) cheaply, in one pass, with no extra dependency. It
will NOT catch near-duplicates that differ in actual wording (e.g. two
paraphrased copies of the same article) — that needs fuzzy techniques like
MinHash/SimHash, which is more complexity than Phase 1 needs; noted here as
a deliberate scope cut, not an oversight, and a natural place to extend
later if duplicate analysis shows it's needed.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

_NON_WORD_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_for_hash(text: str) -> str:
    """Lowercase, strip punctuation, and collapse whitespace — for hashing only."""
    text = text.lower()
    text = _NON_WORD_RE.sub("", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def document_hash(text: str) -> str:
    """SHA-256 hex digest of the normalized text, used as a dedup key."""
    return hashlib.sha256(normalize_for_hash(text).encode("utf-8")).hexdigest()


def deduplicate(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Remove exact/near-exact duplicate documents.

    Returns (kept_records, num_duplicates_removed). Order of `kept_records`
    matches the first occurrence of each document in the input.
    """
    seen: set[str] = set()
    kept: list[dict[str, Any]] = []
    duplicates = 0

    for record in records:
        h = document_hash(record["text"])
        if h in seen:
            duplicates += 1
            continue
        seen.add(h)
        kept.append(record)

    return kept, duplicates
