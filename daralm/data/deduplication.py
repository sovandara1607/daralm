"""Exact document deduplication."""

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
    """Remove exact/near-exact duplicate documents."""
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
