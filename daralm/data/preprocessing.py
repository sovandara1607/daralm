"""Filtering, quality control, and corpus statistics."""

from __future__ import annotations

from collections import Counter
from typing import Any

from daralm.data.cleaner import clean_text, corruption_ratio, detect_language, url_density

MIN_CHARS = 20
MAX_CHARS = 200_000
MAX_CORRUPTION_RATIO = 0.01
MAX_URL_DENSITY = 0.3

FilterReason = str  # e.g. "empty", "too_short", "too_long", "corrupted_encoding", "url_spam"


def filter_reason(raw_text: str, cleaned_text: str) -> FilterReason | None:
    """Return why a document should be dropped, or None if it passes."""
    if not raw_text or not raw_text.strip():
        return "empty"
    if corruption_ratio(raw_text) > MAX_CORRUPTION_RATIO:
        return "corrupted_encoding"
    if not cleaned_text:
        return "empty_after_clean"
    if len(cleaned_text) < MIN_CHARS:
        return "too_short"
    if len(cleaned_text) > MAX_CHARS:
        return "too_long"
    if url_density(cleaned_text) > MAX_URL_DENSITY:
        return "url_spam"
    return None


def process_document(record: dict[str, Any]) -> tuple[dict[str, Any] | None, FilterReason | None]:
    """Clean and quality-filter one raw record."""
    raw_text = record.get("text", "")
    cleaned_text = clean_text(raw_text) if raw_text else ""

    reason = filter_reason(raw_text, cleaned_text)
    if reason is not None:
        return None, reason

    detected = detect_language(cleaned_text)
    language = record.get("language") or detected

    cleaned_record = {
        "text": cleaned_text,
        "language": language,
        "source": record.get("source", "unknown"),
    }
    return cleaned_record, None


def compute_corpus_stats(
    records: list[dict[str, Any]],
    *,
    duplicate_count: int = 0,
    total_seen: int = 0,
) -> dict[str, Any]:
    """Compute corpus-level statistics (spec section 8)."""
    documents = len(records)
    total_chars = sum(len(r["text"]) for r in records)
    total_words = sum(len(r["text"].split()) for r in records)
    language_counts = Counter(r["language"] for r in records)

    def pct(lang: str) -> float:
        return round(100 * language_counts.get(lang, 0) / documents, 2) if documents else 0.0

    return {
        "documents": documents,
        "characters": total_chars,
        "words": total_words,
        "khmer_percentage": pct("km"),
        "english_percentage": pct("en"),
        "mixed_percentage": pct("mixed"),
        "unknown_percentage": pct("unknown"),
        "average_document_chars": round(total_chars / documents, 1) if documents else 0.0,
        "duplicate_rate_percentage": (
            round(100 * duplicate_count / total_seen, 2) if total_seen else 0.0
        ),
        "tokens": None,
        "tokenizer_compression_ratio": None,
        "note": "Token-based stats require a trained tokenizer — see Phase 2.",
    }
