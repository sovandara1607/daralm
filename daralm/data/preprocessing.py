"""Filtering, quality control, and corpus statistics.

`cleaner.py` provides pure text transforms; this module decides, using
those transforms, whether a document is worth keeping at all, and reports
on the corpus as a whole. This is where "collect -> inspect -> clean ->
analyze" (spec section 8) actually happens — nothing here trains a model or
touches a tokenizer, since neither exists until later phases.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from daralm.data.cleaner import clean_text, corruption_ratio, detect_language, url_density

# Quality thresholds. Deliberately generous — Phase 1's job is to catch
# obviously broken documents (empty, corrupted, link-spam, pathologically
# short/long), not to make subjective judgments about writing quality.
MIN_CHARS = 20
MAX_CHARS = 200_000
MAX_CORRUPTION_RATIO = 0.01
MAX_URL_DENSITY = 0.3

FilterReason = str  # e.g. "empty", "too_short", "too_long", "corrupted_encoding", "url_spam"


def filter_reason(raw_text: str, cleaned_text: str) -> FilterReason | None:
    """Return why a document should be dropped, or None if it passes.

    Checked in a fixed order so the reported reason is deterministic and the
    cheapest checks (on raw text, before the cost of cleaning even matters)
    run first.
    """
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
    """Clean and quality-filter one raw record.

    Args:
        record: A raw record with at least a "text" field; "language" and
            "source" are carried through if present.

    Returns:
        `(cleaned_record, None)` if the document passes, or
        `(None, reason)` if it was filtered out. `cleaned_record`'s
        "language" is the script-detected language when the input didn't
        already specify one (or specified one we couldn't confirm).
    """
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
    """Compute corpus-level statistics (spec section 8).

    `duplicate_count` / `total_seen` come from the deduplication step so the
    duplicate rate reflects the whole pipeline, not just this record list.
    Token-based stats (tokens, tokenizer compression ratio) are not
    computable yet — there is no tokenizer until Phase 2 — and are reported
    as `null` with an explanatory note rather than silently omitted.
    """
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
