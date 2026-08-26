"""Character/word error rate — the standard metrics for grammar/spelling
correction (`ROADMAP_NLP_PLATFORM.md`'s recommended eval for this
capability), since ground truth is directly known (the un-corrupted source
sentence), unlike open-ended generation tasks that need a judge or a
proxy metric like perplexity.

Both are Levenshtein edit distance (insertions + deletions + substitutions)
between a hypothesis and a reference, normalized by the reference's length
— CER at the character level, WER at the whitespace-tokenized word level.
"""

from __future__ import annotations


def _levenshtein(a: list[str], b: list[str]) -> int:
    """Classic O(len(a) * len(b)) edit-distance DP over two token sequences."""
    if not a:
        return len(b)
    if not b:
        return len(a)

    previous_row = list(range(len(b) + 1))
    for i, token_a in enumerate(a, start=1):
        current_row = [i] + [0] * len(b)
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                current_row[j] = previous_row[j - 1]
            else:
                current_row[j] = 1 + min(
                    previous_row[j],  # deletion
                    current_row[j - 1],  # insertion
                    previous_row[j - 1],  # substitution
                )
        previous_row = current_row
    return previous_row[-1]


def character_error_rate(hypothesis: str, reference: str) -> float:
    """Levenshtein distance over characters, normalized by len(reference).

    Returns 0.0 for an empty reference and an empty hypothesis (a perfect,
    trivial match) — and 1.0 (not a division error) for an empty reference
    against a non-empty hypothesis, since every character in `hypothesis`
    would need to be deleted.
    """
    if not reference:
        return 0.0 if not hypothesis else 1.0
    distance = _levenshtein(list(hypothesis), list(reference))
    return distance / len(reference)


def word_error_rate(hypothesis: str, reference: str) -> float:
    """Levenshtein distance over whitespace-split words, normalized by
    the reference's word count. Same empty-reference convention as
    `character_error_rate`."""
    reference_words = reference.split()
    hypothesis_words = hypothesis.split()
    if not reference_words:
        return 0.0 if not hypothesis_words else 1.0
    distance = _levenshtein(hypothesis_words, reference_words)
    return distance / len(reference_words)
