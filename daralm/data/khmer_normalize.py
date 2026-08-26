"""Khmer-specific text normalization — the first capability from the NLP
platform roadmap (`ROADMAP_NLP_PLATFORM.md`, "Khmer text normalization"),
and deliberately the smallest possible version of it.

Most of what that roadmap item originally described — Unicode NFC
normalization, zero-width character removal, duplicated-character
collapsing, HTML/whitespace cleanup — is **already implemented** in
`daralm.data.cleaner.clean_text`, verified before writing a line of this
module (`unicodedata.category('\\u200b')` is `Cf`, already stripped by
`normalize_unicode`; same for ZWNJ/ZWJ). Duplicating that here would
violate the project's own "do not rewrite working components
unnecessarily" rule. This module composes `clean_text` and adds only the
two things that were genuinely missing, both verified against the real
training corpus before being written, not assumed:

1. **Khmer digit normalization.** Khmer has its own digit glyphs
   (០-៩, U+17E0-17E9) distinct from Arabic numerals — `clean_text` never
   touched these (they're Unicode category `Nd`, not the `Cf`/control
   categories it strips). No conversion existed anywhere in the pipeline.
2. **Spacing before Khmer sentence punctuation.** Standard Khmer
   typography doesn't put a space before `។` (period-equivalent) the same
   way English doesn't before `.` — but measuring the real training
   corpus found 28.5% of `។` occurrences (9,109 of 31,998 sampled) *do*
   have a preceding space, a real, inconsistent-source-formatting
   artifact, not a hypothetical one.
"""

from __future__ import annotations

import re

from daralm.data.cleaner import clean_text

# Khmer digits are 10 consecutive code points starting at U+17E0, in the
# same 0-9 order as Arabic numerals — a direct positional mapping.
_KHMER_DIGITS = "០១២៣៤៥៦៧៨៩"
_ARABIC_DIGITS = "0123456789"

_KHMER_TO_ARABIC = str.maketrans(_KHMER_DIGITS, _ARABIC_DIGITS)
_ARABIC_TO_KHMER = str.maketrans(_ARABIC_DIGITS, _KHMER_DIGITS)

# The Khmer sentence/clause punctuation actually observed in this
# project's own corpus (counted, not assumed — see module docstring).
# Excludes the rare Khmer Symbols block (U+19E0-19FF, lunar-calendar
# notation) — those appeared only a handful of times across 2,000 sampled
# documents and aren't sentence punctuation.
_KHMER_SENTENCE_PUNCTUATION = "។៕៖៚៘"
_SPACE_BEFORE_KHMER_PUNCT_RE = re.compile(rf"[ \t]+([{_KHMER_SENTENCE_PUNCTUATION}])")


def khmer_digits_to_arabic(text: str) -> str:
    """Convert Khmer digit glyphs (០-៩) to Arabic numerals (0-9).

    A direct character-level translation — Khmer digits are positional
    decimal digits with no separate "value" to parse, so no numeric
    conversion logic is needed beyond the character mapping itself.
    """
    return text.translate(_KHMER_TO_ARABIC)


def arabic_digits_to_khmer(text: str) -> str:
    """Convert Arabic numerals (0-9) to Khmer digit glyphs (០-៩).

    The reverse of `khmer_digits_to_arabic` — useful when generating
    Khmer-styled output (e.g. a Khmer-language date or count) rather than
    normalizing input for the model.
    """
    return text.translate(_ARABIC_TO_KHMER)


def normalize_khmer_punctuation_spacing(text: str) -> str:
    """Remove whitespace immediately before Khmer sentence punctuation.

    Only removes space *directly preceding* one of `_KHMER_SENTENCE_PUNCTUATION`
    — does not touch spacing after punctuation, which is a separate (and
    less clear-cut) convention question not addressed here.
    """
    return _SPACE_BEFORE_KHMER_PUNCT_RE.sub(r"\1", text)


def normalize_khmer_text(text: str, digits: str = "keep") -> str:
    """The full normalization pipeline: `clean_text` (Unicode/HTML/whitespace,
    already covers everything the module docstring lists as pre-existing)
    plus the two genuinely-new Khmer-specific steps above.

    Args:
        text: raw input text.
        digits: `"keep"` (default, no digit conversion), `"arabic"`
            (convert Khmer digits to Arabic numerals — useful before
            tokenization/inference, since the tokenizer was trained on a
            corpus where digit style varies by source rather than a
            single normalized form), or `"khmer"` (convert Arabic
            numerals to Khmer digits — for generating Khmer-styled text).

    Raises:
        ValueError: if `digits` isn't one of the three recognized values
            — fails loudly rather than silently ignoring a typo'd option,
            the same discipline `ModelConfig` applies to config fields.
    """
    if digits not in ("keep", "arabic", "khmer"):
        raise ValueError(f"digits must be 'keep', 'arabic', or 'khmer', got {digits!r}")

    text = clean_text(text)
    text = normalize_khmer_punctuation_spacing(text)
    if digits == "arabic":
        text = khmer_digits_to_arabic(text)
    elif digits == "khmer":
        text = arabic_digits_to_khmer(text)
    return text
