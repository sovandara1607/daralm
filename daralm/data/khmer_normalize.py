from __future__ import annotations

import re

from daralm.data.cleaner import clean_text

# Khmer digits are 10 consecutive code points starting at U+17E0, in the same 0-9 order as Arabic.
_KHMER_DIGITS = "០១២៣៤៥៦៧៨៩"
_ARABIC_DIGITS = "0123456789"

_KHMER_TO_ARABIC = str.maketrans(_KHMER_DIGITS, _ARABIC_DIGITS)
_ARABIC_TO_KHMER = str.maketrans(_ARABIC_DIGITS, _KHMER_DIGITS)

# Khmer sentence and clause punctuation observed in the corpus.
_KHMER_SENTENCE_PUNCTUATION = "។៕៖៚៘"
_SPACE_BEFORE_KHMER_PUNCT_RE = re.compile(rf"[ \t]+([{_KHMER_SENTENCE_PUNCTUATION}])")


def khmer_digits_to_arabic(text: str) -> str:
    """Convert Khmer digit glyphs (០-៩) to Arabic numerals (0-9)."""
    return text.translate(_KHMER_TO_ARABIC)


def arabic_digits_to_khmer(text: str) -> str:
    """Convert Arabic numerals (0-9) to Khmer digit glyphs (០-៩)."""
    return text.translate(_ARABIC_TO_KHMER)


def normalize_khmer_punctuation_spacing(text: str) -> str:
    """Remove whitespace immediately before Khmer sentence punctuation."""
    return _SPACE_BEFORE_KHMER_PUNCT_RE.sub(r"\1", text)


def normalize_khmer_text(text: str, digits: str = "keep") -> str:
    if digits not in ("keep", "arabic", "khmer"):
        raise ValueError(f"digits must be 'keep', 'arabic', or 'khmer', got {digits!r}")

    text = clean_text(text)
    text = normalize_khmer_punctuation_spacing(text)
    if digits == "arabic":
        text = khmer_digits_to_arabic(text)
    elif digits == "khmer":
        text = arabic_digits_to_khmer(text)
    return text
