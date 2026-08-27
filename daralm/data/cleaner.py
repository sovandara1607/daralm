"""Text cleaning and language detection for raw DaraLM documents."""

from __future__ import annotations

import html
import re
import unicodedata
from typing import Literal

# Unicode ranges

_KHMER_RANGES = ((0x1780, 0x17FF), (0x19E0, 0x19FF))


_HTML_TAG_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_INTRA_LINE_WHITESPACE_RE = re.compile(r"[ \t ]+")
_EXCESS_BLANK_LINES_RE = re.compile(r"\n{3,}")
_REPEATED_CHAR_RE = re.compile(r"(.)\1{4,}")

# Language-detection thresholds.
_KHMER_DOMINANT_RATIO = 0.85
_LATIN_DOMINANT_RATIO = 0.85
_MIXED_MIN_RATIO = 0.05


def is_khmer_char(ch: str) -> bool:
    """True if `ch` falls in the Khmer or Khmer Symbols Unicode block."""
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _KHMER_RANGES)


def is_latin_alpha_char(ch: str) -> bool:
    """True if `ch` is an ASCII Latin letter (a-z, A-Z)."""
    return ch.isascii() and ch.isalpha()


def normalize_unicode(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    return "".join(ch for ch in text if ch in "\n\t" or unicodedata.category(ch)[0] != "C")


def corruption_ratio(text: str) -> float:
    """Fraction of `text` that is the Unicode replacement character (U+FFFD)."""
    if not text:
        return 0.0
    return text.count("�") / len(text)


def strip_html(text: str) -> str:
    """Unescape HTML entities and remove tags."""
    text = html.unescape(text)
    return _HTML_TAG_RE.sub(" ", text)


def strip_urls(text: str) -> str:
    """Replace http(s)/www URLs with a space."""
    return _URL_RE.sub(" ", text)


def url_density(text: str) -> float:
    """Fraction of `text`'s characters that are part of a URL."""
    if not text:
        return 0.0
    matched_chars = sum(len(m) for m in _URL_RE.findall(text))
    return matched_chars / len(text)


def collapse_whitespace(text: str) -> str:
    """Collapse runs of spaces/tabs to one space and 3+ blank lines to one."""
    text = _INTRA_LINE_WHITESPACE_RE.sub(" ", text)
    text = _EXCESS_BLANK_LINES_RE.sub("\n\n", text)
    lines = [line.strip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def collapse_repeated_chars(text: str, max_repeat: int = 4) -> str:
    """Collapse runs of the same character longer than `max_repeat`."""
    return _REPEATED_CHAR_RE.sub(lambda m: m.group(1) * max_repeat, text)


def clean_text(text: str) -> str:
    text = normalize_unicode(text)
    text = strip_html(text)
    text = collapse_whitespace(text)
    text = collapse_repeated_chars(text)
    return text


def detect_language(text: str) -> Literal["km", "en", "mixed", "unknown"]:
    """Classify `text` as Khmer, English, mixed, or unknown, by script ratio."""
    khmer_count = sum(1 for ch in text if is_khmer_char(ch))
    latin_count = sum(1 for ch in text if is_latin_alpha_char(ch))
    total = khmer_count + latin_count
    if total == 0:
        return "unknown"

    khmer_ratio = khmer_count / total
    latin_ratio = latin_count / total

    if khmer_ratio >= _KHMER_DOMINANT_RATIO:
        return "km"
    if latin_ratio >= _LATIN_DOMINANT_RATIO:
        return "en"
    if khmer_ratio >= _MIXED_MIN_RATIO and latin_ratio >= _MIXED_MIN_RATIO:
        return "mixed"
    return "unknown"
