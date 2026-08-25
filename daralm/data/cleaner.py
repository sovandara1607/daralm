"""Text cleaning and language detection for raw DaraLM documents.

Every function here is a pure text -> text (or text -> value) transform with
no side effects, so each one is independently unit-testable and the pipeline
in `clean_text` is just their composition. `daralm.data.preprocessing` is the
layer that decides, using these building blocks, whether a document should be
kept at all.

Important: we deliberately do NOT strip punctuation. Punctuation is useful
training signal for a language model (sentence boundaries, question marks,
Khmer's own punctuation like "។" khan/full stop) — see project spec section 6.

Khmer-specific note: Khmer is written in its own Unicode block (U+1780-U+17FF,
plus the Khmer Symbols block U+19E0-U+19FF) with no letter case, no spaces
between words within a sentence, and its own digits and punctuation. English-
oriented text-cleaning heuristics (case-folding, whitespace-based
word-splitting, Latin-only regexes) do not transfer to it, so language
detection here is script-based rather than borrowing English assumptions.
"""

from __future__ import annotations

import html
import re
import unicodedata
from typing import Literal

# --- Unicode ranges -----------------------------------------------------

_KHMER_RANGES = ((0x1780, 0x17FF), (0x19E0, 0x19FF))

# --- Regexes (compiled once) --------------------------------------------

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_INTRA_LINE_WHITESPACE_RE = re.compile(r"[ \t ]+")
_EXCESS_BLANK_LINES_RE = re.compile(r"\n{3,}")
_REPEATED_CHAR_RE = re.compile(r"(.)\1{4,}")

# Language-detection thresholds. Tuned for a script-based split between
# Khmer and English/Latin text, not a general-purpose language identifier.
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
    """Normalize to NFC and drop control/format/surrogate/unassigned code points.

    NFC (canonical composition) ensures visually-identical text compares and
    tokenizes consistently regardless of how it was originally encoded (this
    matters for Khmer in particular, where some sequences of combining marks
    have more than one valid encoding). Stripping Unicode category "C*"
    characters (Cc control, Cf format, Cs surrogate, Co private-use, Cn
    unassigned) removes non-printable garbage while explicitly preserving
    "\\n" and "\\t", which are structurally meaningful.
    """
    text = unicodedata.normalize("NFC", text)
    return "".join(ch for ch in text if ch in "\n\t" or unicodedata.category(ch)[0] != "C")


def corruption_ratio(text: str) -> float:
    """Fraction of `text` that is the Unicode replacement character (U+FFFD).

    A high ratio means the source bytes were decoded with the wrong encoding
    (mojibake) and replaced during decoding — a strong signal the document is
    corrupted and should be dropped rather than "cleaned", since there's no
    way to recover the original characters.
    """
    if not text:
        return 0.0
    return text.count("�") / len(text)


def strip_html(text: str) -> str:
    """Unescape HTML entities and remove tags.

    Raw scrapes (including, as it turns out, Wikipedia's own "text" field)
    routinely contain leftover markup like `<div ...>` — this is a light
    regex-based stripper, not a full HTML parser, which is sufficient for
    stripping tags without needing an extra dependency.
    """
    text = html.unescape(text)
    return _HTML_TAG_RE.sub(" ", text)


def strip_urls(text: str) -> str:
    """Replace http(s)/www URLs with a space. Opt-in, not part of `clean_text`.

    URLs are valid text and can carry signal, so they're kept by default;
    this is exposed separately for callers (e.g. quality filtering) that
    specifically need to reason about URL-heavy documents.
    """
    return _URL_RE.sub(" ", text)


def url_density(text: str) -> float:
    """Fraction of `text`'s characters that are part of a URL."""
    if not text:
        return 0.0
    matched_chars = sum(len(m) for m in _URL_RE.findall(text))
    return matched_chars / len(text)


def collapse_whitespace(text: str) -> str:
    """Collapse runs of spaces/tabs to one space and 3+ blank lines to one.

    Paragraph breaks (a single blank line) are preserved on purpose —
    only *excessive* blank-line runs are collapsed.
    """
    text = _INTRA_LINE_WHITESPACE_RE.sub(" ", text)
    text = _EXCESS_BLANK_LINES_RE.sub("\n\n", text)
    lines = [line.strip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def collapse_repeated_chars(text: str, max_repeat: int = 4) -> str:
    """Collapse runs of the same character longer than `max_repeat`.

    Catches spam/garbage like "!!!!!!!!!!!!" or "aaaaaaaaaaaa" while still
    allowing normal emphatic punctuation ("!!!!") through untouched.
    """
    return _REPEATED_CHAR_RE.sub(lambda m: m.group(1) * max_repeat, text)


def clean_text(text: str) -> str:
    """Full cleaning pipeline: unicode normalize -> strip HTML -> collapse whitespace/repeats.

    Order matters: HTML entities must be unescaped before whitespace
    collapsing (entities can decode to whitespace), and repeated-char
    collapsing runs last so it operates on already-normalized text.
    Punctuation is never stripped — see module docstring.
    """
    text = normalize_unicode(text)
    text = strip_html(text)
    text = collapse_whitespace(text)
    text = collapse_repeated_chars(text)
    return text


def detect_language(text: str) -> Literal["km", "en", "mixed", "unknown"]:
    """Classify `text` as Khmer, English, mixed, or unknown, by script ratio.

    Khmer and Latin scripts don't overlap in Unicode, which makes a simple
    character-ratio heuristic reliable here (unlike, say, distinguishing
    English from French, which share a script) — no ML language-ID model or
    extra dependency is needed for this language pair.
    """
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
