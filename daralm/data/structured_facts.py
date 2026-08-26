"""Synthetic (text, JSON) pairs — the data source for structured JSON
generation (Stage 2, item 4 of `ROADMAP_NLP_PLATFORM.md`), the roadmap's
explicit go/no-go gate before NER, tool calling, or NL-to-SQL are
attempted at all.

**v3 — field values are now genuinely unbounded random strings, not a
vocabulary pool at all.** v1 shared one small fixed vocabulary across
every split — "held-out" only meant held-out *combinations* of known
values, and the model scored 100% officially while getting 0/30 correct
on genuinely novel entities. v2 fixed the split (train/test drew from
disjoint vocabulary pools, 32-40 items per field) but kept the fields
*closed sets* — a meaningfully larger, but still finite and memorizable,
set of names/occupations/cities. Retrained and re-evaluated on v2: still
0.00% exact-value-match on 200 held-out examples, confirmed by an
independent check. Ruling out "vocabulary too small" as the fix pointed
to something structural: the model was learning to classify into a
memorized set, not to copy text, and no *finite* vocabulary — however
large — proves the difference.

v3 removes the vocabulary entirely for the fields that matter. Names,
occupations, and cities are generated as fresh, phonotactically-plausible
but semantically meaningless random strings (`_random_word`, real
consonant/vowel-cluster patterns per language) — a new value every single
call, train or test. There is no fixed pool to memorize: succeeding on a
held-out example now requires the model to have learned a general copy
operation, not a lookup table, by construction rather than by a train/test
split that a large-enough table could still defeat.

Age remains a plain random integer — never part of any of this, and the
stress tests have consistently shown it transfers correctly regardless.

Schema (fixed, same keys regardless of language — only values localize):

    {"name": str, "age": int, "occupation": str, "city": str}
"""

from __future__ import annotations

import random
from typing import Any, Literal

SCHEMA_KEYS = ("name", "age", "occupation", "city")
_MIN_AGE, _MAX_AGE = 18, 90

# ---------------------------------------------------------------------
# v3: unbounded random-string generation (the default going forward)
# ---------------------------------------------------------------------

# Real English syllable-structure building blocks (onset + nucleus +
# optional coda) — combined randomly to produce pronounceable-looking but
# invented words, not drawn from any fixed word list at all.
_EN_ONSETS = [
    "b", "c", "d", "f", "g", "h", "j", "k", "l", "m", "n", "p", "r", "s", "t", "v", "w", "z",
    "ch", "sh", "th", "br", "cr", "dr", "fr", "gr", "pr", "tr", "bl", "cl", "fl", "gl", "pl", "sl",
]
_EN_NUCLEI = ["a", "e", "i", "o", "u", "ai", "ea", "oo", "ou", "ie"]
_EN_CODAS = ["", "", "n", "r", "s", "t", "d", "l", "m", "k", "p", "ng", "sh", "th"]

# Real Khmer orthographic building blocks: base consonants + dependent
# vowel signs, combined the same way — every combination is a valid
# Khmer syllable shape, but the resulting "word" is invented, not real
# vocabulary.
_KM_CONSONANTS = [
    "ក", "ខ", "គ", "ឃ", "ង", "ច", "ឆ", "ជ", "ឈ", "ញ", "ដ", "ឋ", "ឌ", "ឍ", "ណ",
    "ត", "ថ", "ទ", "ធ", "ន", "ប", "ផ", "ព", "ភ", "ម", "យ", "រ", "ល", "វ", "ស", "ហ", "អ",
]
_KM_VOWEL_SIGNS = ["ា", "ិ", "ី", "ឹ", "ឺ", "ុ", "ូ", "ើ", "ោ", "ៅ", "ែ", "ៃ", "ុំ", "ំ"]


def _random_en_syllable(rng: random.Random) -> str:
    return rng.choice(_EN_ONSETS) + rng.choice(_EN_NUCLEI) + rng.choice(_EN_CODAS)


def _random_en_word(rng: random.Random, min_syllables: int = 2, max_syllables: int = 3) -> str:
    n = rng.randint(min_syllables, max_syllables)
    return "".join(_random_en_syllable(rng) for _ in range(n))


def _random_km_syllable(rng: random.Random) -> str:
    return rng.choice(_KM_CONSONANTS) + rng.choice(_KM_VOWEL_SIGNS)


def _random_km_word(rng: random.Random, min_syllables: int = 1, max_syllables: int = 2) -> str:
    n = rng.randint(min_syllables, max_syllables)
    return "".join(_random_km_syllable(rng) for _ in range(n))


def _random_fact(rng: random.Random, language: Literal["en", "km"]) -> dict[str, Any]:
    if language == "en":
        name = f"{_random_en_word(rng).capitalize()} {_random_en_word(rng).capitalize()}"
        occupation = _random_en_word(rng)
        city = _random_en_word(rng).capitalize()
    else:
        name = f"{_random_km_word(rng)} {_random_km_word(rng)}"
        occupation = _random_km_word(rng)
        city = _random_km_word(rng)
    return {
        "name": name,
        "age": rng.randint(_MIN_AGE, _MAX_AGE),
        "occupation": occupation,
        "city": city,
    }


def _render_text(fact: dict[str, Any], language: Literal["en", "km"]) -> str:
    if language == "en":
        return (
            f"{fact['name']} is a {fact['age']}-year-old {fact['occupation']} "
            f"living in {fact['city']}."
        )
    return f"{fact['name']} គឺជា{fact['occupation']} អាយុ {fact['age']} ឆ្នាំ រស់នៅ{fact['city']}។"


def generate_fact_example(rng: random.Random, language: Literal["en", "km"]) -> dict[str, Any]:
    """One (text, json, language) example with genuinely fresh random
    field values — see module docstring for why this replaced v2's
    vocabulary-pool approach. `json` is a plain dict — callers serialize
    it however their training/eval format needs (e.g. compact single-line
    `json.dumps`).

    No `split` parameter (unlike v2): every call draws brand-new random
    values, so there's no fixed pool for train/test to accidentally
    share — any two calls producing the same name/occupation/city is
    astronomically unlikely by construction, not prevented by bookkeeping.
    """
    if language not in ("en", "km"):
        raise ValueError(f"language must be 'en' or 'km', got {language!r}")
    fact = _random_fact(rng, language)
    return {"text": _render_text(fact, language), "json": fact, "language": language}
