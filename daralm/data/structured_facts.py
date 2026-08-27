from __future__ import annotations

import random
from typing import Any, Literal

SCHEMA_KEYS = ("name", "age", "occupation", "city")
_MIN_AGE, _MAX_AGE = 18, 90


_EN_ONSETS = [
    "b",
    "c",
    "d",
    "f",
    "g",
    "h",
    "j",
    "k",
    "l",
    "m",
    "n",
    "p",
    "r",
    "s",
    "t",
    "v",
    "w",
    "z",
    "ch",
    "sh",
    "th",
    "br",
    "cr",
    "dr",
    "fr",
    "gr",
    "pr",
    "tr",
    "bl",
    "cl",
    "fl",
    "gl",
    "pl",
    "sl",
]
_EN_NUCLEI = ["a", "e", "i", "o", "u", "ai", "ea", "oo", "ou", "ie"]
_EN_CODAS = ["", "", "n", "r", "s", "t", "d", "l", "m", "k", "p", "ng", "sh", "th"]

# Khmer base consonants and dependent vowel signs.
_KM_CONSONANTS = [
    "ក",
    "ខ",
    "គ",
    "ឃ",
    "ង",
    "ច",
    "ឆ",
    "ជ",
    "ឈ",
    "ញ",
    "ដ",
    "ឋ",
    "ឌ",
    "ឍ",
    "ណ",
    "ត",
    "ថ",
    "ទ",
    "ធ",
    "ន",
    "ប",
    "ផ",
    "ព",
    "ភ",
    "ម",
    "យ",
    "រ",
    "ល",
    "វ",
    "ស",
    "ហ",
    "អ",
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
    if language not in ("en", "km"):
        raise ValueError(f"language must be 'en' or 'km', got {language!r}")
    fact = _random_fact(rng, language)
    return {"text": _render_text(fact, language), "json": fact, "language": language}
