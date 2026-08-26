"""Synthetic corruption of clean text — the data source for grammar/spelling
correction (Stage 2, item 3 of `ROADMAP_NLP_PLATFORM.md`), the "lowest
friction dataset on this whole list" per that document's own assessment.

No external dataset needed: `data/cleaned/{train,val,test}.jsonl`'s own
text is the ground truth. Each clean sentence is corrupted with a handful
of character-level typo operations (delete, duplicate, swap-adjacent,
substitute) at a configurable rate; the (corrupted, clean) pair becomes one
training example — framed as an instruction ("fix this text") / response
(the original) pair, reusing `daralm.data.dataset.InstructionDataset`
unchanged (see `scripts/prepare_grammar_dataset.py`).

Known, disclosed limitation: corruption operates on Python string indices
(Unicode codepoints), not grapheme clusters. Khmer text combines a base
consonant with one or more dependent vowel/diacritic codepoints; an
operation landing on one of those combining marks can occasionally produce
a typo that wouldn't occur from an actual Khmer typing mistake (e.g. a
swap that separates a diacritic from its base consonant). Accepted for a
first version rather than fixed — flagged here, not hidden, matching this
project's documented-limitation discipline elsewhere (e.g. the tokenizer's
`<`/`>` coverage gap noted in `MODEL_CARD_INSTRUCT.md`).
"""

from __future__ import annotations

import random

# Substitution pools are deliberately small and script-specific rather than
# "any Unicode character" — a substitution should still look like a
# plausible same-script typo, not an obviously foreign character.
LATIN_SUBSTITUTION_POOL = "abcdefghijklmnopqrstuvwxyz"
KHMER_SUBSTITUTION_POOL = "កខគឃងចឆជឈញដឋឌឍណតថទធនបផពភមយរលវសហឡអ"

_OPERATIONS = ("delete", "duplicate", "swap", "substitute")


def _substitution_pool_for(char: str) -> str:
    """Khmer's Unicode block is U+1780-U+17FF; everything else falls back
    to the Latin pool (English, and any other Latin-script text)."""
    return KHMER_SUBSTITUTION_POOL if "ក" <= char <= "៿" else LATIN_SUBSTITUTION_POOL


def corrupt_word(word: str, rng: random.Random) -> str:
    """Apply exactly one random character-level typo operation to `word`.

    Returns `word` unchanged if it's shorter than 2 characters — too short
    to meaningfully corrupt without just replacing the whole word, which
    wouldn't teach "fix a typo," it would teach "guess a different word."
    """
    if len(word) < 2:
        return word

    op = rng.choice(_OPERATIONS)
    idx = rng.randrange(len(word))

    if op == "delete":
        return word[:idx] + word[idx + 1 :]
    if op == "duplicate":
        return word[: idx + 1] + word[idx] + word[idx + 1 :]
    if op == "swap":
        if idx == len(word) - 1:
            idx -= 1  # the last character has no "next" character to swap with
        chars = list(word)
        chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
        return "".join(chars)
    # substitute
    pool = _substitution_pool_for(word[idx])
    return word[:idx] + rng.choice(pool) + word[idx + 1 :]


def corrupt_text(text: str, corruption_rate: float = 0.15, seed: int | None = None) -> str:
    """Corrupt roughly `corruption_rate` of `text`'s words, independently.

    Words are split on whitespace and rejoined with a single space —
    acceptable here because callers pass single, already-normalized
    sentences (see `scripts/prepare_grammar_dataset.py`), not whole
    documents where original spacing/newlines would matter.

    Args:
        text: Clean source text (the eventual training *response*).
        corruption_rate: Probability, per word, that it gets corrupted.
        seed: For reproducible corruption (e.g. one fixed seed per dataset
            build, so re-running dataset prep with the same inputs gives
            byte-identical output).

    Raises:
        ValueError: if `corruption_rate` is outside [0, 1].
    """
    if not 0.0 <= corruption_rate <= 1.0:
        raise ValueError(f"corruption_rate must be in [0, 1], got {corruption_rate}")

    rng = random.Random(seed)
    words = text.split()
    corrupted = [
        corrupt_word(word, rng) if rng.random() < corruption_rate else word for word in words
    ]
    return " ".join(corrupted)
