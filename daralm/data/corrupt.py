from __future__ import annotations

import random

# Use small, script-specific substitution pools.
LATIN_SUBSTITUTION_POOL = "abcdefghijklmnopqrstuvwxyz"
KHMER_SUBSTITUTION_POOL = "កខគឃងចឆជឈញដឋឌឍណតថទធនបផពភមយរលវសហឡអ"

_OPERATIONS = ("delete", "duplicate", "swap", "substitute")


def _substitution_pool_for(char: str) -> str:
    return KHMER_SUBSTITUTION_POOL if "ក" <= char <= "៿" else LATIN_SUBSTITUTION_POOL


def corrupt_word(word: str, rng: random.Random) -> str:
    """Apply exactly one random character-level typo operation to `word`."""
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
    pool = _substitution_pool_for(word[idx])
    return word[:idx] + rng.choice(pool) + word[idx + 1 :]


def corrupt_text(text: str, corruption_rate: float = 0.15, seed: int | None = None) -> str:
    """Corrupt roughly `corruption_rate` of `text`'s words, independently."""
    if not 0.0 <= corruption_rate <= 1.0:
        raise ValueError(f"corruption_rate must be in [0, 1], got {corruption_rate}")

    rng = random.Random(seed)
    words = text.split()
    corrupted = [
        corrupt_word(word, rng) if rng.random() < corruption_rate else word for word in words
    ]
    return " ".join(corrupted)
