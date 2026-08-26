"""Tests for daralm.data.corrupt — the synthetic typo generator behind
grammar/spelling correction's training data (Stage 2 of ROADMAP_NLP_PLATFORM.md).
"""

from __future__ import annotations

import random

import pytest

from daralm.data.corrupt import corrupt_text, corrupt_word


def test_corrupt_word_leaves_single_char_words_unchanged():
    rng = random.Random(0)
    assert corrupt_word("a", rng) == "a"
    assert corrupt_word("", rng) == ""


def test_corrupt_word_changes_length_by_at_most_one():
    rng = random.Random(1)
    word = "hello"
    for _ in range(50):
        result = corrupt_word(word, rng)
        assert abs(len(result) - len(word)) <= 1


def test_corrupt_word_is_deterministic_given_same_rng_state():
    result_a = corrupt_word("testing", random.Random(42))
    result_b = corrupt_word("testing", random.Random(42))
    assert result_a == result_b


def test_corrupt_word_produces_variety_across_many_calls():
    # Not every call should produce the identical corruption — otherwise
    # the "random operation" logic isn't actually varying.
    rng = random.Random(7)
    results = {corrupt_word("keyboard", rng) for _ in range(30)}
    assert len(results) > 1


def test_corrupt_text_rejects_out_of_range_corruption_rate():
    with pytest.raises(ValueError):
        corrupt_text("hello world", corruption_rate=1.5)
    with pytest.raises(ValueError):
        corrupt_text("hello world", corruption_rate=-0.1)


def test_corrupt_text_zero_rate_leaves_text_unchanged():
    text = "the quick brown fox jumps"
    assert corrupt_text(text, corruption_rate=0.0, seed=0) == text


def test_corrupt_text_high_rate_changes_most_long_words():
    text = "elephant butterfly chocolate mountain telephone"
    result = corrupt_text(text, corruption_rate=1.0, seed=0)
    original_words = text.split()
    result_words = result.split()
    assert len(original_words) == len(result_words)
    changed = sum(1 for o, r in zip(original_words, result_words, strict=True) if o != r)
    assert changed == len(original_words)  # every word is >= 2 chars, all corrupted


def test_corrupt_text_is_reproducible_with_same_seed():
    text = "the quick brown fox jumps over the lazy dog"
    result_a = corrupt_text(text, corruption_rate=0.5, seed=123)
    result_b = corrupt_text(text, corruption_rate=0.5, seed=123)
    assert result_a == result_b


def test_corrupt_text_differs_with_different_seeds():
    text = "the quick brown fox jumps over the lazy dog and more words here too"
    result_a = corrupt_text(text, corruption_rate=0.5, seed=1)
    result_b = corrupt_text(text, corruption_rate=0.5, seed=2)
    assert result_a != result_b


def test_corrupt_text_preserves_word_count():
    text = "one two three four five six seven"
    result = corrupt_text(text, corruption_rate=0.3, seed=5)
    assert len(result.split()) == len(text.split())


def test_corrupt_text_handles_khmer_script():
    text = "សួស្តី​ពិភពលោក នេះជាការសាកល្បង"
    result = corrupt_text(text, corruption_rate=0.8, seed=9)
    # Just needs to run without crashing and preserve word count — no
    # claim about typo *realism* for Khmer (see module docstring).
    assert len(result.split()) == len(text.split())
