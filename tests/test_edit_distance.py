"""Tests for daralm.evaluation.edit_distance (CER/WER)."""

from __future__ import annotations

from daralm.evaluation.edit_distance import character_error_rate, word_error_rate


def test_identical_strings_have_zero_cer():
    assert character_error_rate("hello world", "hello world") == 0.0


def test_identical_strings_have_zero_wer():
    assert word_error_rate("hello world", "hello world") == 0.0


def test_cer_counts_single_substitution():
    # "cat" -> "car": 1 substitution / 3 reference chars
    assert character_error_rate("car", "cat") == 1 / 3


def test_cer_counts_single_deletion_needed():
    # hypothesis "ct" vs reference "cat": 1 deletion from hypothesis's
    # perspective (need to insert 'a') / 3 reference chars
    assert character_error_rate("ct", "cat") == 1 / 3


def test_wer_counts_one_wrong_word():
    assert word_error_rate("the quick fox", "the quick brown fox") == 1 / 4


def test_wer_counts_completely_different_sentences():
    hyp = "completely different words entirely"
    ref = "the quick brown fox"
    # every reference word must be replaced (allowing for length mismatch)
    result = word_error_rate(hyp, ref)
    assert result > 0.5


def test_empty_reference_and_empty_hypothesis_is_zero():
    assert character_error_rate("", "") == 0.0
    assert word_error_rate("", "") == 0.0


def test_empty_reference_nonempty_hypothesis_is_one():
    assert character_error_rate("oops", "") == 1.0
    assert word_error_rate("oops", "") == 1.0


def test_cer_can_exceed_one_for_very_wrong_hypothesis():
    # A hypothesis much longer than a short reference can have more edits
    # than the reference has characters.
    result = character_error_rate("xxxxxxxxxxxxxxxxxxxx", "hi")
    assert result > 1.0
