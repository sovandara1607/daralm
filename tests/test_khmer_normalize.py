"""Tests for daralm.data.khmer_normalize — the first NLP-platform-roadmap
capability (Khmer text normalization). Khmer Unicode handling is
explicitly called out as needing "extensive tests" in the roadmap this
module implements — every function here is exercised on real script
ranges, not just ASCII placeholders.
"""

from __future__ import annotations

import pytest

from daralm.data.khmer_normalize import (
    arabic_digits_to_khmer,
    khmer_digits_to_arabic,
    normalize_khmer_punctuation_spacing,
    normalize_khmer_text,
)

KHMER_SENTENCE = "កម្ពុជាជាប្រទេសមួយនៅអាស៊ីអាគ្នេយ៍។"


# --- khmer_digits_to_arabic -------------------------------------------


def test_khmer_digits_to_arabic_converts_all_ten_digits():
    assert khmer_digits_to_arabic("០១២៣៤៥៦៧៨៩") == "0123456789"


def test_khmer_digits_to_arabic_in_real_context():
    # "the year 2024" written with Khmer digits, as it appears in real
    # corpus text
    assert khmer_digits_to_arabic("ឆ្នាំ២០២៤") == "ឆ្នាំ2024"


def test_khmer_digits_to_arabic_leaves_non_digit_khmer_text_untouched():
    assert khmer_digits_to_arabic(KHMER_SENTENCE) == KHMER_SENTENCE


def test_khmer_digits_to_arabic_leaves_arabic_digits_untouched():
    assert khmer_digits_to_arabic("already 2024") == "already 2024"


# --- arabic_digits_to_khmer (the reverse direction) --------------------


def test_arabic_digits_to_khmer_converts_all_ten_digits():
    assert arabic_digits_to_khmer("0123456789") == "០១២៣៤៥៦៧៨៩"


def test_digit_conversion_round_trips():
    original = "០១២៣៤៥៦៧៨៩"
    assert arabic_digits_to_khmer(khmer_digits_to_arabic(original)) == original


# --- normalize_khmer_punctuation_spacing --------------------------------


def test_removes_space_before_khan():
    # The exact real-corpus pattern found by sampling data/cleaned/train.jsonl
    assert normalize_khmer_punctuation_spacing("ប្រទេសកម្ពុជា ។") == "ប្រទេសកម្ពុជា។"


def test_removes_space_before_each_recognized_punctuation_mark():
    for mark in "។៕៖៚៘":
        result = normalize_khmer_punctuation_spacing(f"text {mark}")
        assert result == f"text{mark}", f"failed for {mark!r}"


def test_does_not_touch_punctuation_with_no_preceding_space():
    text = "ប្រទេសកម្ពុជា។"
    assert normalize_khmer_punctuation_spacing(text) == text


def test_does_not_touch_spacing_after_punctuation():
    text = "ចប់។ បន្ត"
    assert normalize_khmer_punctuation_spacing(text) == text


def test_does_not_affect_ordinary_latin_punctuation():
    text = "hello , world ."
    assert normalize_khmer_punctuation_spacing(text) == text


# --- normalize_khmer_text (the composed pipeline) -----------------------


def test_normalize_khmer_text_default_keeps_digits_unconverted():
    result = normalize_khmer_text("ឆ្នាំ២០២៤")
    assert "២០២៤" in result


def test_normalize_khmer_text_arabic_mode_converts_khmer_digits():
    result = normalize_khmer_text("ឆ្នាំ២០២៤", digits="arabic")
    assert "2024" in result
    assert "២០២៤" not in result


def test_normalize_khmer_text_khmer_mode_converts_arabic_digits():
    result = normalize_khmer_text("year 2024", digits="khmer")
    assert "២០២៤" in result


def test_normalize_khmer_text_composes_with_existing_cleaning():
    # Reuses daralm.data.cleaner.clean_text — HTML/control-char stripping
    # should still happen, not just the two new Khmer-specific steps.
    result = normalize_khmer_text("<b>hello</b>\x00 world ។")
    assert "<b>" not in result
    assert "\x00" not in result
    assert "world។" in result  # the space-before-punctuation fix also applied


def test_normalize_khmer_text_rejects_invalid_digits_option():
    with pytest.raises(ValueError):
        normalize_khmer_text("test", digits="latin")  # not a recognized option
