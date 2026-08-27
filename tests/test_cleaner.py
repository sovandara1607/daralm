"""Tests for daralm.data.cleaner — text cleaning and language detection."""

from __future__ import annotations

from daralm.data.cleaner import (
    clean_text,
    collapse_repeated_chars,
    collapse_whitespace,
    corruption_ratio,
    detect_language,
    normalize_unicode,
    strip_html,
    strip_urls,
    url_density,
)

KHMER_SENTENCE = "កម្ពុជាជាប្រទេសមួយនៅអាស៊ីអាគ្នេយ៍។"
ENGLISH_SENTENCE = "Cambodia is a country in Southeast Asia."
MIXED_SENTENCE = "Cambodia (កម្ពុជា) is a country in Southeast Asia."


def test_normalize_unicode_strips_control_chars_but_keeps_newline_and_tab():
    text = "hello\x00\x07 world\n\ttab"
    result = normalize_unicode(text)
    assert "\x00" not in result
    assert "\x07" not in result
    assert "\n" in result
    assert "\t" in result


def test_normalize_unicode_is_nfc():
    decomposed = "é"
    result = normalize_unicode(decomposed)
    assert result == "é"


def test_corruption_ratio_detects_replacement_chars():
    assert corruption_ratio("clean text") == 0.0
    corrupted = "�" * 5 + "x" * 5
    assert corruption_ratio(corrupted) == 0.5


def test_corruption_ratio_empty_string():
    assert corruption_ratio("") == 0.0


def test_strip_html_removes_tags_and_unescapes_entities():
    html_text = "<div class='x'>Hello &amp; welcome</div>"
    result = strip_html(html_text)
    assert "<div" not in result
    assert "&amp;" not in result
    assert "Hello & welcome" in result


def test_strip_urls_removes_urls():
    text = "Visit https://example.com/page?x=1 for more info."
    result = strip_urls(text)
    assert "https://" not in result
    assert "for more info" in result


def test_url_density():
    assert url_density("no urls here") == 0.0
    text = "https://example.com"
    assert url_density(text) == 1.0


def test_collapse_whitespace_collapses_runs():
    text = "hello    world\n\n\n\n\nfoo"
    result = collapse_whitespace(text)
    assert "    " not in result
    assert "\n\n\n" not in result


def test_collapse_whitespace_preserves_single_blank_line():
    text = "para one\n\npara two"
    assert collapse_whitespace(text) == "para one\n\npara two"


def test_collapse_repeated_chars():
    text = "wow!!!!!!!!!!!! amazing"
    result = collapse_repeated_chars(text)
    assert result == "wow!!!! amazing"


def test_collapse_repeated_chars_leaves_short_runs_alone():
    assert collapse_repeated_chars("!!!!") == "!!!!"


def test_clean_text_does_not_strip_punctuation():
    result = clean_text(KHMER_SENTENCE)
    assert "។" in result  # Khmer full stop


def test_clean_text_on_empty_string():
    assert clean_text("") == ""


def test_clean_text_pipeline_end_to_end():
    messy = "<p>Hello   &amp;   welcome!!!!!!!!!!!</p>\n\n\n\n\nSee https://example.com"
    result = clean_text(messy)
    assert "<p>" not in result
    assert "&amp;" not in result
    assert "   " not in result
    assert "!!!!!!!!!!!" not in result
    assert "https://example.com" in result


def test_detect_language_khmer():
    assert detect_language(KHMER_SENTENCE) == "km"


def test_detect_language_english():
    assert detect_language(ENGLISH_SENTENCE) == "en"


def test_detect_language_mixed():
    assert detect_language(MIXED_SENTENCE) == "mixed"


def test_detect_language_empty_string_is_unknown():
    assert detect_language("") == "unknown"


def test_detect_language_numbers_only_is_unknown():
    assert detect_language("1234567890") == "unknown"
