"""Tests for daralm.data.sentences.split_sentences."""

from __future__ import annotations

import pytest

from daralm.data.sentences import split_sentences


def test_splits_english_sentences_on_terminators():
    text = "This is one sentence. This is another sentence! Is this a third one?"
    result = split_sentences(text, min_chars=5, max_chars=100)
    assert result == [
        "This is one sentence.",
        "This is another sentence!",
        "Is this a third one?",
    ]


def test_splits_khmer_sentences_on_khan():
    text = "នេះជាប្រយោគទីមួយដែលមានប្រវែងគ្រប់គ្រាន់សម្រាប់តេស្តនេះ។ នេះជាប្រយោគទីពីរដែលមានប្រវែងគ្រប់គ្រាន់ដែរ។"
    result = split_sentences(text, min_chars=5, max_chars=200)
    assert len(result) == 2
    assert all(s.endswith("។") for s in result)


def test_drops_chunks_shorter_than_min_chars():
    text = "Hi. This one is long enough to survive the minimum length filter easily."
    result = split_sentences(text, min_chars=20, max_chars=200)
    assert "Hi." not in result
    assert any("long enough" in s for s in result)


def test_drops_chunks_longer_than_max_chars():
    short = "This fits."
    long_sentence = "This one is deliberately way too long. " * 10
    text = f"{short} {long_sentence.strip()}"
    result = split_sentences(text, min_chars=1, max_chars=30)
    assert short in result
    assert not any(len(s) > 30 for s in result)


def test_rejects_non_positive_bounds():
    with pytest.raises(ValueError):
        split_sentences("text", min_chars=0, max_chars=100)
    with pytest.raises(ValueError):
        split_sentences("text", min_chars=100, max_chars=0)


def test_rejects_min_greater_than_max():
    with pytest.raises(ValueError):
        split_sentences("text", min_chars=100, max_chars=10)


def test_empty_text_returns_empty_list():
    assert split_sentences("", min_chars=1, max_chars=100) == []
