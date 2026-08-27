"""Tests for daralm.data.corpus_qa.extract_definitional_qa."""

from __future__ import annotations

from daralm.data.corpus_qa import extract_definitional_qa


def test_extracts_simple_english_definition():
    text = "Bakassi is a peninsula on the Gulf of Guinea. More text follows here for length."
    result = extract_definitional_qa(text, "en")
    assert result is not None
    assert result["subject"] == "Bakassi"
    assert result["question"] == "What is Bakassi?"
    assert result["answer"].startswith("Bakassi is a peninsula")


def test_extracts_english_definition_with_parenthetical():
    text = (
        "Hollie Smith (born 17 November 1982) is a New Zealand soul singer-songwriter "
        "based in Auckland, New Zealand. More text follows for length reasons."
    )
    result = extract_definitional_qa(text, "en")
    assert result is not None
    assert "Hollie Smith" in result["subject"]


def test_extracts_khmer_definition():
    text = "អង្គកែវ គឺជាមន្ត្រីមួយរូបក្នុងរជ្ជកាលរបស់ព្រះកេតុមាលា។ អត្ថបទបន្ថែមទៀតនៅទីនេះសម្រាប់ប្រវែង។"
    result = extract_definitional_qa(text, "km")
    assert result is not None
    assert result["subject"] == "អង្គកែវ"
    assert "ជាអ្វី?" in result["question"]


def test_returns_none_for_non_definitional_text():
    text = "Yesterday it rained heavily across the entire region for many hours straight."
    result = extract_definitional_qa(text, "en")
    assert result is None


def test_returns_none_for_empty_text():
    assert extract_definitional_qa("", "en") is None


def test_returns_none_for_too_short_text():
    assert extract_definitional_qa("Hi.", "en") is None


def test_returns_none_when_subject_is_implausibly_long():
    text = (
        "This is a very long and unusually verbose opening clause that goes on and on "
        "before finally getting to the point and is technically still one sentence here."
    )
    result = extract_definitional_qa(text, "en")
    assert result is None or len(result["subject"]) <= 60


def test_answer_is_verbatim_from_source_text():
    text = (
        "The economy of Botswana is currently one of the fastest growing. "
        "Padding text follows here."
    )
    result = extract_definitional_qa(text, "en")
    assert result is not None
    assert result["answer"] in text
