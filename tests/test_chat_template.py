"""Tests for daralm.data.chat_template."""

from __future__ import annotations

from daralm.data.chat_template import (
    ASSISTANT_CLOSE,
    ASSISTANT_OPEN,
    USER_CLOSE,
    USER_OPEN,
    format_example,
    format_prompt,
    format_response,
)


def test_format_prompt_contains_markers_and_instruction():
    prompt = format_prompt("What is a database?")
    assert USER_OPEN in prompt
    assert USER_CLOSE in prompt
    assert ASSISTANT_OPEN in prompt
    assert "What is a database?" in prompt
    assert ASSISTANT_CLOSE not in prompt


def test_format_response_contains_closing_marker():
    response = format_response("A database is an organized collection of data.")
    assert "A database is an organized collection of data." in response
    assert ASSISTANT_CLOSE in response
    assert USER_OPEN not in response


def test_format_prompt_strips_whitespace():
    prompt = format_prompt("  What is a database?  \n")
    assert "  What is a database?  " not in prompt
    assert "What is a database?" in prompt


def test_format_example_splits_at_the_masking_boundary():
    prompt_text, response_text = format_example("2+2?", "4.")
    assert "2+2?" in prompt_text
    assert "4." in response_text
    assert "4." not in prompt_text
    assert "2+2?" not in response_text


def test_format_prompt_and_response_concatenate_to_full_template():
    prompt_text, response_text = format_example("Question", "Answer")
    full = prompt_text + response_text
    assert USER_OPEN in full
    assert USER_CLOSE in full
    assert ASSISTANT_OPEN in full
    assert ASSISTANT_CLOSE in full


def test_khmer_instruction_round_trips_through_template():
    prompt = format_prompt("តើទឹកមានធាតុគីមីអ្វីខ្លះ?")
    assert "តើទឹកមានធាតុគីមីអ្វីខ្លះ?" in prompt
