"""Tests for daralm.evaluation.json_validity."""

from __future__ import annotations

from daralm.evaluation.json_validity import (
    exact_value_match,
    is_valid_json,
    parse_json_or_none,
    schema_matches,
)


def test_valid_object_is_valid_json():
    assert is_valid_json('{"a": 1}') is True


def test_valid_non_object_json_is_still_valid_json():
    assert is_valid_json("42") is True
    assert is_valid_json('"just a string"') is True
    assert is_valid_json("[1, 2, 3]") is True


def test_malformed_json_is_invalid():
    assert is_valid_json('{"a": 1') is False
    assert is_valid_json("not json at all") is False
    assert is_valid_json("") is False


def test_trailing_garbage_after_valid_json_is_invalid():
    assert is_valid_json('{"a": 1} extra text') is False


def test_parse_json_or_none_returns_parsed_value():
    assert parse_json_or_none('{"a": 1}') == {"a": 1}


def test_parse_json_or_none_returns_none_for_invalid():
    assert parse_json_or_none("not json") is None


def test_schema_matches_exact_keys():
    assert schema_matches({"name": "x", "age": 1}, ("name", "age")) is True


def test_schema_matches_rejects_missing_key():
    assert schema_matches({"name": "x"}, ("name", "age")) is False


def test_schema_matches_rejects_extra_key():
    assert schema_matches({"name": "x", "age": 1, "extra": True}, ("name", "age")) is False


def test_schema_matches_rejects_non_dict():
    assert schema_matches([1, 2, 3], ("name", "age")) is False
    assert schema_matches("a string", ("name", "age")) is False


def test_schema_matches_is_key_order_independent():
    assert schema_matches({"age": 1, "name": "x"}, ("name", "age")) is True


def test_exact_value_match_true_for_identical_dict():
    expected = {"name": "x", "age": 1}
    assert exact_value_match({"name": "x", "age": 1}, expected) is True


def test_exact_value_match_false_for_wrong_value():
    expected = {"name": "x", "age": 1}
    assert exact_value_match({"name": "x", "age": 2}, expected) is False


def test_exact_value_match_false_for_non_dict():
    assert exact_value_match([1, 2], {"a": 1}) is False
