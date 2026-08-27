from __future__ import annotations

import random

import pytest

from daralm.data.structured_facts import SCHEMA_KEYS, generate_fact_example


def test_rejects_unknown_language():
    with pytest.raises(ValueError):
        generate_fact_example(random.Random(0), "fr")


def test_example_has_expected_top_level_keys():
    example = generate_fact_example(random.Random(0), "en")
    assert set(example.keys()) == {"text", "json", "language"}


def test_json_has_exactly_the_schema_keys():
    for language in ("en", "km"):
        example = generate_fact_example(random.Random(1), language)
        assert set(example["json"].keys()) == set(SCHEMA_KEYS)


def test_json_field_types():
    example = generate_fact_example(random.Random(2), "en")
    assert isinstance(example["json"]["name"], str)
    assert isinstance(example["json"]["age"], int)
    assert isinstance(example["json"]["occupation"], str)
    assert isinstance(example["json"]["city"], str)


def test_age_is_in_a_plausible_range():
    rng = random.Random(3)
    for _ in range(50):
        example = generate_fact_example(rng, "en")
        assert 18 <= example["json"]["age"] <= 90


def test_english_text_contains_every_field_value():
    example = generate_fact_example(random.Random(4), "en")
    text, fact = example["text"], example["json"]
    assert fact["name"] in text
    assert fact["occupation"] in text
    assert fact["city"] in text
    assert str(fact["age"]) in text


def test_khmer_text_contains_every_field_value():
    example = generate_fact_example(random.Random(5), "km")
    text, fact = example["text"], example["json"]
    assert fact["name"] in text
    assert fact["occupation"] in text
    assert fact["city"] in text
    assert str(fact["age"]) in text


def test_language_field_matches_requested_language():
    for language in ("en", "km"):
        example = generate_fact_example(random.Random(6), language)
        assert example["language"] == language


def test_deterministic_given_same_rng_state():
    a = generate_fact_example(random.Random(42), "en")
    b = generate_fact_example(random.Random(42), "en")
    assert a == b


def test_varies_across_calls():
    rng = random.Random(7)
    examples = {generate_fact_example(rng, "en")["text"] for _ in range(20)}
    assert len(examples) > 1


def test_random_values_are_effectively_never_repeated():
    rng = random.Random(9)
    names, occupations, cities = set(), set(), set()
    n = 300
    for _ in range(n):
        fact = generate_fact_example(rng, "en")["json"]
        names.add(fact["name"])
        occupations.add(fact["occupation"])
        cities.add(fact["city"])
    assert len(names) > n * 0.9
    assert len(occupations) > n * 0.5  # single random words collide a bit more than two-word names
    assert len(cities) > n * 0.5


def test_khmer_random_values_are_effectively_never_repeated():
    rng = random.Random(10)
    names, occupations, cities = set(), set(), set()
    n = 300
    for _ in range(n):
        fact = generate_fact_example(rng, "km")["json"]
        names.add(fact["name"])
        occupations.add(fact["occupation"])
        cities.add(fact["city"])
    assert len(names) > n * 0.7
    assert len(occupations) > n * 0.3
    assert len(cities) > n * 0.3
