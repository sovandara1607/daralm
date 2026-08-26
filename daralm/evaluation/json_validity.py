"""Structured-generation metrics: valid-JSON rate and schema-match rate —
the two metrics `ROADMAP_NLP_PLATFORM.md` specifies for this capability,
in the order it specifies them: "valid-JSON rate as the *first* metric to
establish before any downstream schema-correctness metric matters at
all." A model that can't reliably produce parseable JSON has no chance of
being schema-correct, so these are reported as two separate, ordered
numbers, not folded into one score.
"""

from __future__ import annotations

import json
from typing import Any


def is_valid_json(text: str) -> bool:
    """Does `text` parse as JSON at all (any JSON value, not just objects)?"""
    try:
        json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return False
    return True


def parse_json_or_none(text: str) -> Any | None:
    """Parsed JSON value, or None if `text` isn't valid JSON."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None


def schema_matches(parsed: Any, expected_keys: tuple[str, ...]) -> bool:
    """Does `parsed` look like the target schema: a JSON object with
    exactly `expected_keys` (no missing, no extra) — regardless of value
    correctness. Deliberately weaker than an exact-value match: this
    answers "did it produce the right *shape*", a separate, prerequisite
    question from "did it get the right *values*" (see `exact_value_match`).
    """
    if not isinstance(parsed, dict):
        return False
    return set(parsed.keys()) == set(expected_keys)


def exact_value_match(parsed: Any, expected: dict[str, Any]) -> bool:
    """Does `parsed` have exactly `expected`'s keys and values? The
    strictest of the three checks — schema-correct AND value-correct."""
    if not isinstance(parsed, dict):
        return False
    return parsed == expected
