from __future__ import annotations

import json
from typing import Any


def is_valid_json(text: str) -> bool:
    """Does `text` parse as JSON at all (any JSON value, not just objects)?."""
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
    if not isinstance(parsed, dict):
        return False
    return set(parsed.keys()) == set(expected_keys)


def exact_value_match(parsed: Any, expected: dict[str, Any]) -> bool:
    """Does `parsed` have exactly `expected`'s keys and values?."""
    if not isinstance(parsed, dict):
        return False
    return parsed == expected
