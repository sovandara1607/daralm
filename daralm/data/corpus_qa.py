from __future__ import annotations

import re
from typing import Any, Literal

from daralm.data.sentences import split_sentences

_EN_DEFINITION_PATTERN = re.compile(r"^(.{2,80}?)(?:\s*\([^)]*\))?\s+(?:is|was|are|were)\s+")
_KM_DEFINITION_PATTERN = re.compile(r"^(.{1,40}?)\s*(?:\([^)]*\))?\s*គឺជា")

_MIN_SUBJECT_CHARS, _MAX_SUBJECT_CHARS = 2, 60


def extract_definitional_qa(text: str, language: Literal["en", "km"]) -> dict[str, Any] | None:
    sentences = split_sentences(text, min_chars=20, max_chars=300)
    if not sentences:
        return None

    first_sentence = sentences[0]
    pattern = _EN_DEFINITION_PATTERN if language == "en" else _KM_DEFINITION_PATTERN
    match = pattern.match(first_sentence)
    if not match:
        return None

    subject = match.group(1).strip()
    if not (_MIN_SUBJECT_CHARS <= len(subject) <= _MAX_SUBJECT_CHARS):
        return None

    question = f"What is {subject}?" if language == "en" else f"{subject}ជាអ្វី?"
    return {"subject": subject, "question": question, "answer": first_sentence}
