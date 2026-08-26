"""Corpus-grounded QA pair extraction — real question/answer pairs derived
directly from `data/cleaned/*.jsonl`, the same corpus this project's
models were pretrained on.

Why this exists: three separate SFT attempts this session (grammar
correction, structured JSON, instruction-following) each showed the same
pattern — perplexity improved with more/better training data, but real
generation quality did not meaningfully improve. The likely reason isn't
the SFT data itself: at ~29M pretraining tokens, this model has nowhere
near enough exposure to have reliably learned broad world knowledge
(Alpaca-style instructions assume general trivia — "what is the capital
of France" — that a corpus this size covers only a handful of times, if
at all). Corpus-grounded QA tests a different, more answerable question:
did the model learn what's actually *in* the text it was pretrained on?
A question built from an article's own opening sentence has a real chance
of being answerable from that same pretraining exposure — closed-book,
not open-book (no context given at inference), the actual test of whether
SFT can surface knowledge the base model already has.

Extraction method: Wikipedia articles conventionally open with a
definitional sentence — "<Subject> is/was <description>" (English) or
"<Subject> គឺជា<description>" (Khmer). `extract_definitional_qa` matches
that pattern on each article's first sentence
(`daralm.data.sentences.split_sentences`) and, when it matches, returns a
(question, answer) pair built directly from real corpus text — never
invented. Real measured match rate on a 500-document sample: ~50% (the
other half don't open with this exact pattern and are skipped, not
forced) — a real, disclosed yield, not assumed to be higher.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from daralm.data.sentences import split_sentences

_EN_DEFINITION_PATTERN = re.compile(r"^(.{2,80}?)(?:\s*\([^)]*\))?\s+(?:is|was|are|were)\s+")
_KM_DEFINITION_PATTERN = re.compile(r"^(.{1,40}?)\s*(?:\([^)]*\))?\s*គឺជា")

_MIN_SUBJECT_CHARS, _MAX_SUBJECT_CHARS = 2, 60


def extract_definitional_qa(text: str, language: Literal["en", "km"]) -> dict[str, Any] | None:
    """Try to build a (subject, question, answer) triple from `text`'s
    opening sentence. Returns None if the text doesn't open with a
    recognizable definitional pattern — not forced, not guessed.
    """
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
