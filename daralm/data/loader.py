"""Loading raw documents into DaraLM's common record schema.

Every record — regardless of where it came from — is normalized to:

    {"text": "...", "language": "km" | "en" | ..., "source": "source_name"}

stored one JSON object per line (JSONL). This module has two jobs: reading
and writing that JSONL format, and pulling small public samples of raw text
from Hugging Face to seed `data/raw/` (see `fetch_wikipedia_sample`).

Malformed individual JSONL lines are skipped with a logged warning rather
than aborting the whole load — one bad line in a 10,000-line file shouldn't
discard the other 9,999 — but the skip is never silent (see spec section 29,
"fail loudly": the caller can see exactly how many records were dropped).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from daralm.utils.logging import get_logger

logger = get_logger(__name__)

REQUIRED_FIELDS = ("text", "language", "source")


def load_jsonl(path: str | Path, required_field: str = "text") -> list[dict[str, Any]]:
    """Read a JSONL file into a list of record dicts.

    Raises `FileNotFoundError` if `path` doesn't exist. Individual lines that
    are malformed JSON, or valid JSON missing `required_field`, are skipped
    with a warning rather than aborting the whole load. `required_field`
    defaults to `"text"` (Phase 1 base-pretraining record schema); Phase 9's
    instruction records use `"instruction"` instead — pass it explicitly
    rather than adding a second loader function for the same file format.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")

    records: list[dict[str, Any]] = []
    skipped = 0
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed JSON at %s:%d", path, lineno)
                skipped += 1
                continue
            if required_field not in record:
                logger.warning(
                    "Skipping record missing '%s' field at %s:%d", required_field, path, lineno
                )
                skipped += 1
                continue
            records.append(record)

    if skipped:
        logger.warning("Skipped %d invalid record(s) while loading %s", skipped, path)
    return records


def save_jsonl(records: list[dict[str, Any]], path: str | Path) -> None:
    """Write records to `path` as JSONL, one JSON object per line.

    Creates parent directories if needed. `ensure_ascii=False` so Khmer text
    is written as literal UTF-8 characters, not `\\uXXXX` escapes — keeps the
    files human-readable for spot-checking.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def fetch_wikipedia_sample(
    language: str,
    n_docs: int,
    min_chars: int = 200,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Stream a small sample of a Wikipedia language edition from Hugging Face.

    Uses `datasets` in streaming mode so we never download a full Wikipedia
    dump — only the first `n_docs` articles (after a minimum-length filter to
    skip near-empty stub pages) are pulled over the network.

    Args:
        language: Wikipedia language code, e.g. "km" or "en".
        n_docs: Number of documents to collect.
        min_chars: Skip articles shorter than this (raw, pre-cleaning) —
            filters out near-empty Wikipedia stub pages before we even
            bother running them through the cleaning pipeline.

    Returns:
        A tuple of (records, manifest_entry). `records` are in DaraLM's
        common schema. `manifest_entry` documents exactly what was fetched
        — dataset id, config, license, and count — so the source is always
        traceable (spec section 30: never download data without recording
        its source/license).
    """
    from datasets import load_dataset  # local import: heavy, only needed here

    config = f"20231101.{language}"
    logger.info("Streaming wikimedia/wikipedia config=%s (target %d docs)", config, n_docs)
    stream = load_dataset("wikimedia/wikipedia", config, split="train", streaming=True)

    records: list[dict[str, Any]] = []
    for row in stream:
        text = (row.get("text") or "").strip()
        if len(text) < min_chars:
            continue
        records.append({"text": text, "language": language, "source": f"wikipedia-{language}"})
        if len(records) >= n_docs:
            break

    manifest_entry = {
        "source": f"wikipedia-{language}",
        "hf_dataset": "wikimedia/wikipedia",
        "hf_config": config,
        "hf_url": "https://huggingface.co/datasets/wikimedia/wikipedia",
        "license": "CC BY-SA 4.0 / GFDL (Wikimedia Foundation)",
        "language": language,
        "documents_fetched": len(records),
        "min_chars_filter": min_chars,
    }
    logger.info("Fetched %d documents for language=%s", len(records), language)
    return records, manifest_entry


def fetch_alpaca_sample(n_examples: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Stream a small sample of the Alpaca instruction-tuning dataset (Phase 9).

    A different record schema from `fetch_wikipedia_sample`'s — instruction
    data is `{"instruction", "response", "language", "source"}`, not
    `{"text", "language", "source"}` — since instruction-tuning needs the
    prompt/response boundary preserved, not a single blob of text.

    Only examples with an empty `input` field are kept: Alpaca's `input` is
    optional extra context appended to some instructions (e.g. a passage to
    summarize); folding it in would need its own place in the chat template,
    which adds a real design question (spec section 25's template has no
    slot for it) — dropping those examples is a scope cut, not a bug.
    """
    from datasets import load_dataset  # local import: heavy, only needed here

    logger.info("Streaming tatsu-lab/alpaca (target %d examples)", n_examples)
    stream = load_dataset("tatsu-lab/alpaca", split="train", streaming=True)

    records: list[dict[str, Any]] = []
    for row in stream:
        if (row.get("input") or "").strip():
            continue  # skip examples that need extra context we don't template for
        instruction = (row.get("instruction") or "").strip()
        response = (row.get("output") or "").strip()
        if not instruction or not response:
            continue
        records.append(
            {
                "instruction": instruction,
                "response": response,
                "language": "en",
                "source": "alpaca",
            }
        )
        if len(records) >= n_examples:
            break

    manifest_entry = {
        "source": "alpaca",
        "hf_dataset": "tatsu-lab/alpaca",
        "hf_url": "https://huggingface.co/datasets/tatsu-lab/alpaca",
        "license": "CC BY-NC 4.0 (non-commercial — this project is educational/portfolio "
        "use, never commercial; swap this source before any commercial use)",
        "language": "en",
        "examples_fetched": len(records),
    }
    logger.info("Fetched %d instruction examples", len(records))
    return records, manifest_entry
