"""Loading raw documents into DaraLM's common record schema."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from daralm.utils.logging import get_logger

logger = get_logger(__name__)

REQUIRED_FIELDS = ("text", "language", "source")


def load_jsonl(path: str | Path, required_field: str = "text") -> list[dict[str, Any]]:
    """Read a JSONL file into a list of record dicts."""
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
    """Write records to `path` as JSONL, one JSON object per line."""
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
    """Stream a small sample of a Wikipedia language edition from Hugging Face."""
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
    """Stream a small sample of the Alpaca instruction-tuning dataset (Phase 9)."""
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


def fetch_alpaca_khmer_sample(n_examples: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from datasets import load_dataset  # local import: heavy, only needed here

    logger.info("Streaming saillab/alpaca_khmer_taco (target %d examples)", n_examples)
    stream = load_dataset("saillab/alpaca_khmer_taco", split="train", streaming=True)

    marker = "Response in Khmer:"
    records: list[dict[str, Any]] = []
    for row in stream:
        output = row.get("output") or ""
        if marker not in output:
            continue  # incompletely-translated row — no Khmer response to extract
        response = output.split(marker, 1)[1].strip()

        instruction = (row.get("instruction") or "").strip()
        extra_input = row.get("input")
        if extra_input is not None and str(extra_input).strip().lower() not in ("", "nan"):
            instruction = f"{instruction}\n{str(extra_input).strip()}"

        if not instruction or not response:
            continue
        records.append(
            {
                "instruction": instruction,
                "response": response,
                "language": "km",
                "source": "alpaca_khmer_taco",
            }
        )
        if len(records) >= n_examples:
            break

    manifest_entry = {
        "source": "alpaca_khmer_taco",
        "hf_dataset": "saillab/alpaca_khmer_taco",
        "hf_url": "https://huggingface.co/datasets/saillab/alpaca_khmer_taco",
        "license": "No explicit license tag on the HF dataset page. Treated as inheriting "
        "Alpaca's CC BY-NC 4.0 (non-commercial) — this is a Khmer translation of Alpaca "
        "via the TaCo method, not an independently-licensed dataset. Verify directly with "
        "the dataset author before any commercial use.",
        "language": "km",
        "examples_fetched": len(records),
    }
    logger.info("Fetched %d Khmer instruction examples", len(records))
    return records, manifest_entry
