#!/usr/bin/env python

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from daralm.data.corpus_qa import extract_definitional_qa  # noqa: E402
from daralm.data.loader import load_jsonl, save_jsonl  # noqa: E402
from daralm.utils.logging import get_logger  # noqa: E402

logger = get_logger(__name__)


def build_split(records: list[dict]) -> tuple[list[dict], int]:
    """Extract every real (question, answer) pair this split's documents yield."""
    examples = []
    skipped = 0
    for record in records:
        language = record.get("language", "en")
        if language not in ("en", "km"):
            skipped += 1
            continue
        result = extract_definitional_qa(record["text"], language)
        if result is None:
            skipped += 1
            continue
        examples.append(
            {
                "instruction": result["question"],
                "response": result["answer"],
                "language": language,
                "source": "corpus_grounded_qa",
            }
        )
    return examples, skipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/cleaned"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/corpus_qa"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for split_name in ("train", "val", "test"):
        input_path = args.input_dir / f"{split_name}.jsonl"
        records = load_jsonl(input_path)
        examples, skipped = build_split(records)
        output_path = args.output_dir / f"instructions_{split_name}.jsonl"
        save_jsonl(examples, output_path)
        match_rate = len(examples) / len(records) if records else 0.0
        logger.info(
            "Wrote %d examples to %s (%d/%d documents matched, %.1f%%, %d skipped)",
            len(examples),
            output_path,
            len(examples),
            len(records),
            match_rate * 100,
            skipped,
        )
        summary[split_name] = len(examples)

    print()
    print("=== Corpus-Grounded QA Dataset Preparation Summary ===")
    for split_name, count in summary.items():
        print(f"{split_name}: {count} examples")
    if summary["train"] == 0:
        raise RuntimeError("No QA examples extracted from the train split — check the corpus path.")


if __name__ == "__main__":
    main()
