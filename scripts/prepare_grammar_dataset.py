#!/usr/bin/env python
"""Build the grammar/spelling correction dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import random  # noqa: E402

from daralm.data.corrupt import corrupt_text  # noqa: E402
from daralm.data.loader import load_jsonl, save_jsonl  # noqa: E402
from daralm.data.sentences import split_sentences  # noqa: E402
from daralm.utils.logging import get_logger  # noqa: E402
from daralm.utils.seed import set_seed  # noqa: E402

logger = get_logger(__name__)

INSTRUCTION_TEMPLATE = "Fix any spelling and grammar errors in this text: {corrupted}"


def build_split(
    records: list[dict],
    sentences_wanted: int,
    corruption_rate: float,
    noop_fraction: float,
    seed: int,
) -> list[dict]:
    rng = random.Random(seed)

    all_sentences: list[tuple[str, str]] = []  # (sentence, language)
    for record in records:
        for sentence in split_sentences(record["text"]):
            all_sentences.append((sentence, record.get("language", "unknown")))

    rng.shuffle(all_sentences)
    chosen = all_sentences[:sentences_wanted]

    examples = []
    for i, (sentence, language) in enumerate(chosen):
        example_rng = random.Random(seed + i)
        is_noop = example_rng.random() < noop_fraction
        corrupted = (
            sentence
            if is_noop
            else corrupt_text(sentence, corruption_rate=corruption_rate, seed=seed + i)
        )
        examples.append(
            {
                "instruction": INSTRUCTION_TEMPLATE.format(corrupted=corrupted),
                "response": sentence,
                "language": language,
                "source": "synthetic_noop" if is_noop else "synthetic_corruption",
            }
        )
    return examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/cleaned"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/grammar"))
    parser.add_argument("--sentences-per-split", type=int, default=3000)
    parser.add_argument(
        "--val-test-fraction",
        type=float,
        default=0.15,
        help="Fraction of --sentences-per-split to use for val/test splits, "
        "relative to train (val and test each get this fraction of train's count)",
    )
    parser.add_argument("--corruption-rate", type=float, default=0.15)
    parser.add_argument(
        "--noop-fraction",
        type=float,
        default=0.15,
        help="Fraction of each split's sentences left uncorrupted — teaches "
        "'nothing to fix here, leave it alone' as well as 'fix this'.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    split_sizes = {
        "train": args.sentences_per_split,
        "val": int(args.sentences_per_split * args.val_test_fraction),
        "test": int(args.sentences_per_split * args.val_test_fraction),
    }

    summary = {}
    for split_name, wanted in split_sizes.items():
        input_path = args.input_dir / f"{split_name}.jsonl"
        records = load_jsonl(input_path)
        examples = build_split(
            records,
            sentences_wanted=wanted,
            corruption_rate=args.corruption_rate,
            noop_fraction=args.noop_fraction,
            seed=args.seed,
        )
        output_path = args.output_dir / f"instructions_{split_name}.jsonl"
        save_jsonl(examples, output_path)
        noop_count = sum(1 for e in examples if e["source"] == "synthetic_noop")
        logger.info(
            "Wrote %d examples to %s (%d no-op, %.1f%%)",
            len(examples),
            output_path,
            noop_count,
            100 * noop_count / len(examples),
        )
        summary[split_name] = len(examples)

    print()
    print("=== Grammar Dataset Preparation Summary ===")
    print(f"Corruption rate: {args.corruption_rate}")
    print(f"No-op fraction: {args.noop_fraction}")
    for split_name, count in summary.items():
        print(f"{split_name}: {count} examples")


if __name__ == "__main__":
    main()
