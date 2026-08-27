#!/usr/bin/env python
"""Build the structured JSON generation dataset."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from daralm.data.loader import save_jsonl  # noqa: E402
from daralm.data.structured_facts import generate_fact_example  # noqa: E402
from daralm.utils.logging import get_logger  # noqa: E402
from daralm.utils.seed import set_seed  # noqa: E402

logger = get_logger(__name__)

INSTRUCTION_TEMPLATE = (
    "Extract the following fields as a JSON object with keys "
    '"name", "age", "occupation", "city": {text}'
)


def build_split(n_examples: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    examples = []
    for _ in range(n_examples):
        language = rng.choice(["en", "km"])
        raw = generate_fact_example(rng, language)
        examples.append(
            {
                "instruction": INSTRUCTION_TEMPLATE.format(text=raw["text"]),
                "response": json.dumps(raw["json"], ensure_ascii=False, separators=(", ", ": ")),
                "language": raw["language"],
                "source": "synthetic_structured_facts",
            }
        )
    return examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/structured_json"))
    parser.add_argument("--n-train", type=int, default=12000)
    parser.add_argument("--n-val", type=int, default=1800)
    parser.add_argument("--n-test", type=int, default=1800)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    splits = {
        "train": (args.n_train, args.seed),
        "val": (args.n_val, args.seed + 1),
        "test": (args.n_test, args.seed + 2),
    }

    summary = {}
    for split_name, (n, seed) in splits.items():
        examples = build_split(n, seed)
        output_path = args.output_dir / f"instructions_{split_name}.jsonl"
        save_jsonl(examples, output_path)
        logger.info("Wrote %d examples to %s", len(examples), output_path)
        summary[split_name] = len(examples)

    print()
    print("=== Structured JSON Dataset Preparation Summary ===")
    for split_name, count in summary.items():
        print(f"{split_name}: {count} examples")


if __name__ == "__main__":
    main()
