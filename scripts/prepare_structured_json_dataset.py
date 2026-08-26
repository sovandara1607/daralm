#!/usr/bin/env python
"""Build the structured JSON generation dataset — Stage 2, item 4 of
`ROADMAP_NLP_PLATFORM.md`, the roadmap's explicit go/no-go gate before
NER, tool calling, or NL-to-SQL are attempted at all.

Fully synthetic, template-generated (`daralm.data.structured_facts`), no
external sourcing — same discipline as grammar correction's synthetic
typos. Each example is an `{"instruction", "response"}` pair —
`InstructionDataset`/`scripts/train_sft.py` need zero changes to train on
this, just a different `--data-dir`.

`response` is a *compact*, single-line `json.dumps` of the fact dict
(no extra whitespace) — deliberately, not pretty-printed: a single line
keeps the target short (cheaper to learn, cleaner `<eos>` placement) and
matches how structured-output APIs conventionally emit JSON in practice.

**v3 — every example's field values are freshly-generated random
strings** (`daralm.data.structured_facts`'s v3 redesign — see that
module's docstring for the full history: v1 shared one small vocabulary
across every split, 100% official / 0% novel-entity; v2 split the
vocabulary into disjoint pools, still 0.00% exact-value-match on 200
*genuinely* held-out examples, ruling out "vocabulary too small" as the
fix). With v3 there is no fixed pool at all — train and test each draw
brand-new random values by construction, so there's nothing left to
accidentally share between them and no vocab-split bookkeeping needed.

Usage:
    python scripts/prepare_structured_json_dataset.py --n-train 12000
"""

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

    # Distinct seeds per split (not the same seed re-sliced) so val/test
    # aren't just a prefix of train's own sequence repeated — matches the
    # independence grammar correction's split derivation already relies on.
    # No vocab-pool bookkeeping needed anymore (v3): every example's field
    # values are freshly random regardless of which split generates them.
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
