#!/usr/bin/env python
"""Build the grammar/spelling correction dataset — Stage 2, item 3 of
`ROADMAP_NLP_PLATFORM.md`, "the lowest-friction dataset on this whole list."

No external sourcing needed: every example is derived from this project's
own `data/cleaned/{train,val,test}.jsonl` corpus. Each split's clean text
is chunked into sentences (`daralm.data.sentences.split_sentences`), a
sample of sentences is synthetically corrupted
(`daralm.data.corrupt.corrupt_text`), and each (corrupted, clean) pair is
written out as an `{"instruction", "response"}` record — the exact schema
`daralm.data.dataset.InstructionDataset` already consumes, so
`scripts/train_sft.py` needs zero changes to train on this, just a
different `--data-dir`.

Splits are derived split-for-split (grammar train from cleaned train,
grammar val from cleaned val, etc.) — never re-split independently — so no
sentence from a base-pretraining val/test document can leak into this
task's train split.

**Now includes a no-op subset** (`--noop-fraction`, default 0.15): that
fraction of each split's sentences are used *uncorrupted* — instruction
and response are the same clean sentence. Two real reasons, not just "more
data": (1) a real held-out evaluation (README.md, Stage 2 item 3) already
treats the no-op baseline — do nothing, output the input unchanged — as
the benchmark this task has to beat; the model was never actually trained
on a single example of "this input needs no changes," so it had no
opportunity to learn when to leave text alone, only when to change it.
(2) this directly encodes "minimal edits" as a training signal: a model
that's seen both "fix this" and "this is already fine, don't touch it"
examples has a real incentive to change only what's actually wrong,
rather than rewriting on every input out of habit.

Usage:
    python scripts/prepare_grammar_dataset.py --sentences-per-split 3000
"""

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
    """Extract sentences from `records`, corrupt most of them (leaving
    `noop_fraction` uncorrupted), return instruction/response pairs."""
    rng = random.Random(seed)

    all_sentences: list[tuple[str, str]] = []  # (sentence, language)
    for record in records:
        for sentence in split_sentences(record["text"]):
            all_sentences.append((sentence, record.get("language", "unknown")))

    rng.shuffle(all_sentences)
    chosen = all_sentences[:sentences_wanted]

    examples = []
    for i, (sentence, language) in enumerate(chosen):
        # A per-sentence seed derived from the split seed keeps the whole
        # build reproducible (same seed -> byte-identical output) without
        # every sentence sharing one rng.Random instance's exact call
        # sequence, which would make `sentences_wanted` change every
        # sentence's corruption downstream of it.
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
            len(examples), output_path, noop_count, 100 * noop_count / len(examples),
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
