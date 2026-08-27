#!/usr/bin/env python
"""Evaluate a structured-JSON-generation fine-tune."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from daralm.data.loader import load_jsonl  # noqa: E402
from daralm.evaluation.json_validity import (  # noqa: E402
    exact_value_match,
    is_valid_json,
    parse_json_or_none,
    schema_matches,
)
from daralm.inference.generator import generate_chat  # noqa: E402
from daralm.model.config import ModelConfig  # noqa: E402
from daralm.model.transformer import DaraLMTransformer  # noqa: E402
from daralm.tokenizer.tokenizer import DaraLMTokenizer  # noqa: E402
from daralm.training.checkpoint import load_checkpoint  # noqa: E402
from daralm.utils.device import get_device, get_device_name  # noqa: E402
from daralm.utils.logging import get_logger  # noqa: E402

logger = get_logger(__name__)

SCHEMA_KEYS = ("name", "age", "occupation", "city")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument(
        "--tokenizer", type=Path, default=Path("checkpoints/tokenizer/unigram.model")
    )
    parser.add_argument("--n-examples", type=int, default=200)
    parser.add_argument("--max-new-tokens", type=int, default=60)
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="0.0 = greedy decoding (default) — structured extraction has one "
        "right answer per input, so greedy is the fair evaluation setting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = ModelConfig.from_yaml(args.config)
    tokenizer = DaraLMTokenizer.from_pretrained(args.tokenizer)
    device = get_device()
    logger.info("Using device: %s", get_device_name(device))

    model = DaraLMTransformer(config.architecture, pad_token_id=tokenizer.pad_id)
    load_checkpoint(args.checkpoint, model, tokenizer_path=args.tokenizer)
    model.to(device)
    model.eval()

    examples = load_jsonl(args.data, required_field="instruction")[: args.n_examples]
    logger.info("Evaluating on %d examples from %s", len(examples), args.data)

    valid_count, schema_count, exact_count = 0, 0, 0
    baseline_valid_count, baseline_schema_count = 0, 0

    for i, example in enumerate(examples):
        expected = parse_json_or_none(example["response"])
        generated = generate_chat(
            model,
            tokenizer,
            example["instruction"],
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )

        if is_valid_json(generated):
            valid_count += 1
            parsed = parse_json_or_none(generated)
            if schema_matches(parsed, SCHEMA_KEYS):
                schema_count += 1
            if expected is not None and exact_value_match(parsed, expected):
                exact_count += 1

        if is_valid_json("{}"):
            baseline_valid_count += 1
        if schema_matches(parse_json_or_none("{}"), SCHEMA_KEYS):
            baseline_schema_count += 1

        if i < 5:
            logger.info(
                "Example %d\n  instruction: %s\n  generated:   %s\n  expected:    %s",
                i,
                example["instruction"],
                generated,
                example["response"],
            )

    n = len(examples)
    print()
    print("=== Structured JSON Generation Evaluation ===")
    print(f"Examples evaluated: {n}")
    print(
        f"Valid-JSON rate:        {valid_count / n:.2%}   "
        f"(baseline '{{}}': {baseline_valid_count / n:.2%})"
    )
    print(
        f"Schema-match rate:      {schema_count / n:.2%}   "
        f"(baseline '{{}}': {baseline_schema_count / n:.2%})"
    )
    print(f"Exact-value-match rate: {exact_count / n:.2%}")


if __name__ == "__main__":
    main()
