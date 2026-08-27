#!/usr/bin/env python

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from daralm.data.loader import load_jsonl  # noqa: E402
from daralm.evaluation.edit_distance import character_error_rate, word_error_rate  # noqa: E402
from daralm.inference.generator import generate_chat  # noqa: E402
from daralm.model.config import ModelConfig  # noqa: E402
from daralm.model.transformer import DaraLMTransformer  # noqa: E402
from daralm.tokenizer.tokenizer import DaraLMTokenizer  # noqa: E402
from daralm.training.checkpoint import load_checkpoint  # noqa: E402
from daralm.utils.device import get_device, get_device_name  # noqa: E402
from daralm.utils.logging import get_logger  # noqa: E402

logger = get_logger(__name__)


def _extract_corrupted_text(instruction: str) -> str:
    prefix = "Fix any spelling and grammar errors in this text: "
    if not instruction.startswith(prefix):
        raise ValueError(
            f"Unexpected instruction format, can't extract corrupted text: {instruction!r}"
        )
    return instruction[len(prefix) :]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument(
        "--tokenizer", type=Path, default=Path("checkpoints/tokenizer/unigram.model")
    )
    parser.add_argument("--n-examples", type=int, default=200)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="0.0 = greedy decoding (default) — a correction task has one "
        "right answer, so greedy is the fair evaluation setting, not "
        "creative-writing-style sampling.",
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

    model_cer, model_wer, baseline_cer, baseline_wer = [], [], [], []
    exact_matches = 0

    for i, example in enumerate(examples):
        reference = example["response"]
        corrupted = _extract_corrupted_text(example["instruction"])

        generated = generate_chat(
            model,
            tokenizer,
            example["instruction"],
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )

        model_cer.append(character_error_rate(generated, reference))
        model_wer.append(word_error_rate(generated, reference))
        baseline_cer.append(character_error_rate(corrupted, reference))
        baseline_wer.append(word_error_rate(corrupted, reference))
        if generated.strip() == reference.strip():
            exact_matches += 1

        if i < 5:
            logger.info(
                "Example %d\n  corrupted: %s\n  generated: %s\n  reference: %s",
                i,
                corrupted,
                generated,
                reference,
            )

    n = len(examples)
    avg_model_cer = sum(model_cer) / n
    avg_model_wer = sum(model_wer) / n
    avg_baseline_cer = sum(baseline_cer) / n
    avg_baseline_wer = sum(baseline_wer) / n

    passed = avg_model_cer < avg_baseline_cer

    print()
    print("=== Grammar Correction Evaluation ===")
    print(f"Examples evaluated: {n}")
    print(f"Model CER:    {avg_model_cer:.4f}   (no-op baseline: {avg_baseline_cer:.4f})")
    print(f"Model WER:    {avg_model_wer:.4f}   (no-op baseline: {avg_baseline_wer:.4f})")
    print(f"Exact match:  {exact_matches}/{n} ({exact_matches / n:.2%})")
    print()
    if passed:
        print(
            f"PASS — model CER ({avg_model_cer:.4f}) beats the "
            f"no-op baseline ({avg_baseline_cer:.4f})."
        )
    else:
        print(
            f"FAIL — model CER ({avg_model_cer:.4f}) does not beat the "
            f"no-op baseline ({avg_baseline_cer:.4f})."
        )
        print("The model has not demonstrated real correction capability on this held-out set.")

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
