#!/usr/bin/env python
"""Evaluate a grammar/spelling-correction fine-tune with real generation +
CER/WER — the metrics `ROADMAP_NLP_PLATFORM.md` recommends for this
capability, since ground truth (the un-corrupted sentence) is directly
known, unlike open-ended generation.

Runs real inference (`daralm.inference.generator.generate_chat`) on a held-
out split, not a loss/perplexity proxy — perplexity says how surprised the
model was by the *correct* continuation; CER/WER says whether what it
actually *generated* matches the correct text, which is the real question
for a correction task.

A no-op baseline (CER/WER of the corrupted input against the reference,
with zero model involvement) is reported alongside the model's score —
without it, a low CER could just mean "the corruption was mild," not "the
model fixed anything." That baseline is now the explicit pass/fail gate,
not just context: a grammar-correction model that can't beat "do nothing"
hasn't demonstrated the capability at all, whatever its own raw CER looks
like in isolation.

Usage:
    python scripts/evaluate_grammar.py \\
        --config configs/50m-grammar.yaml \\
        --checkpoint checkpoints/daralm-50m-grammar/best \\
        --data data/grammar/instructions_test.jsonl
"""

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
    """Pull the corrupted sentence back out of the training instruction
    template ("Fix any spelling and grammar errors in this text: {x}") so
    the no-op baseline can be scored against it directly."""
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
                i, corrupted, generated, reference,
            )

    n = len(examples)
    avg_model_cer = sum(model_cer) / n
    avg_model_wer = sum(model_wer) / n
    avg_baseline_cer = sum(baseline_cer) / n
    avg_baseline_wer = sum(baseline_wer) / n

    # The no-op baseline as an explicit gate, not just reported context —
    # matches overfit_test.py's/overfit_test_grammar.py's pattern of a
    # real pass/fail verdict with an exit code, rather than a printout
    # the reader has to interpret themselves.
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
