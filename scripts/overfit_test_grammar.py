#!/usr/bin/env python

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from daralm.data.dataset import InstructionDataset  # noqa: E402
from daralm.data.loader import load_jsonl  # noqa: E402
from daralm.evaluation.edit_distance import character_error_rate  # noqa: E402
from daralm.inference.generator import generate_chat  # noqa: E402
from daralm.model.config import ModelConfig  # noqa: E402
from daralm.model.transformer import DaraLMTransformer  # noqa: E402
from daralm.tokenizer.tokenizer import DaraLMTokenizer  # noqa: E402
from daralm.training.checkpoint import load_checkpoint  # noqa: E402
from daralm.training.trainer import Trainer  # noqa: E402
from daralm.utils.device import get_device, get_device_name  # noqa: E402
from daralm.utils.logging import get_logger  # noqa: E402
from daralm.utils.seed import set_seed  # noqa: E402

logger = get_logger(__name__)

# Required CER improvement after overfitting the training set.
MAX_ACCEPTABLE_CER_VS_BASELINE_RATIO = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/50m-grammar.yaml"))
    parser.add_argument("--base-checkpoint", type=Path, default=Path("checkpoints/daralm-50m/best"))
    parser.add_argument(
        "--tokenizer", type=Path, default=Path("checkpoints/tokenizer/unigram.model")
    )
    parser.add_argument("--data", type=Path, default=Path("data/grammar/instructions_train.jsonl"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--n-examples", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=800)
    parser.add_argument("--warmup-steps", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = ModelConfig.from_yaml(args.config)
    config.model_name = f"{config.model_name}-overfit-test"
    config.training = config.training.model_copy(
        update={
            "max_steps": args.max_steps,
            "warmup_steps": args.warmup_steps,
            "eval_interval": max(args.max_steps // 6, 1),
            "save_interval": args.max_steps,  # only save once, at the end
            "log_interval": max(args.max_steps // 15, 1),
        }
    )
    set_seed(config.training.seed)

    tokenizer = DaraLMTokenizer.from_pretrained(args.tokenizer)
    examples = load_jsonl(args.data, required_field="instruction")[: args.n_examples]
    if len(examples) < args.n_examples:
        raise ValueError(f"Only {len(examples)} examples available in {args.data}")
    logger.info("Overfitting %d fixed examples from %s", len(examples), args.data)

    dataset = InstructionDataset(examples, tokenizer, block_size=args.block_size)

    device = get_device()
    logger.info("Using device: %s", get_device_name(device))

    model = DaraLMTransformer(config.architecture, pad_token_id=tokenizer.pad_id)
    load_checkpoint(args.base_checkpoint, model, tokenizer_path=args.tokenizer)
    logger.info("Initialized weights from Base checkpoint: %s", args.base_checkpoint)

    trainer = Trainer(
        config=config,
        model=model,
        train_dataset=dataset,  # same set for both — see module docstring
        val_dataset=dataset,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        tokenizer_path=args.tokenizer,
    )
    trainer.train()

    model.eval()
    model_cer_scores, baseline_cer_scores = [], []
    exact_matches = 0

    print()
    print("=== Memorization check (greedy generation vs. the exact training target) ===")
    for i, example in enumerate(examples):
        reference = example["response"]
        corrupted = example["instruction"].split(": ", 1)[-1]
        generated = generate_chat(model, tokenizer, example["instruction"], temperature=0.0)

        model_cer = character_error_rate(generated, reference)
        baseline_cer = character_error_rate(corrupted, reference)
        model_cer_scores.append(model_cer)
        baseline_cer_scores.append(baseline_cer)
        if generated.strip() == reference.strip():
            exact_matches += 1

        if i < 5:
            print(f"  reference:  {reference!r}")
            print(f"  generated:  {generated!r}")
            print(f"  CER: model={model_cer:.3f} baseline={baseline_cer:.3f}")

    n = len(examples)
    avg_model_cer = sum(model_cer_scores) / n
    avg_baseline_cer = sum(baseline_cer_scores) / n
    ratio = avg_model_cer / avg_baseline_cer if avg_baseline_cer > 0 else float("inf")
    passed = ratio <= MAX_ACCEPTABLE_CER_VS_BASELINE_RATIO

    print()
    print("=== Grammar-Correction Overfit Diagnostic Verdict ===")
    print(f"Examples memorized (attempted): {n}")
    print(f"Exact matches:                  {exact_matches}/{n}")
    print(f"Avg model CER (memorized set):  {avg_model_cer:.4f}")
    print(f"Avg no-op baseline CER:         {avg_baseline_cer:.4f}")
    print(
        f"Model/baseline CER ratio:       {ratio:.2f}  "
        f"(threshold: <= {MAX_ACCEPTABLE_CER_VS_BASELINE_RATIO})"
    )
    print()
    if passed:
        print("PASS — the model CAN overfit a tiny grammar-correction set.")
        print("The real run's failure was likely 'not enough training budget/data',")
        print("not a hard capacity ceiling. More steps/data is a reasonable next bet.")
    else:
        print("FAIL — the model could NOT memorize even a tiny, fixed set.")
        print("This points to a deeper ceiling in the SFT-for-generation approach")
        print("itself (or this task's specific setup) — more budget alone is unlikely")
        print("to fix it. Structured JSON generation should be expected to hit the")
        print("same wall if attempted with this same recipe.")

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
