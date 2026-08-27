#!/usr/bin/env python
"""Generate text from a trained DaraLM checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from daralm.inference.generator import generate  # noqa: E402
from daralm.model.config import ModelConfig  # noqa: E402
from daralm.model.transformer import DaraLMTransformer  # noqa: E402
from daralm.tokenizer.tokenizer import DaraLMTokenizer  # noqa: E402
from daralm.training.checkpoint import load_checkpoint  # noqa: E402
from daralm.utils.device import get_device  # noqa: E402
from daralm.utils.logging import get_logger  # noqa: E402

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path, help="Checkpoint directory")
    parser.add_argument(
        "--tokenizer", type=Path, default=Path("checkpoints/tokenizer/unigram.model")
    )
    parser.add_argument("--prompt", required=True, type=str)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    tokenizer = DaraLMTokenizer.from_pretrained(args.tokenizer)

    # The checkpoint config defines the model architecture.
    config_path = args.checkpoint / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"No config.yaml found in {args.checkpoint} — is this a valid checkpoint directory "
            "produced by scripts/train.py?"
        )
    config = ModelConfig.from_yaml(config_path)

    device = get_device()
    model = DaraLMTransformer(config.architecture, pad_token_id=tokenizer.pad_id)
    load_checkpoint(args.checkpoint, model, tokenizer_path=args.tokenizer)
    model.to(device)
    model.eval()

    logger.info(
        "Generating from %s (temperature=%.2f, top_k=%d, top_p=%.2f, repetition_penalty=%.2f)",
        args.checkpoint,
        args.temperature,
        args.top_k,
        args.top_p,
        args.repetition_penalty,
    )
    text = generate(
        model,
        tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )
    print()
    print(text)


if __name__ == "__main__":
    main()
