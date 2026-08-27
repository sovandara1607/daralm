#!/usr/bin/env python
"""Supervised fine-tune DaraLM-50M Base into DaraLM-50M-Instruct (Phase 9)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from daralm.data.dataset import InstructionDataset  # noqa: E402
from daralm.data.loader import load_jsonl  # noqa: E402
from daralm.model.config import ModelConfig  # noqa: E402
from daralm.model.transformer import DaraLMTransformer  # noqa: E402
from daralm.tokenizer.tokenizer import DaraLMTokenizer  # noqa: E402
from daralm.training.checkpoint import load_checkpoint  # noqa: E402
from daralm.training.trainer import Trainer  # noqa: E402
from daralm.utils.device import get_device, get_device_name  # noqa: E402
from daralm.utils.logging import get_logger  # noqa: E402
from daralm.utils.seed import set_seed  # noqa: E402

logger = get_logger(__name__)

# Instruction examples use shorter blocks than base pretraining.
DEFAULT_BLOCK_SIZE = 512


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="SFT config YAML")
    parser.add_argument(
        "--base-checkpoint",
        required=True,
        type=Path,
        help="Base checkpoint dir to init weights from (e.g. checkpoints/daralm-50m/best)",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("checkpoints/tokenizer/unigram.model"),
        help="Trained tokenizer .model file (must be the same one Base was trained with)",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/cleaned"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = ModelConfig.from_yaml(args.config)
    set_seed(config.training.seed)
    logger.info("Loaded SFT config for %s (seed=%d)", config.model_name, config.training.seed)

    if not args.tokenizer.exists():
        raise FileNotFoundError(f"Tokenizer not found: {args.tokenizer}")
    tokenizer = DaraLMTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.vocab_size != config.architecture.vocab_size:
        raise ValueError(
            f"Tokenizer vocab_size ({tokenizer.vocab_size}) does not match "
            f"{args.config}'s architecture.vocab_size ({config.architecture.vocab_size})."
        )

    train_records = load_jsonl(
        args.data_dir / "instructions_train.jsonl", required_field="instruction"
    )
    val_records = load_jsonl(args.data_dir / "instructions_val.jsonl", required_field="instruction")
    train_dataset = InstructionDataset(train_records, tokenizer, block_size=args.block_size)
    val_dataset = InstructionDataset(val_records, tokenizer, block_size=args.block_size)
    logger.info(
        "Instruction dataset: train=%d examples, val=%d examples, block_size=%d",
        len(train_dataset),
        len(val_dataset),
        args.block_size,
    )

    device = get_device()
    logger.info("Using device: %s", get_device_name(device))

    model = DaraLMTransformer(config.architecture, pad_token_id=tokenizer.pad_id)
    logger.info("Model parameters: %s", f"{model.num_parameters():,}")

    load_checkpoint(args.base_checkpoint, model, tokenizer_path=args.tokenizer)
    logger.info("Initialized weights from Base checkpoint: %s", args.base_checkpoint)

    trainer = Trainer(
        config=config,
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        tokenizer_path=args.tokenizer,
    )

    trainer.train()


if __name__ == "__main__":
    main()
