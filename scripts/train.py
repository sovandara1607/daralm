#!/usr/bin/env python

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from daralm.data.dataset import PackedTokenDataset  # noqa: E402
from daralm.data.loader import load_jsonl  # noqa: E402
from daralm.model.config import ModelConfig  # noqa: E402
from daralm.model.transformer import DaraLMTransformer  # noqa: E402
from daralm.tokenizer.tokenizer import DaraLMTokenizer  # noqa: E402
from daralm.training.checkpoint import find_latest_checkpoint, load_checkpoint  # noqa: E402
from daralm.training.trainer import Trainer  # noqa: E402
from daralm.utils.device import get_device, get_device_name  # noqa: E402
from daralm.utils.logging import get_logger  # noqa: E402
from daralm.utils.seed import set_seed  # noqa: E402

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Path to a model config YAML")
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("checkpoints/tokenizer/unigram.model"),
        help="Trained tokenizer .model file (default: Phase 2's recommended tokenizer)",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/cleaned"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument(
        "--resume", action="store_true", help="Resume from the latest step-N checkpoint, if any"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = ModelConfig.from_yaml(args.config)
    set_seed(config.training.seed)
    logger.info("Loaded config for %s (seed=%d)", config.model_name, config.training.seed)

    if not args.tokenizer.exists():
        raise FileNotFoundError(
            f"Tokenizer not found: {args.tokenizer}. Run scripts/train_tokenizer.py first "
            "(Phase 2), or pass --tokenizer pointing at a trained .model file."
        )
    tokenizer = DaraLMTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.vocab_size != config.architecture.vocab_size:
        raise ValueError(
            f"Tokenizer vocab_size ({tokenizer.vocab_size}) does not match "
            f"{args.config}'s architecture.vocab_size ({config.architecture.vocab_size}). "
            "Every DaraLM config must share one tokenizer — fix the config or retrain the "
            "tokenizer at the matching vocab size."
        )

    train_records = load_jsonl(args.data_dir / "train.jsonl")
    val_records = load_jsonl(args.data_dir / "val.jsonl")
    block_size = config.architecture.max_position_embeddings
    train_dataset = PackedTokenDataset(train_records, tokenizer, block_size=block_size)
    val_dataset = PackedTokenDataset(val_records, tokenizer, block_size=block_size)
    logger.info(
        "Packed dataset: train=%d blocks (%d tokens), val=%d blocks (%d tokens), block_size=%d",
        len(train_dataset),
        train_dataset.total_tokens,
        len(val_dataset),
        val_dataset.total_tokens,
        block_size,
    )

    device = get_device()
    logger.info("Using device: %s", get_device_name(device))

    model = DaraLMTransformer(config.architecture, pad_token_id=tokenizer.pad_id)
    logger.info("Model parameters: %s", f"{model.num_parameters():,}")

    trainer = Trainer(
        config=config,
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        tokenizer_path=args.tokenizer,
    )

    if args.resume:
        latest = find_latest_checkpoint(args.checkpoint_dir, config.model_name)
        if latest is None:
            logger.info("No checkpoint found for %s; starting from step 0", config.model_name)
        else:
            resumed = load_checkpoint(
                latest,
                trainer.model,
                trainer.optimizer,
                trainer.scheduler,
                tokenizer_path=args.tokenizer,
            )
            trainer.state.step = resumed["step"]
            trainer.state.tokens_processed = resumed["tokens_processed"]
            logger.info("Resumed from %s at step=%d", latest, trainer.state.step)

    trainer.train()


if __name__ == "__main__":
    main()
