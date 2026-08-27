#!/usr/bin/env python

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from daralm.data.dataset import ClassificationDataset  # noqa: E402
from daralm.data.loader import load_jsonl  # noqa: E402
from daralm.model.classification_head import ClassificationHead  # noqa: E402
from daralm.model.config import ModelConfig  # noqa: E402
from daralm.model.transformer import DaraLMTransformer  # noqa: E402
from daralm.tokenizer.tokenizer import DaraLMTokenizer  # noqa: E402
from daralm.training.checkpoint import load_checkpoint  # noqa: E402
from daralm.utils.device import get_device, get_device_name  # noqa: E402
from daralm.utils.logging import get_logger  # noqa: E402
from daralm.utils.seed import set_seed  # noqa: E402

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone-config", required=True, type=Path)
    parser.add_argument("--backbone-checkpoint", required=True, type=Path)
    parser.add_argument(
        "--tokenizer", type=Path, default=Path("checkpoints/tokenizer/unigram.model")
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/cleaned"))
    parser.add_argument(
        "--label-field",
        default="language",
        help="Which field in the cleaned JSONL records to use as the label",
    )
    parser.add_argument("--label-list", nargs="+", default=["en", "km"])
    parser.add_argument(
        "--max-examples-per-split",
        type=int,
        default=2000,
        help="Cap train/val size — this is a pipeline-validation task, not "
        "production training; no need to run every one of 14k+ records "
        "through a frozen backbone to prove the mechanism works.",
    )
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=Path("checkpoints/daralm-50m-classify-language")
    )
    return parser.parse_args()


def _load_records(path: Path, label_field: str, limit: int) -> list[dict]:
    records = load_jsonl(path)
    records = [{"text": r["text"], "label": r[label_field]} for r in records]
    random.Random(42).shuffle(records)
    return records[:limit]


@torch.no_grad()
def evaluate(
    backbone: DaraLMTransformer, head: ClassificationHead, loader: DataLoader, device: torch.device
) -> tuple[float, float]:
    """Returns (avg_loss, accuracy) over the full loader."""
    backbone.eval()
    head.eval()
    total_loss, correct, total = 0.0, 0, 0
    for input_ids, labels in loader:
        input_ids, labels = input_ids.to(device), labels.to(device)
        hidden = backbone(input_ids, return_hidden_states=True).hidden_states
        logits = head(hidden, input_ids, pad_token_id=backbone.pad_token_id)
        loss = F.cross_entropy(logits, labels)
        total_loss += loss.item() * input_ids.size(0)
        correct += (logits.argmax(dim=-1) == labels).sum().item()
        total += input_ids.size(0)
    head.train()
    return total_loss / total, correct / total


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    backbone_config = ModelConfig.from_yaml(args.backbone_config)
    tokenizer = DaraLMTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.vocab_size != backbone_config.architecture.vocab_size:
        raise ValueError(
            f"Tokenizer vocab_size ({tokenizer.vocab_size}) does not match "
            f"{args.backbone_config}'s architecture.vocab_size "
            f"({backbone_config.architecture.vocab_size})."
        )

    device = get_device()
    logger.info("Using device: %s", get_device_name(device))

    backbone = DaraLMTransformer(backbone_config.architecture, pad_token_id=tokenizer.pad_id)
    load_checkpoint(args.backbone_checkpoint, backbone, tokenizer_path=args.tokenizer)
    backbone.to(device)
    # Count parameters before freezing them.
    backbone_param_count = backbone.num_parameters()
    for param in backbone.parameters():
        param.requires_grad_(False)
    backbone.eval()
    logger.info(
        "Loaded backbone from %s (%s params, now frozen)",
        args.backbone_checkpoint,
        f"{backbone_param_count:,}",
    )

    train_records = _load_records(
        args.data_dir / "train.jsonl", args.label_field, args.max_examples_per_split
    )
    val_records = _load_records(
        args.data_dir / "val.jsonl", args.label_field, args.max_examples_per_split
    )
    train_dataset = ClassificationDataset(
        train_records, tokenizer, block_size=args.block_size, label_list=args.label_list
    )
    val_dataset = ClassificationDataset(
        val_records, tokenizer, block_size=args.block_size, label_list=args.label_list
    )
    logger.info(
        "Dataset: train=%d, val=%d, labels=%s",
        len(train_dataset),
        len(val_dataset),
        args.label_list,
    )

    head = ClassificationHead(
        hidden_size=backbone_config.architecture.hidden_size, num_classes=len(args.label_list)
    ).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.learning_rate)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    best_val_accuracy = 0.0
    for epoch in range(1, args.epochs + 1):
        for input_ids, labels in train_loader:
            input_ids, labels = input_ids.to(device), labels.to(device)
            with torch.no_grad():  # backbone is frozen — no need to build its graph at all
                hidden = backbone(input_ids, return_hidden_states=True).hidden_states
            logits = head(hidden, input_ids, pad_token_id=backbone.pad_token_id)
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        val_loss, val_accuracy = evaluate(backbone, head, val_loader, device)
        logger.info("epoch=%d val_loss=%.4f val_accuracy=%.4f", epoch, val_loss, val_accuracy)
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "head_state_dict": head.state_dict(),
                    "label_list": args.label_list,
                    "backbone_checkpoint": str(args.backbone_checkpoint),
                    "backbone_config": str(args.backbone_config),
                    "block_size": args.block_size,
                    "val_accuracy": val_accuracy,
                },
                args.checkpoint_dir / "best.pt",
            )
            logger.info("Saved new best head checkpoint to %s", args.checkpoint_dir / "best.pt")

    print()
    print("=== Classification Head Training Summary ===")
    print(f"Task: {args.label_field} classification, labels={args.label_list}")
    print(f"Backbone: {backbone_config.model_name} (frozen)")
    print(f"Best val accuracy: {best_val_accuracy:.4f}")


if __name__ == "__main__":
    main()
