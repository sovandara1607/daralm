#!/usr/bin/env python
"""Model Optimization Stage 2: write an int8-quantized release checkpoint and
measure the compression trade honestly — on-disk size and perplexity delta on
the same validation data used at training time, side by side. No compression
claim without both numbers, per this project's standing rule (see
ROADMAP_NLP_PLATFORM.md).

Writes pytorch_model_int8.pt next to the source checkpoint's release artifact,
in the same release format ModelService.from_checkpoint already reads (so it
can be served directly by pointing --checkpoint at the output directory and
copying config.yaml/tokenizer.model alongside it, exactly like pytorch_model.pt).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from daralm.data.dataset import PackedTokenDataset  # noqa: E402
from daralm.data.loader import load_jsonl  # noqa: E402
from daralm.evaluation.perplexity import compute_perplexity  # noqa: E402
from daralm.inference.quantization import quantize_dynamic_int8  # noqa: E402
from daralm.model.config import ModelConfig  # noqa: E402
from daralm.model.transformer import DaraLMTransformer  # noqa: E402
from daralm.tokenizer.tokenizer import DaraLMTokenizer  # noqa: E402
from daralm.training.checkpoint import load_checkpoint  # noqa: E402
from daralm.utils.logging import get_logger  # noqa: E402

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path, help="Checkpoint directory")
    parser.add_argument(
        "--tokenizer", type=Path, default=Path("checkpoints/tokenizer/unigram.model")
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/cleaned"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to --checkpoint's own directory (writes pytorch_model_int8.pt there)",
    )
    parser.add_argument("--report", type=Path, default=Path("experiments/quantization_report.json"))
    return parser.parse_args()


def _load_model(checkpoint_dir: Path, tokenizer: DaraLMTokenizer, config: ModelConfig):
    model = DaraLMTransformer(config.architecture, pad_token_id=tokenizer.pad_id)
    checkpoint_path = checkpoint_dir / "checkpoint.pt"
    if checkpoint_path.exists():
        load_checkpoint(checkpoint_dir, model, tokenizer_path=None)
    else:
        release_path = checkpoint_dir / "pytorch_model.pt"
        if not release_path.exists():
            raise FileNotFoundError(
                f"No checkpoint.pt or pytorch_model.pt found in {checkpoint_dir}"
            )
        release = torch.load(release_path, map_location="cpu", weights_only=True)
        model.load_state_dict(release["model_state_dict"])
    return model


def main() -> None:
    args = parse_args()

    tokenizer = DaraLMTokenizer.from_pretrained(args.tokenizer)
    config = ModelConfig.from_yaml(args.checkpoint / "config.yaml")
    if tokenizer.vocab_size != config.architecture.vocab_size:
        raise ValueError(
            f"Tokenizer vocab_size ({tokenizer.vocab_size}) does not match "
            f"config's architecture.vocab_size ({config.architecture.vocab_size})."
        )

    # Dynamic quantization is CPU-only, see daralm/inference/quantization.py.
    device = torch.device("cpu")
    model = _load_model(args.checkpoint, tokenizer, config).to(device)
    model.eval()

    val_records = load_jsonl(args.data_dir / "val.jsonl")
    val_dataset = PackedTokenDataset(
        val_records, tokenizer, block_size=config.architecture.max_position_embeddings
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    logger.info("Evaluating original (fp32/bf16) perplexity on %d val blocks", len(val_dataset))
    original_loss, original_ppl = compute_perplexity(model, val_loader, device)

    original_state = model.state_dict()
    original_bytes = sum(t.numel() * t.element_size() for t in original_state.values())

    logger.info("Quantizing to int8 (dynamic, CPU)")
    quantized_model = quantize_dynamic_int8(model)

    logger.info("Evaluating quantized perplexity on the same %d val blocks", len(val_dataset))
    quantized_loss, quantized_ppl = compute_perplexity(quantized_model, val_loader, device)

    output_dir = args.output_dir or args.checkpoint
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "pytorch_model_int8.pt"
    torch.save(
        {
            "model_state_dict": quantized_model.state_dict(),
            "step": None,
            "tokens_processed": None,
            "config": config.model_dump(),
            "quantized": True,
        },
        output_path,
    )
    quantized_bytes = output_path.stat().st_size

    relative_ppl_delta = (quantized_ppl - original_ppl) / original_ppl

    report = {
        "model_name": config.model_name,
        "source_checkpoint": str(args.checkpoint),
        "output_path": str(output_path),
        "original_val_loss": original_loss,
        "original_perplexity": original_ppl,
        "original_state_dict_bytes": original_bytes,
        "quantized_val_loss": quantized_loss,
        "quantized_perplexity": quantized_ppl,
        "quantized_file_bytes": quantized_bytes,
        "relative_perplexity_delta": relative_ppl_delta,
        "size_reduction_pct": (
            (1 - quantized_bytes / original_bytes) * 100 if original_bytes else None
        ),
    }

    print(json.dumps(report, indent=2))

    args.report.parent.mkdir(parents=True, exist_ok=True)
    history = []
    if args.report.exists():
        with args.report.open(encoding="utf-8") as f:
            history = json.load(f)
    history.append(report)
    with args.report.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"\nAppended to {args.report} ({len(history)} total entries)")

    if abs(relative_ppl_delta) > 0.05:
        logger.warning(
            "Quantized perplexity moved %.1f%% relative to original — outside this project's "
            "5%% tolerance (see tests/test_quantization.py). Do not ship this artifact without "
            "reviewing whether that's acceptable for %s.",
            relative_ppl_delta * 100,
            config.model_name,
        )


if __name__ == "__main__":
    main()
