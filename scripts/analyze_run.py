#!/usr/bin/env python
"""Analyze a completed (or in-progress) training run: loss curves, throughput,
memory, and generation quality (spec section 28, Phase 6's explicit ask).

Reads `history.json` (written by `daralm.training.trainer.Trainer`) for the
quantitative timeline, and runs generation from the best checkpoint for a
qualitative read — the two together are what "analyze this run" means; loss
curves alone can't tell you whether the model produces anything reasonable,
and generation samples alone can't tell you whether training was efficient.

Usage:
    python scripts/analyze_run.py --checkpoint-dir checkpoints/daralm-10m
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from daralm.evaluation.benchmarks import FIXED_PROMPTS, summarize_history  # noqa: E402
from daralm.inference.generator import generate  # noqa: E402
from daralm.model.config import ModelConfig  # noqa: E402
from daralm.model.transformer import DaraLMTransformer  # noqa: E402
from daralm.tokenizer.tokenizer import DaraLMTokenizer  # noqa: E402
from daralm.training.checkpoint import load_checkpoint  # noqa: E402
from daralm.utils.device import get_device, get_device_name  # noqa: E402
from daralm.utils.logging import get_logger  # noqa: E402

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument(
        "--tokenizer", type=Path, default=Path("checkpoints/tokenizer/unigram.model")
    )
    parser.add_argument("--max-new-tokens", type=int, default=60)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    return parser.parse_args()


def print_summary(summary: dict, device_name: str) -> None:
    print()
    print("=== Training Loss & Validation Loss ===")
    if "initial_train_loss" in summary:
        print(
            f"Train loss:  {summary['initial_train_loss']:.4f} -> {summary['final_train_loss']:.4f}"
        )
    if "initial_val_loss" in summary:
        print(f"Val loss:    {summary['initial_val_loss']:.4f} -> {summary['final_val_loss']:.4f}")
        print(
            f"Best val loss: {summary['best_val_loss']:.4f} "
            f"(perplexity {summary['best_perplexity']:.1f})"
        )

    print()
    print("=== Throughput & Memory ===")
    print(f"Device: {device_name}")
    if "avg_tokens_per_sec" in summary:
        print(
            f"Tokens/sec: avg={summary['avg_tokens_per_sec']:.0f} "
            f"max={summary['max_tokens_per_sec']:.0f}"
        )
    if summary.get("peak_device_memory_gb") is not None:
        print(f"Peak device memory: {summary['peak_device_memory_gb']:.2f} GB")
    else:
        print(
            "Peak device memory: not measurable on this device type (see module docstring / "
            "daralm.training.trainer._gpu_memory_gb — CUDA and MPS expose different, "
            "non-interchangeable memory APIs; no PyTorch-level utilization % exists for MPS "
            "the way nvidia-smi provides for CUDA)"
        )


def main() -> None:
    args = parse_args()

    history_path = args.checkpoint_dir / "history.json"
    if not history_path.exists():
        raise FileNotFoundError(
            f"No history.json in {args.checkpoint_dir} — was this checkpoint produced by "
            "scripts/train.py (or scripts/overfit_test.py)?"
        )
    with history_path.open() as f:
        history = json.load(f)
    summary = summarize_history(history)

    device = get_device()
    print_summary(summary, get_device_name(device))

    # --- Generation quality ---
    best_dir = args.checkpoint_dir / "best"
    tokenizer = DaraLMTokenizer.from_pretrained(args.tokenizer)
    config = ModelConfig.from_yaml(best_dir / "config.yaml")
    model = DaraLMTransformer(config.architecture, pad_token_id=tokenizer.pad_id)
    load_checkpoint(best_dir, model, tokenizer_path=args.tokenizer)
    model.to(device)
    model.eval()

    print()
    print("=== Generation Quality (best checkpoint, fixed prompts) ===")
    generations = {}
    for domain, prompt in FIXED_PROMPTS.items():
        text = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        generations[domain] = text
        print(f"\n[{domain}] prompt={prompt!r}")
        print(f"  {text}")

    report = {
        "model_name": config.model_name,
        "history_summary": summary,
        "device": get_device_name(device),
        "generations": generations,
    }
    report_path = args.checkpoint_dir / "analysis_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nFull analysis report written to: {report_path}")


if __name__ == "__main__":
    main()
