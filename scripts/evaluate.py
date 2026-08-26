#!/usr/bin/env python
"""Phase 8: consolidated evaluation report across every trained checkpoint.

Runs the full spec section 15 checklist — language modeling metrics,
fixed-prompt generation quality, memorization, overfitting — uniformly
across DaraLM-Tiny, DaraLM-10M, and DaraLM-50M, and writes one comparable
report. This is the point of the phase: not any single new number, but
being able to look at every model trained so far side by side, on the
exact same measurements.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --checkpoints checkpoints/daralm-10m checkpoints/daralm-50m
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from daralm.data.loader import load_jsonl  # noqa: E402
from daralm.evaluation.benchmarks import compare_checkpoints  # noqa: E402
from daralm.tokenizer.tokenizer import DaraLMTokenizer  # noqa: E402
from daralm.utils.device import get_device, get_device_name  # noqa: E402
from daralm.utils.logging import get_logger  # noqa: E402

logger = get_logger(__name__)

DEFAULT_CHECKPOINTS = [
    Path("checkpoints/daralm-tiny"),
    Path("checkpoints/daralm-10m"),
    Path("checkpoints/daralm-50m"),
    Path("checkpoints/daralm-150m"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", type=Path, default=DEFAULT_CHECKPOINTS)
    parser.add_argument(
        "--tokenizer", type=Path, default=Path("checkpoints/tokenizer/unigram.model")
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/cleaned"))
    parser.add_argument(
        "--n-memorization-samples",
        type=int,
        default=30,
        help="Real training documents sampled (fixed seed) to check for memorization",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("experiments/evaluation_report.json"))
    return parser.parse_args()


def print_report(report: dict) -> None:
    print()
    print("=== Phase 8 Evaluation Report ===")
    header = (
        f"{'model':<16}{'params':>10}{'val_loss':>10}{'perplexity':>12}"
        f"{'mem_vs_chance':>15}{'overfit':>10}"
    )
    print(header)
    print("-" * len(header))
    for name, result in report["models"].items():
        hist = result["history_summary"]
        val_loss = hist.get("final_val_loss")
        ppl = hist.get("final_perplexity")
        mem_ratio = result["memorization"]["match_rate_vs_chance"]
        overfit_short = (
            "OK" if "no overfitting" in result["overfitting"]["verdict"]
            else "WATCH" if "watch" in result["overfitting"]["verdict"]
            else "?!" if "early overfitting" in result["overfitting"]["verdict"]
            else "n/a"
        )
        print(
            f"{name:<16}{result['parameters']:>10,}"
            f"{(f'{val_loss:.3f}' if val_loss is not None else 'n/a'):>10}"
            f"{(f'{ppl:.1f}' if ppl is not None else 'n/a'):>12}"
            f"{(f'{mem_ratio:.1f}x' if mem_ratio is not None else 'n/a'):>15}"
            f"{overfit_short:>10}"
        )

    print()
    for name, result in report["models"].items():
        print(f"--- {name} ---")
        print(f"  Overfitting: {result['overfitting']['verdict']}")
        print(
            f"  Memorization: {result['memorization']['avg_match_rate']:.4f} avg match rate "
            f"over {result['memorization']['num_documents_checked']} real training docs "
            f"({result['memorization']['match_rate_vs_chance']:.1f}x random-chance baseline)"
        )
        for domain, text in result["generations"].items():
            print(f"  [{domain}] {text[:100]}{'...' if len(text) > 100 else ''}")
        print()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    tokenizer = DaraLMTokenizer.from_pretrained(args.tokenizer)
    device = get_device()
    logger.info("Using device: %s", get_device_name(device))

    all_records = load_jsonl(args.data_dir / "train.jsonl")
    memorization_sample = random.sample(
        all_records, min(args.n_memorization_samples, len(all_records))
    )
    real_documents = [r["text"] for r in memorization_sample]
    logger.info("Sampled %d real training documents for memorization check", len(real_documents))

    existing_checkpoints = [c for c in args.checkpoints if c.exists()]
    missing = set(args.checkpoints) - set(existing_checkpoints)
    if missing:
        logger.warning("Skipping missing checkpoint dirs: %s", sorted(str(m) for m in missing))
    if not existing_checkpoints:
        raise FileNotFoundError(f"None of the requested checkpoints exist: {args.checkpoints}")

    report = compare_checkpoints(existing_checkpoints, tokenizer, device, real_documents)
    print_report(report)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Full evaluation report written to: {args.output}")


if __name__ == "__main__":
    main()
