#!/usr/bin/env python
"""Model Optimization Stage 0: measure generation tokens/sec, latency, and peak
memory for a trained checkpoint — the before/after harness every later
optimization stage (SDPA, quantization, torch.compile, ...) is checked against.

Appends one timestamped entry per run to --output, so a "before" and "after" run
against the same checkpoint sit side by side instead of overwriting each other.

Loads via ModelService.from_checkpoint — the same code path api/main.py serves
from — so it transparently handles both a resumable checkpoint.pt directory and
a stripped pytorch_model.pt release, and honors DARALM_QUANTIZE if set (compare
quantized vs. not by running this script twice with/without that env var).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.services.model_service import ModelService  # noqa: E402
from daralm.evaluation.benchmarks import FIXED_PROMPTS  # noqa: E402
from daralm.evaluation.inference_benchmark import benchmark_generation  # noqa: E402
from daralm.utils.logging import get_logger  # noqa: E402

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path, help="Checkpoint directory")
    parser.add_argument(
        "--tokenizer", type=Path, default=Path("checkpoints/tokenizer/unigram.model")
    )
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument(
        "--num-repeats", type=int, default=3, help="Timed samples per prompt, to smooth noise"
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Free-text tag for this run (e.g. 'baseline', 'sdpa', 'int8') — makes "
        "before/after runs easy to tell apart in --output",
    )
    parser.add_argument("--output", type=Path, default=Path("experiments/inference_benchmark.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config_path = args.checkpoint / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"No config.yaml found in {args.checkpoint} — is this a valid checkpoint/release "
            "directory produced by scripts/train.py or scripts/quantize_checkpoint.py?"
        )
    service = ModelService.from_checkpoint(config_path, args.checkpoint, args.tokenizer)

    logger.info(
        "Benchmarking %s on %s (quantized=%s, max_new_tokens=%d, num_repeats=%d)",
        service.config.model_name,
        service.model_info()["device"],
        service.quantized,
        args.max_new_tokens,
        args.num_repeats,
    )
    result = benchmark_generation(
        service.model,
        service.tokenizer,
        prompts=list(FIXED_PROMPTS.values()),
        max_new_tokens=args.max_new_tokens,
        num_repeats=args.num_repeats,
    )

    info = service.model_info()
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_name": info["model_name"],
        "parameters": info["parameters"],
        "device": info["device"],
        "quantized": info["quantized"],
        "label": args.label,
        **result,
    }

    print(json.dumps(entry, indent=2))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    history = []
    if args.output.exists():
        with args.output.open(encoding="utf-8") as f:
            history = json.load(f)
    history.append(entry)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"\nAppended to {args.output} ({len(history)} total entries)")


if __name__ == "__main__":
    main()
