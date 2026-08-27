#!/usr/bin/env python
"""Inspect a DaraLM model config: validate it and report its shape."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from daralm.model.config import ArchitectureConfig, ModelConfig  # noqa: E402
from daralm.utils.device import get_device, get_device_name  # noqa: E402
from daralm.utils.logging import get_logger  # noqa: E402

logger = get_logger(__name__)

_BYTES_PER_PARAM = {"fp32": 4, "fp16": 2, "bf16": 2}


def estimate_parameters(arch: ArchitectureConfig) -> int:
    """Estimate total parameter count for DaraLM's decoder-only Transformer."""
    h = arch.hidden_size
    i = arch.intermediate_size
    v = arch.vocab_size
    n = arch.num_layers

    embedding_params = v * h

    attn_params_per_layer = 4 * h**2
    norm_params_per_layer = 2 * h
    ffn_params_per_layer = 2 * h * i
    per_layer_total = attn_params_per_layer + norm_params_per_layer + ffn_params_per_layer
    total_layer_params = n * per_layer_total

    final_norm_params = h

    return embedding_params + total_layer_params + final_norm_params


def estimate_memory_bytes(total_params: int, precision: str) -> dict[str, int]:
    """Estimate model / gradient / optimizer / total memory footprint, in bytes."""
    bytes_per_param = _BYTES_PER_PARAM[precision]
    model_memory = total_params * bytes_per_param
    gradient_memory = total_params * bytes_per_param
    optimizer_memory = total_params * 4 * 2
    total_training_memory = model_memory + gradient_memory + optimizer_memory
    return {
        "model_memory": model_memory,
        "gradient_memory": gradient_memory,
        "optimizer_memory": optimizer_memory,
        "total_training_memory": total_training_memory,
    }


def format_params(n: int) -> str:
    """Format a parameter count like '52.3M' or '1.2B'."""
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.1f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def format_bytes(n: int) -> str:
    """Format a byte count as MB or GB, whichever reads more naturally."""
    gb = n / 1024**3
    if gb >= 1.0:
        return f"{gb:.2f} GB"
    return f"{n / 1024**2:.1f} MB"


def print_report(config: ModelConfig, total_params: int, memory: dict[str, int]) -> None:
    arch = config.architecture
    device = get_device()
    device_name = get_device_name(device)

    print(f"Model: {config.model_name}")
    print("Architecture: Decoder-only Transformer")
    print()
    print(f"Vocabulary: {arch.vocab_size:,}")
    print(f"Context Length: {arch.max_position_embeddings:,}")
    print(f"Hidden Size: {arch.hidden_size}")
    print(f"Layers: {arch.num_layers}")
    print(f"Attention Heads: {arch.num_attention_heads}")
    print()
    print(f"Estimated Parameters: {format_params(total_params)}")
    print()
    print("Device:")
    print(device_name)
    print()
    print("Precision:")
    print(config.training.precision.upper())
    print()
    print("Estimated Memory (static config estimate — excludes activations):")
    print(f"  Model:     {format_bytes(memory['model_memory'])}")
    print(f"  Gradients: {format_bytes(memory['gradient_memory'])}")
    print(f"  Optimizer: {format_bytes(memory['optimizer_memory'])}")
    print(f"  Total:     {format_bytes(memory['total_training_memory'])}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a DaraLM model config.")
    parser.add_argument("--config", required=True, type=Path, help="Path to a config YAML file")
    args = parser.parse_args()

    logger.info("Loading config from %s", args.config)
    config = ModelConfig.from_yaml(args.config)

    total_params = estimate_parameters(config.architecture)
    memory = estimate_memory_bytes(total_params, config.training.precision)

    print_report(config, total_params, memory)


if __name__ == "__main__":
    main()
