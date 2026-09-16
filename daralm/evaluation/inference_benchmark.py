"""Measure generation throughput/latency/memory — the before/after harness every
Model Optimization stage's claims are checked against (no optimization claim without
a number from this module, per this project's standing rule)."""

from __future__ import annotations

import time
from typing import Any

import torch

from daralm.inference.generator import generate
from daralm.model.transformer import DaraLMTransformer
from daralm.tokenizer.tokenizer import DaraLMTokenizer
from daralm.utils.device import peak_memory_gb


@torch.no_grad()
def benchmark_generation(
    model: DaraLMTransformer,
    tokenizer: DaraLMTokenizer,
    prompts: list[str],
    max_new_tokens: int = 50,
    num_repeats: int = 3,
    temperature: float = 0.8,
    top_p: float = 0.9,
) -> dict[str, Any]:
    """Run `generate()` over `prompts` (each `num_repeats` times) and report
    tokens/sec, latency percentiles, and peak device memory.

    Each (prompt, repeat) pair is one timed sample — `num_repeats` exists to smooth
    out scheduling noise, not to warm up caches (this project's `generate()` has no
    persistent cache across calls).
    """
    if not prompts:
        raise ValueError("prompts must be non-empty — nothing to benchmark")
    if num_repeats < 1:
        raise ValueError(f"num_repeats must be >= 1, got {num_repeats}")

    device = next(model.parameters()).device
    was_training = model.training
    model.eval()

    latencies_ms: list[float] = []
    tokens_per_call: list[int] = []

    for prompt in prompts:
        for _ in range(num_repeats):
            prompt_len = len(tokenizer.encode(prompt, add_bos=True, add_eos=False))
            start = time.perf_counter()
            text = generate(
                model,
                tokenizer,
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            elapsed_s = time.perf_counter() - start

            generated_len = len(tokenizer.encode(text, add_bos=True, add_eos=False)) - prompt_len
            generated_len = max(generated_len, 0)

            latencies_ms.append(elapsed_s * 1000)
            tokens_per_call.append(generated_len)

    model.train(was_training)

    total_tokens = sum(tokens_per_call)
    total_time_s = sum(latencies_ms) / 1000
    sorted_latencies = sorted(latencies_ms)

    return {
        "num_samples": len(latencies_ms),
        "tokens_per_sec": total_tokens / total_time_s if total_time_s > 0 else 0.0,
        "latency_p50_ms": _percentile(sorted_latencies, 0.50),
        "latency_p95_ms": _percentile(sorted_latencies, 0.95),
        "peak_memory_gb": peak_memory_gb(device),
        "total_tokens_generated": total_tokens,
        "total_time_s": total_time_s,
    }


def _percentile(sorted_values: list[float], fraction: float) -> float:
    """Nearest-rank percentile — no interpolation, fine for small sample benchmarks."""
    if not sorted_values:
        return 0.0
    index = min(int(fraction * len(sorted_values)), len(sorted_values) - 1)
    return sorted_values[index]
