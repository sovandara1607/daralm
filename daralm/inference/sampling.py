"""Logit-processing utilities for autoregressive generation."""

from __future__ import annotations

import torch


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Rescale logits by `1 / temperature`."""
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    return logits / temperature


def apply_repetition_penalty(
    logits: torch.Tensor, generated_ids: list[int], penalty: float
) -> torch.Tensor:
    """Discourage tokens that already appeared earlier in `generated_ids`."""
    if penalty <= 0:
        raise ValueError(f"repetition_penalty must be > 0, got {penalty}")
    if penalty == 1.0 or not generated_ids:
        return logits
    logits = logits.clone()
    for token_id in set(generated_ids):
        score = logits[token_id]
        logits[token_id] = score / penalty if score > 0 else score * penalty
    return logits


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Keep only the `k` highest-scoring tokens; mask the rest to -inf."""
    vocab_size = logits.size(-1)
    if k <= 0 or k >= vocab_size:
        return logits
    threshold = torch.topk(logits, k).values[..., -1]
    return logits.masked_fill(logits < threshold, float("-inf"))


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    if not 0.0 < p < 1.0:
        return logits

    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Keep the token that crosses the threshold and all earlier tokens.
    sorted_remove = cumulative_probs > p
    sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
    sorted_remove[..., 0] = False

    remove_mask = torch.zeros_like(logits, dtype=torch.bool)
    remove_mask.scatter_(-1, sorted_indices, sorted_remove)
    return logits.masked_fill(remove_mask, float("-inf"))


def sample_next_token(
    logits: torch.Tensor,
    generated_ids: list[int],
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
) -> int:
    """Apply repetition penalty -> temperature -> top-k -> top-p, then sample."""
    logits = apply_repetition_penalty(logits, generated_ids, repetition_penalty)

    if temperature == 0.0:
        return int(torch.argmax(logits).item())

    logits = apply_temperature(logits, temperature)
    logits = top_k_filter(logits, top_k)
    logits = top_p_filter(logits, top_p)

    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())
