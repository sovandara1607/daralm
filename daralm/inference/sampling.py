"""Logit-processing utilities for autoregressive generation.

Each function here takes a 1D tensor of next-token logits (one vocabulary-
sized vector, for a single sequence's most recent position) and returns a
modified version — they compose in the order `daralm.inference.generator`
applies them: repetition penalty -> temperature -> top-k -> top-p -> sample.

Why sampling controls exist at all: taking the single highest-probability
token every step (greedy decoding) is deterministic but tends to produce
dull, repetitive text and can't express genuine model uncertainty between
several plausible continuations. Sampling from the model's actual
probability distribution — optionally reshaped by temperature/top-k/top-p —
trades some of that determinism for more natural-sounding variation.
"""

from __future__ import annotations

import torch


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Rescale logits by `1 / temperature`.

    Temperature < 1 sharpens the distribution (more confident, closer to
    greedy as it approaches 0); temperature > 1 flattens it (more random,
    approaching uniform as it grows). `temperature == 1.0` leaves the
    distribution unchanged.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    return logits / temperature


def apply_repetition_penalty(
    logits: torch.Tensor, generated_ids: list[int], penalty: float
) -> torch.Tensor:
    """Discourage tokens that already appeared earlier in `generated_ids`.

    Standard CTRL-style penalty: for each previously-generated token,
    positive logits are divided by `penalty` (pushed down) and negative
    logits are multiplied by `penalty` (also pushed down, since multiplying
    a negative number by a value > 1 makes it more negative) — either way,
    already-used tokens become less likely, discouraging loops like
    "the the the the...". `penalty == 1.0` is a no-op.
    """
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
    """Keep only the `k` highest-scoring tokens; mask the rest to -inf.

    `k <= 0` or `k >= vocab_size` is a no-op (nothing to filter).
    """
    vocab_size = logits.size(-1)
    if k <= 0 or k >= vocab_size:
        return logits
    threshold = torch.topk(logits, k).values[..., -1]
    return logits.masked_fill(logits < threshold, float("-inf"))


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Nucleus sampling: keep the smallest set of highest-probability tokens
    whose cumulative probability is >= p; mask everything else to -inf.

    Unlike top-k (a fixed count), this adapts to the shape of the
    distribution: a peaked distribution keeps very few tokens, a flat one
    keeps many. `p >= 1.0` is a no-op.
    """
    if not 0.0 < p < 1.0:
        return logits

    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Keep the first token that crosses the threshold, and everything
    # before it (shift the "remove" mask right by one so we don't drop the
    # very token that crosses p).
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
    """Apply repetition penalty -> temperature -> top-k -> top-p, then sample.

    `temperature == 0` is treated as a request for greedy decoding (argmax),
    bypassing sampling entirely — dividing by zero would otherwise be
    undefined.
    """
    logits = apply_repetition_penalty(logits, generated_ids, repetition_penalty)

    if temperature == 0.0:
        return int(torch.argmax(logits).item())

    logits = apply_temperature(logits, temperature)
    logits = top_k_filter(logits, top_k)
    logits = top_p_filter(logits, top_p)

    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())
