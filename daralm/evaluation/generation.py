"""Memorization checking: does the model reproduce documents it trained on?

Spec section 15 explicitly lists this as its own evaluation concern,
distinct from perplexity: a model can have low loss on training data either
because it learned generalizable patterns, or because it simply memorized
the specific sequences. The two are indistinguishable from loss alone — you
have to actually check whether the model's own greedy continuation of a
training-document prefix matches that document's real continuation.

This module works at the token-ID level, not decoded text — comparing
decoded strings would blur token-boundary differences that are irrelevant
to whether the model reproduced the same underlying token sequence.

For Phase 5 (the overfitting sanity test), a *high* match rate on the tiny
overfit set is the desired outcome — proof the model can memorize at all.
Later, on the full training set, a suspiciously high match rate would be a
red flag for problematic memorization instead (spec section 15's actual
concern) — same measurement, opposite interpretation, depending on which
set you point it at.
"""

from __future__ import annotations

import torch

from daralm.model.transformer import DaraLMTransformer
from daralm.tokenizer.tokenizer import DaraLMTokenizer


@torch.no_grad()
def _greedy_continue(
    model: DaraLMTransformer, prompt_ids: list[int], num_tokens: int, device: torch.device
) -> list[int]:
    """Greedily generate `num_tokens` continuation IDs from `prompt_ids`.

    Greedy (not sampled) is deliberate here: memorization checking asks
    "what is the model *most confident* comes next", which is exactly what
    argmax decoding answers — sampling would inject randomness into a
    measurement that's supposed to be deterministic.
    """
    generated = list(prompt_ids)
    for _ in range(num_tokens):
        context = generated[-model.config.max_position_embeddings :]
        input_tensor = torch.tensor([context], dtype=torch.long, device=device)
        logits = model(input_tensor).logits[0, -1, :]
        generated.append(int(torch.argmax(logits).item()))
    return generated[len(prompt_ids) :]


def token_match_rate(generated_ids: list[int], reference_ids: list[int]) -> float:
    """Fraction of `reference_ids` positions exactly reproduced by `generated_ids`.

    Normalized by `len(reference_ids)` (not the shorter of the two), so a
    `generated_ids` that's shorter than the reference is penalized for the
    positions it never even reached, rather than let off the hook.
    """
    if not reference_ids:
        return 0.0
    compare_len = min(len(generated_ids), len(reference_ids))
    pairs = zip(generated_ids[:compare_len], reference_ids[:compare_len], strict=True)
    matches = sum(1 for a, b in pairs if a == b)
    return matches / len(reference_ids)


def check_memorization(
    model: DaraLMTransformer,
    tokenizer: DaraLMTokenizer,
    documents: list[str],
    prompt_fraction: float = 0.3,
    min_tokens: int = 20,
    max_reference_tokens: int = 50,
) -> list[dict]:
    """For each document, prompt with its first `prompt_fraction` of tokens
    and check how much of the *real* remainder the model's greedy
    continuation reproduces.

    Documents shorter than `min_tokens` (after tokenizing) are skipped —
    too little signal either way from a handful of tokens.

    `max_reference_tokens` caps how much of the continuation is actually
    checked (default 50). This isn't just a speed concession — generation
    here has no KV cache (Phase 4's deliberate simplicity trade-off), so
    checking a full multi-thousand-token article would mean thousands of
    uncached forward passes per document. It's also the methodologically
    right call independent of speed: real memorization audits (e.g. Carlini
    et al.'s "extractable memorization") test a fixed-length continuation
    window, not exact reproduction of an entire document — a bounded window
    is the standard, not a shortcut.

    Returns one result dict per evaluated document: decoded previews of the
    prompt/reference/generated continuation (for manual inspection) plus
    `token_match_rate`.
    """
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()

    results = []
    for text in documents:
        ids = tokenizer.encode(text, add_bos=True, add_eos=True)
        if len(ids) < min_tokens:
            continue

        prompt_len = max(1, int(len(ids) * prompt_fraction))
        prompt_ids = ids[:prompt_len]
        reference_ids = ids[prompt_len : prompt_len + max_reference_tokens]
        if not reference_ids:
            continue

        generated_ids = _greedy_continue(model, prompt_ids, len(reference_ids), device)
        results.append(
            {
                "prompt_preview": tokenizer.decode(prompt_ids)[:80],
                "reference_preview": tokenizer.decode(reference_ids)[:80],
                "generated_preview": tokenizer.decode(generated_ids)[:80],
                "num_reference_tokens": len(reference_ids),
                "token_match_rate": token_match_rate(generated_ids, reference_ids),
            }
        )

    model.train(was_training)
    return results


def average_match_rate(results: list[dict]) -> float:
    """Mean `token_match_rate` across `check_memorization`'s results, or 0.0 if empty."""
    if not results:
        return 0.0
    return sum(r["token_match_rate"] for r in results) / len(results)
