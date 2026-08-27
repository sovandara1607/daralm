"""Memorization checking: does the model reproduce documents it trained on?."""

from __future__ import annotations

import torch

from daralm.model.transformer import DaraLMTransformer
from daralm.tokenizer.tokenizer import DaraLMTokenizer


@torch.no_grad()
def _greedy_continue(
    model: DaraLMTransformer, prompt_ids: list[int], num_tokens: int, device: torch.device
) -> list[int]:
    """Greedily generate `num_tokens` continuation IDs from `prompt_ids`."""
    generated = list(prompt_ids)
    for _ in range(num_tokens):
        context = generated[-model.config.max_position_embeddings :]
        input_tensor = torch.tensor([context], dtype=torch.long, device=device)
        logits = model(input_tensor).logits[0, -1, :]
        generated.append(int(torch.argmax(logits).item()))
    return generated[len(prompt_ids) :]


def token_match_rate(generated_ids: list[int], reference_ids: list[int]) -> float:
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
    if not results:
        return 0.0
    return sum(r["token_match_rate"] for r in results) / len(results)
