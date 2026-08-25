"""Autoregressive text generation.

Generation is the same forward pass used in training, run repeatedly: feed
the current sequence in, take the logits at the last position, turn them
into a next token (via `daralm.inference.sampling`), append it, repeat.
There is no separate "generation mode" in the model itself — the causal
mask already guarantees position i only depends on positions <= i, which is
exactly what makes this loop valid: appending a new token never changes the
logits already computed for earlier positions.

This implementation recomputes the *entire* sequence's forward pass at every
new token (no KV cache), which is simple and correct but means generation
cost grows quadratically with sequence length. A KV cache (reusing
previously-computed Key/Value projections instead of recomputing them) is a
natural and significant speedup — deliberately deferred, in the same spirit
as FlashAttention/torch.compile (spec section 20): get the simple, obviously
correct version working first.
"""

from __future__ import annotations

import torch

from daralm.data.chat_template import ASSISTANT_CLOSE, format_prompt
from daralm.inference.sampling import sample_next_token
from daralm.model.transformer import DaraLMTransformer
from daralm.tokenizer.tokenizer import DaraLMTokenizer


@torch.no_grad()
def generate(
    model: DaraLMTransformer,
    tokenizer: DaraLMTokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    stop_on_eos: bool = True,
) -> str:
    """Generate a continuation of `prompt`, returning the full decoded text
    (prompt + generated continuation).

    Args:
        model: a `DaraLMTransformer` in eval mode is assumed by the caller
            (this function doesn't call `.eval()` itself, so training-mode
            dropout would otherwise silently leak into generation).
        tokenizer: used both to encode the prompt and decode the result.
        prompt: seed text; encoded with a leading `<bos>` but no `<eos>`
            (there's more to generate).
        max_new_tokens: hard cap on generated tokens, regardless of EOS.
        temperature, top_k, top_p, repetition_penalty: see
            `daralm.inference.sampling.sample_next_token`.
        stop_on_eos: stop as soon as `<eos>` is generated, rather than
            continuing to `max_new_tokens` regardless.
    """
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    generated_ids = list(input_ids)

    for _ in range(max_new_tokens):
        context = generated_ids[-model.config.max_position_embeddings :]
        input_tensor = torch.tensor([context], dtype=torch.long, device=device)

        output = model(input_tensor)
        next_token_logits = output.logits[0, -1, :]

        next_id = sample_next_token(
            next_token_logits,
            generated_ids=generated_ids,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
        generated_ids.append(next_id)

        if stop_on_eos and next_id == tokenizer.eos_id:
            break

    return tokenizer.decode(generated_ids)


@torch.no_grad()
def generate_chat(
    model: DaraLMTransformer,
    tokenizer: DaraLMTokenizer,
    instruction: str,
    max_new_tokens: int = 150,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
) -> str:
    """Generate an assistant response to `instruction` using the chat template.

    Wraps `instruction` with `daralm.data.chat_template.format_prompt` (the
    same `<user>...</user>\\n<assistant>\\n` framing `InstructionDataset`
    trains on) before generating, then returns only the assistant's turn —
    not the echoed prompt.

    Since `</assistant>` is plain text, not a real special token (see
    `daralm.data.chat_template`'s module docstring for why), the model has
    no guaranteed stopping signal beyond `<eos>` itself, which a lightly
    fine-tuned model may not reliably emit right after `</assistant>`. This
    loop stops as soon as the literal `</assistant>` marker text appears in
    the decoded output, in addition to stopping on `<eos>` — belt and
    suspenders, not a replacement for actually training the model to emit
    `<eos>` reliably.
    """
    device = next(model.parameters()).device
    prompt = format_prompt(instruction)
    prompt_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    generated_ids = list(prompt_ids)
    num_prompt_tokens = len(prompt_ids)

    for _ in range(max_new_tokens):
        context = generated_ids[-model.config.max_position_embeddings :]
        input_tensor = torch.tensor([context], dtype=torch.long, device=device)

        output = model(input_tensor)
        next_token_logits = output.logits[0, -1, :]

        next_id = sample_next_token(
            next_token_logits,
            generated_ids=generated_ids,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
        generated_ids.append(next_id)

        if next_id == tokenizer.eos_id:
            break

        response_so_far = tokenizer.decode(generated_ids[num_prompt_tokens:])
        if ASSISTANT_CLOSE in response_so_far:
            break

    response_text = tokenizer.decode(generated_ids[num_prompt_tokens:])
    response_text = response_text.split(ASSISTANT_CLOSE)[0]
    return response_text.strip()
