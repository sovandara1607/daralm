"""Autoregressive text generation.

Generation is the same forward pass used in training, run repeatedly: feed
the current sequence in, take the logits at the last position, turn them
into a next token (via `daralm.inference.sampling`), append it, repeat.
There is no separate "generation mode" in the model itself — the causal
mask already guarantees position i only depends on positions <= i, which is
exactly what makes this loop valid: appending a new token never changes the
logits already computed for earlier positions.

**KV-cached**, as of this project's own optimization pass: the prompt is
processed once (one forward pass over every prompt token, `use_cache=True`),
then each generated token is fed through the model *alone*, reusing the
growing per-layer key/value cache from every prior step
(`daralm.model.attention.CausalSelfAttention` — see its docstring for the
full mechanism). This is an O(n) total-work loop instead of the O(n^2) loop
recomputing every prior token's K/V from scratch on every step would be —
the same optimization every production LLM's decoding uses. Verified
bit-identical to full recomputation before being wired in here, not just
assumed correct (`tests/test_model.py`'s cache-correctness tests).

One real, disclosed behavior change from the pre-cache version: the old
loop silently truncated to a sliding window of the last
`max_position_embeddings` tokens whenever the running sequence grew past
that bound, which (because RoPE positions were always recomputed from 0 on
every full-recompute step) quietly re-numbered the truncated window's
positions rather than preserving true absolute position — an unintended
approximation, not a documented feature. The cached version instead simply
stops generating once the cache would exceed `max_position_embeddings`
(checked via `max_new_tokens` and the prompt length up front) — a real
constraint on how long a single generation can run, made explicit instead
of silently approximated around.
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

    # Prompt is truncated up front (once), not re-sliced every step — with
    # a KV cache, positions are real and monotonic, so there's no sliding
    # window to maintain mid-generation the way the pre-cache loop had to.
    context = generated_ids[-model.config.max_position_embeddings :]
    prompt_tensor = torch.tensor([context], dtype=torch.long, device=device)
    output = model(prompt_tensor, use_cache=True)
    past_key_values = output.past_key_values
    next_token_logits = output.logits[0, -1, :]

    for _ in range(max_new_tokens):
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

        cached_len = past_key_values[0][0].size(2)
        if cached_len >= model.config.max_position_embeddings:
            break  # out of position budget — see module docstring

        next_input = torch.tensor([[next_id]], dtype=torch.long, device=device)
        output = model(next_input, past_key_values=past_key_values, use_cache=True)
        past_key_values = output.past_key_values
        next_token_logits = output.logits[0, -1, :]

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
    loop stops as soon as the marker text appears in the decoded output, in
    addition to stopping on `<eos>` — belt and suspenders, not a
    replacement for actually training the model to emit `<eos>` reliably.

    A real bug lived here until it was caught by a grammar-correction
    overfit diagnostic: the stop check compared against the literal string
    `"</assistant>"`, but this tokenizer has essentially no coverage of
    `<`/`>` characters (a known, disclosed gap — see
    `ROADMAP_NLP_PLATFORM.md` Phase 1), so `<` in the marker round-trips
    through `encode`+`decode` as `<unk>`'s placeholder glyph, not `<`
    itself — `tokenizer.decode(tokenizer.encode("</assistant>"))` produces
    `" ⁇ /assistant>"`, never the literal string this used to check for.
    The stop condition silently never fired: generation ran to
    `max_new_tokens` even when the model had already produced a correct
    response immediately followed by its (undetectable) attempt at the
    closing marker. Fixed by checking against the marker's own actual
    decoded form instead of its literal source text — computed once,
    up front, from the same tokenizer doing the generating, so it stays
    correct regardless of that tokenizer's specific `<unk>` behavior.
    """
    device = next(model.parameters()).device
    prompt = format_prompt(instruction)
    prompt_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    generated_ids = list(prompt_ids)
    num_prompt_tokens = len(prompt_ids)

    # The marker as it will *actually* come back out of `decode`, not as it
    # went into `encode` — see the docstring above for why those differ.
    decoded_assistant_close = tokenizer.decode(
        tokenizer.encode(ASSISTANT_CLOSE, add_bos=False, add_eos=False)
    )

    context = generated_ids[-model.config.max_position_embeddings :]
    prompt_tensor = torch.tensor([context], dtype=torch.long, device=device)
    output = model(prompt_tensor, use_cache=True)
    past_key_values = output.past_key_values
    next_token_logits = output.logits[0, -1, :]

    for _ in range(max_new_tokens):
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
        if decoded_assistant_close in response_so_far:
            break

        cached_len = past_key_values[0][0].size(2)
        if cached_len >= model.config.max_position_embeddings:
            break  # out of position budget — see module docstring

        next_input = torch.tensor([[next_id]], dtype=torch.long, device=device)
        output = model(next_input, past_key_values=past_key_values, use_cache=True)
        past_key_values = output.past_key_values
        next_token_logits = output.logits[0, -1, :]

    response_text = tokenizer.decode(generated_ids[num_prompt_tokens:])
    response_text = response_text.split(decoded_assistant_close)[0]
    return response_text.strip()
