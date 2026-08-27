"""Autoregressive text generation."""

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
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    generated_ids = list(input_ids)

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
    """Generate an assistant response to `instruction` using the chat template."""
    device = next(model.parameters()).device
    prompt = format_prompt(instruction)
    prompt_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    generated_ids = list(prompt_ids)
    num_prompt_tokens = len(prompt_ids)

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
