"""Tests for daralm.inference — sampling utilities and the generation loop."""

from __future__ import annotations

import pytest
import torch

from daralm.inference.generator import generate, generate_chat
from daralm.inference.sampling import (
    apply_repetition_penalty,
    apply_temperature,
    sample_next_token,
    top_k_filter,
    top_p_filter,
)
from daralm.model.config import ArchitectureConfig
from daralm.model.transformer import DaraLMTransformer
from daralm.tokenizer.tokenizer import DaraLMTokenizer
from daralm.tokenizer.train import train_sentencepiece


def test_temperature_one_is_identity():
    logits = torch.tensor([1.0, 2.0, 3.0])
    assert torch.equal(apply_temperature(logits, 1.0), logits)


def test_temperature_below_one_sharpens_relative_gap():
    logits = torch.tensor([1.0, 2.0])
    result = apply_temperature(logits, 0.5)
    assert (result[1] - result[0]) > (logits[1] - logits[0])


def test_temperature_zero_or_negative_raises():
    with pytest.raises(ValueError):
        apply_temperature(torch.tensor([1.0]), 0.0)
    with pytest.raises(ValueError):
        apply_temperature(torch.tensor([1.0]), -1.0)


def test_top_k_keeps_only_k_highest():
    logits = torch.tensor([5.0, 1.0, 4.0, 2.0, 3.0])
    result = top_k_filter(logits, k=2)
    kept = torch.isfinite(result)
    assert kept.sum().item() == 2
    assert result[0] == 5.0  # highest
    assert result[2] == 4.0  # second highest
    assert not torch.isfinite(result[1])


def test_top_k_zero_is_noop():
    logits = torch.tensor([5.0, 1.0, 4.0])
    assert torch.equal(top_k_filter(logits, k=0), logits)


def test_top_k_larger_than_vocab_is_noop():
    logits = torch.tensor([5.0, 1.0])
    assert torch.equal(top_k_filter(logits, k=100), logits)


def test_top_p_keeps_smallest_set_covering_probability_mass():
    logits = torch.tensor([10.0, 0.0, 0.0, 0.0])
    result = top_p_filter(logits, p=0.9)
    kept = torch.isfinite(result)
    assert kept[0]  # the dominant token is always kept
    assert kept.sum().item() < 4  # not everything survives


def test_top_p_one_is_noop():
    logits = torch.tensor([5.0, 1.0, 4.0])
    assert torch.equal(top_p_filter(logits, p=1.0), logits)


def test_top_p_always_keeps_at_least_one_token():
    logits = torch.tensor([1.0, 1.0, 1.0, 1.0])
    result = top_p_filter(logits, p=0.01)
    assert torch.isfinite(result).sum().item() >= 1


def test_repetition_penalty_one_is_noop():
    logits = torch.tensor([1.0, -1.0, 2.0])
    result = apply_repetition_penalty(logits, generated_ids=[0, 1], penalty=1.0)
    assert torch.equal(result, logits)


def test_repetition_penalty_reduces_repeated_token_score():
    logits = torch.tensor([2.0, -2.0])
    result = apply_repetition_penalty(logits, generated_ids=[0, 1], penalty=1.5)
    assert result[0] < logits[0]  # positive logit pushed down
    assert result[1] < logits[1]  # negative logit pushed further down


def test_repetition_penalty_empty_history_is_noop():
    logits = torch.tensor([1.0, 2.0])
    result = apply_repetition_penalty(logits, generated_ids=[], penalty=1.5)
    assert torch.equal(result, logits)


def test_sample_next_token_temperature_zero_is_greedy():
    logits = torch.tensor([1.0, 5.0, 2.0])
    token = sample_next_token(logits, generated_ids=[], temperature=0.0)
    assert token == 1  # argmax


def test_sample_next_token_top_k_one_is_deterministic():
    logits = torch.tensor([1.0, 5.0, 2.0])
    for _ in range(5):
        token = sample_next_token(logits, generated_ids=[], temperature=1.0, top_k=1)
        assert token == 1  # only the highest-scoring token survives top_k=1


@pytest.fixture(scope="module")
def tiny_tokenizer(tmp_path_factory):
    corpus_dir = tmp_path_factory.mktemp("gen_corpus")
    corpus_path = corpus_dir / "corpus.txt"
    lines = ["hello world this is a test sentence", "another different sentence here"] * 30
    corpus_path.write_text("\n".join(lines), encoding="utf-8")
    model_path = train_sentencepiece(
        corpus_path, corpus_dir / "tok", vocab_size=30, model_type="unigram"
    )
    return DaraLMTokenizer.from_pretrained(model_path)


@pytest.fixture(scope="module")
def tiny_model(tiny_tokenizer):
    config = ArchitectureConfig(
        vocab_size=tiny_tokenizer.vocab_size,
        hidden_size=16,
        num_layers=2,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=32,
    )
    model = DaraLMTransformer(config, pad_token_id=tiny_tokenizer.pad_id)
    model.eval()
    return model


def test_generate_respects_max_new_tokens(tiny_model, tiny_tokenizer):
    text = generate(tiny_model, tiny_tokenizer, prompt="hello", max_new_tokens=5, stop_on_eos=False)
    assert isinstance(text, str)
    assert len(text) > 0


def test_generate_greedy_is_deterministic(tiny_model, tiny_tokenizer):
    text_a = generate(
        tiny_model,
        tiny_tokenizer,
        prompt="hello world",
        max_new_tokens=10,
        temperature=0.0,
        stop_on_eos=False,
    )
    text_b = generate(
        tiny_model,
        tiny_tokenizer,
        prompt="hello world",
        max_new_tokens=10,
        temperature=0.0,
        stop_on_eos=False,
    )
    assert text_a == text_b


def test_generate_stops_on_eos_when_requested(tiny_model, tiny_tokenizer, monkeypatch):
    import daralm.inference.generator as generator_module

    def fake_sample(*args, **kwargs):
        return tiny_tokenizer.eos_id

    monkeypatch.setattr(generator_module, "sample_next_token", fake_sample)
    text = generate(tiny_model, tiny_tokenizer, prompt="hello", max_new_tokens=50, stop_on_eos=True)
    assert isinstance(text, str)


def test_generate_output_is_a_valid_string_for_various_prompts(tiny_model, tiny_tokenizer):
    for prompt in ["hello", "world test", "a"]:
        text = generate(tiny_model, tiny_tokenizer, prompt=prompt, max_new_tokens=5)
        assert isinstance(text, str)


def test_generate_chat_returns_a_string(tiny_model, tiny_tokenizer):
    text = generate_chat(tiny_model, tiny_tokenizer, instruction="hello", max_new_tokens=10)
    assert isinstance(text, str)


def test_generate_chat_response_never_contains_the_assistant_close_marker(
    tiny_model, tiny_tokenizer
):
    from daralm.data.chat_template import ASSISTANT_CLOSE

    text = generate_chat(tiny_model, tiny_tokenizer, instruction="hello world", max_new_tokens=30)
    assert ASSISTANT_CLOSE not in text


def test_generate_chat_stops_when_assistant_close_is_generated(
    tiny_model, tiny_tokenizer, monkeypatch
):
    import daralm.inference.generator as generator_module
    from daralm.data.chat_template import ASSISTANT_CLOSE

    monkeypatch.setattr(generator_module, "sample_next_token", lambda *a, **k: 0)

    real_decode = tiny_tokenizer.decode
    decoded_marker = real_decode(
        tiny_tokenizer.encode(ASSISTANT_CLOSE, add_bos=False, add_eos=False)
    )

    call_count = {"n": 0}

    def fake_decode(ids):
        call_count["n"] += 1
        if call_count["n"] >= 4:
            return f"a response {decoded_marker}"
        return real_decode(ids)

    monkeypatch.setattr(tiny_tokenizer, "decode", fake_decode)

    text = generate_chat(tiny_model, tiny_tokenizer, instruction="hello", max_new_tokens=100)
    assert call_count["n"] < 100  # stopped well before exhausting max_new_tokens
    assert decoded_marker not in text
