"""Tests for daralm.evaluation.inference_benchmark — the Model Optimization
before/after measurement harness. Tiny fixture model, no real checkpoint."""

from __future__ import annotations

import pytest

from daralm.evaluation.inference_benchmark import benchmark_generation
from daralm.model.config import ArchitectureConfig
from daralm.model.transformer import DaraLMTransformer
from daralm.tokenizer.tokenizer import DaraLMTokenizer
from daralm.tokenizer.train import train_sentencepiece


@pytest.fixture(scope="module")
def tiny_tokenizer(tmp_path_factory):
    corpus_dir = tmp_path_factory.mktemp("bench_corpus")
    corpus_path = corpus_dir / "corpus.txt"
    lines = ["the quick brown fox jumps over the lazy dog and runs away"] * 60
    corpus_path.write_text("\n".join(lines), encoding="utf-8")
    model_path = train_sentencepiece(
        corpus_path, corpus_dir / "tok", vocab_size=32, model_type="unigram"
    )
    return DaraLMTokenizer.from_pretrained(model_path)


@pytest.fixture
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


def test_benchmark_returns_expected_keys(tiny_model, tiny_tokenizer):
    result = benchmark_generation(
        tiny_model, tiny_tokenizer, prompts=["the quick"], max_new_tokens=5, num_repeats=2
    )
    assert set(result) == {
        "num_samples",
        "tokens_per_sec",
        "latency_p50_ms",
        "latency_p95_ms",
        "peak_memory_gb",
        "total_tokens_generated",
        "total_time_s",
    }


def test_num_samples_is_prompts_times_repeats(tiny_model, tiny_tokenizer):
    result = benchmark_generation(
        tiny_model,
        tiny_tokenizer,
        prompts=["the quick", "brown fox"],
        max_new_tokens=5,
        num_repeats=3,
    )
    assert result["num_samples"] == 2 * 3


def test_tokens_per_sec_is_positive_for_real_generation(tiny_model, tiny_tokenizer):
    result = benchmark_generation(
        tiny_model, tiny_tokenizer, prompts=["the quick"], max_new_tokens=10, num_repeats=1
    )
    assert result["tokens_per_sec"] > 0
    assert result["total_tokens_generated"] > 0


def test_latency_p95_is_at_least_p50(tiny_model, tiny_tokenizer):
    result = benchmark_generation(
        tiny_model, tiny_tokenizer, prompts=["the quick"], max_new_tokens=5, num_repeats=5
    )
    assert result["latency_p95_ms"] >= result["latency_p50_ms"] >= 0


def test_empty_prompts_raises(tiny_model, tiny_tokenizer):
    with pytest.raises(ValueError):
        benchmark_generation(tiny_model, tiny_tokenizer, prompts=[], max_new_tokens=5)


def test_zero_repeats_raises(tiny_model, tiny_tokenizer):
    with pytest.raises(ValueError):
        benchmark_generation(
            tiny_model, tiny_tokenizer, prompts=["the quick"], max_new_tokens=5, num_repeats=0
        )


def test_restores_original_training_mode(tiny_model, tiny_tokenizer):
    tiny_model.train()
    benchmark_generation(
        tiny_model, tiny_tokenizer, prompts=["the quick"], max_new_tokens=5, num_repeats=1
    )
    assert tiny_model.training is True

    tiny_model.eval()
    benchmark_generation(
        tiny_model, tiny_tokenizer, prompts=["the quick"], max_new_tokens=5, num_repeats=1
    )
    assert tiny_model.training is False


def test_peak_memory_gb_is_none_on_cpu(tiny_model, tiny_tokenizer):
    # matches daralm.utils.device.peak_memory_gb's documented CPU behavior
    assert next(tiny_model.parameters()).device.type == "cpu"
    result = benchmark_generation(
        tiny_model, tiny_tokenizer, prompts=["the quick"], max_new_tokens=5, num_repeats=1
    )
    assert result["peak_memory_gb"] is None
