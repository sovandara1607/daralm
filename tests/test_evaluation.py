"""Tests for daralm.evaluation — perplexity and memorization checking."""

from __future__ import annotations

import math

import pytest
import torch
from torch.utils.data import DataLoader

from daralm.evaluation.generation import average_match_rate, check_memorization, token_match_rate
from daralm.evaluation.perplexity import compute_perplexity
from daralm.model.config import ArchitectureConfig
from daralm.model.transformer import DaraLMTransformer
from daralm.tokenizer.tokenizer import DaraLMTokenizer
from daralm.tokenizer.train import train_sentencepiece

# --- token_match_rate --------------------------------------------------


def test_token_match_rate_identical_sequences():
    assert token_match_rate([1, 2, 3], [1, 2, 3]) == 1.0


def test_token_match_rate_completely_different():
    assert token_match_rate([9, 9, 9], [1, 2, 3]) == 0.0


def test_token_match_rate_partial_match():
    assert token_match_rate([1, 2, 9], [1, 2, 3]) == pytest.approx(2 / 3)


def test_token_match_rate_empty_reference():
    assert token_match_rate([1, 2, 3], []) == 0.0


def test_token_match_rate_generated_shorter_than_reference_is_penalized():
    # Only 1 of 4 reference positions was even reached.
    assert token_match_rate([1], [1, 2, 3, 4]) == pytest.approx(0.25)


# --- compute_perplexity -----------------------------------------------


def _tiny_model_and_loader():
    config = ArchitectureConfig(
        vocab_size=30,
        hidden_size=16,
        num_layers=2,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=16,
    )
    model = DaraLMTransformer(config)
    data = torch.randint(0, config.vocab_size, (20, 8))
    loader = DataLoader(data, batch_size=4)
    return model, loader


def test_compute_perplexity_matches_exp_of_loss():
    torch.manual_seed(0)
    model, loader = _tiny_model_and_loader()
    avg_loss, perplexity = compute_perplexity(model, loader, torch.device("cpu"))
    assert perplexity == pytest.approx(math.exp(avg_loss), rel=1e-4)


def test_compute_perplexity_restores_training_mode():
    model, loader = _tiny_model_and_loader()
    model.train()
    compute_perplexity(model, loader, torch.device("cpu"))
    assert model.training is True

    model.eval()
    compute_perplexity(model, loader, torch.device("cpu"))
    assert model.training is False


def test_compute_perplexity_empty_loader_raises():
    model, _ = _tiny_model_and_loader()
    empty_loader = DataLoader(torch.empty(0, 8, dtype=torch.long), batch_size=4)
    with pytest.raises(ValueError):
        compute_perplexity(model, empty_loader, torch.device("cpu"))


# --- check_memorization --------------------------------------------------


@pytest.fixture(scope="module")
def tiny_tokenizer(tmp_path_factory):
    corpus_dir = tmp_path_factory.mktemp("eval_corpus")
    corpus_path = corpus_dir / "corpus.txt"
    lines = ["the quick brown fox jumps over the lazy dog"] * 50
    corpus_path.write_text("\n".join(lines), encoding="utf-8")
    model_path = train_sentencepiece(
        corpus_path, corpus_dir / "tok", vocab_size=32, model_type="unigram"
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
    return DaraLMTransformer(config, pad_token_id=tiny_tokenizer.pad_id)


def test_check_memorization_returns_one_result_per_valid_document(tiny_model, tiny_tokenizer):
    documents = ["the quick brown fox jumps over the lazy dog and runs away quickly"] * 3
    results = check_memorization(tiny_model, tiny_tokenizer, documents, min_tokens=5)
    assert len(results) == 3
    for r in results:
        assert 0.0 <= r["token_match_rate"] <= 1.0
        assert r["num_reference_tokens"] > 0


def test_check_memorization_skips_short_documents(tiny_model, tiny_tokenizer):
    results = check_memorization(tiny_model, tiny_tokenizer, ["hi"], min_tokens=50)
    assert results == []


def test_check_memorization_caps_reference_length(tiny_model, tiny_tokenizer):
    # A long, repetitive document so tokenization yields far more than the cap.
    long_doc = "the quick brown fox jumps over the lazy dog. " * 50
    results = check_memorization(
        tiny_model, tiny_tokenizer, [long_doc], min_tokens=5, max_reference_tokens=10
    )
    assert len(results) == 1
    assert results[0]["num_reference_tokens"] <= 10


def test_check_memorization_restores_model_mode(tiny_model, tiny_tokenizer):
    tiny_model.train()
    check_memorization(tiny_model, tiny_tokenizer, ["the quick brown fox"], min_tokens=2)
    assert tiny_model.training is True


def test_average_match_rate_empty_results():
    assert average_match_rate([]) == 0.0


def test_average_match_rate_computes_mean():
    results = [{"token_match_rate": 0.2}, {"token_match_rate": 0.8}]
    assert average_match_rate(results) == pytest.approx(0.5)
