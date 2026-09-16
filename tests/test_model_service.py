"""Tests for api.services.model_service.ModelService.from_checkpoint, focused on
the Stage 1 DARALM_QUANTIZE opt-in wiring (existing loading/serving behavior is
already covered end-to-end via tests/test_api.py's fixture)."""

from __future__ import annotations

import asyncio

import pytest
import torch

import api.services.model_service as model_service_module
from api.services.model_service import ModelService
from daralm.model.config import ArchitectureConfig, ModelConfig, TrainingConfig
from daralm.model.transformer import DaraLMTransformer
from daralm.tokenizer.tokenizer import DaraLMTokenizer
from daralm.tokenizer.train import train_sentencepiece
from daralm.training.checkpoint import save_checkpoint
from daralm.training.optimizer import build_optimizer
from daralm.training.scheduler import build_scheduler


@pytest.fixture(scope="module")
def tiny_tokenizer(tmp_path_factory):
    corpus_dir = tmp_path_factory.mktemp("service_corpus")
    corpus_path = corpus_dir / "corpus.txt"
    lines = ["the quick brown fox jumps over the lazy dog and runs away"] * 60
    corpus_path.write_text("\n".join(lines), encoding="utf-8")
    model_path = train_sentencepiece(
        corpus_path, corpus_dir / "tok", vocab_size=32, model_type="unigram"
    )
    return model_path, DaraLMTokenizer.from_pretrained(model_path)


@pytest.fixture
def tiny_checkpoint(tmp_path, tiny_tokenizer):
    tokenizer_path, tokenizer = tiny_tokenizer
    config = ModelConfig(
        model_name="test-service-model",
        architecture=ArchitectureConfig(
            vocab_size=tokenizer.vocab_size,
            hidden_size=16,
            num_layers=2,
            num_attention_heads=2,
            intermediate_size=32,
            max_position_embeddings=16,
        ),
        training=TrainingConfig(
            learning_rate=0.001,
            batch_size=2,
            max_steps=10,
            warmup_steps=1,
            eval_interval=5,
            save_interval=10,
        ),
    )
    model = DaraLMTransformer(config.architecture, pad_token_id=tokenizer.pad_id)
    optimizer = build_optimizer(model, config.training)
    scheduler = build_scheduler(optimizer, config.training)

    checkpoint_root = tmp_path / "checkpoints"
    best_dir = save_checkpoint(
        checkpoint_root, "best", model, optimizer, scheduler, 10, 1000, config, tokenizer_path
    )
    return best_dir, tokenizer_path


def test_from_checkpoint_loads_unquantized_by_default(tiny_checkpoint, monkeypatch):
    best_dir, tokenizer_path = tiny_checkpoint
    monkeypatch.delenv("DARALM_QUANTIZE", raising=False)

    service = ModelService.from_checkpoint(best_dir / "config.yaml", best_dir, tokenizer_path)

    assert service.quantized is False
    assert service.model_info()["quantized"] is False
    assert service.model_info()["parameters"] > 0


def test_from_checkpoint_quantizes_on_cpu_when_opted_in(tiny_checkpoint, monkeypatch):
    best_dir, tokenizer_path = tiny_checkpoint
    monkeypatch.setenv("DARALM_QUANTIZE", "1")
    # get_device() picks MPS on this dev Mac; force CPU so quantization actually
    # activates (there's no MPS dynamic-quantization backend to test against).
    monkeypatch.setattr(model_service_module, "get_device", lambda: torch.device("cpu"))

    service = ModelService.from_checkpoint(best_dir / "config.yaml", best_dir, tokenizer_path)

    assert service.quantized is True
    info = service.model_info()
    assert info["quantized"] is True
    # parameter count must come from the pre-quantization model, not the packed
    # quantized weights (see ModelService.__init__'s num_parameters comment).
    assert info["parameters"] > 0


def test_quantized_service_still_generates(tiny_checkpoint, monkeypatch):
    best_dir, tokenizer_path = tiny_checkpoint
    monkeypatch.setenv("DARALM_QUANTIZE", "1")
    # get_device() picks MPS on this dev Mac; force CPU so quantization actually
    # activates (there's no MPS dynamic-quantization backend to test against).
    monkeypatch.setattr(model_service_module, "get_device", lambda: torch.device("cpu"))

    service = ModelService.from_checkpoint(best_dir / "config.yaml", best_dir, tokenizer_path)
    result = asyncio.run(
        service.generate(
            prompt="the quick",
            max_new_tokens=5,
            temperature=0.8,
            top_p=0.9,
            top_k=0,
            repetition_penalty=1.0,
            stop_on_eos=True,
        )
    )
    assert isinstance(result["generated_text"], str)
    assert result["generated_text"] != ""
