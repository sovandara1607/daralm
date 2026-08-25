"""Tests for daralm.training.checkpoint — save/resume correctness.

Per spec section 16: "train -> save -> reload -> generate produces valid
behavior." These tests exercise that exact loop at a tiny scale.
"""

from __future__ import annotations

import pytest
import torch

from daralm.model.config import ModelConfig
from daralm.model.transformer import DaraLMTransformer
from daralm.training.checkpoint import find_latest_checkpoint, load_checkpoint, save_checkpoint
from daralm.training.optimizer import build_optimizer
from daralm.training.scheduler import build_scheduler


def _tiny_model_config() -> ModelConfig:
    return ModelConfig(
        model_name="test-checkpoint-model",
        architecture={
            "vocab_size": 50,
            "hidden_size": 16,
            "num_layers": 2,
            "num_attention_heads": 2,
            "intermediate_size": 32,
            "max_position_embeddings": 16,
        },
        training={
            "learning_rate": 0.001,
            "batch_size": 2,
            "max_steps": 20,
            "warmup_steps": 2,
            "eval_interval": 5,
            "save_interval": 5,
        },
    )


def _make_tokenizer_file(tmp_path) -> str:
    """A checkpoint records a tokenizer fingerprint, but doesn't need a real
    SentencePiece model to test that mechanism — any file with bytes works.
    """
    path = tmp_path / "fake_tokenizer.model"
    path.write_bytes(b"fake-tokenizer-bytes-v1")
    return str(path)


def test_save_and_load_roundtrip_restores_model_weights(tmp_path):
    config = _tiny_model_config()
    model = DaraLMTransformer(config.architecture)
    optimizer = build_optimizer(model, config.training)
    scheduler = build_scheduler(optimizer, config.training)
    tokenizer_path = _make_tokenizer_file(tmp_path)

    checkpoint_dir = save_checkpoint(
        tmp_path / "checkpoints", "step-10", model, optimizer, scheduler, 10, 1000, config,
        tokenizer_path,
    )

    # A fresh model with different (freshly re-initialized) weights.
    new_model = DaraLMTransformer(config.architecture)
    assert not torch.equal(
        next(model.parameters()), next(new_model.parameters())
    ), "test setup invariant broken: fresh model happened to match by chance"

    result = load_checkpoint(checkpoint_dir, new_model, tokenizer_path=tokenizer_path)

    assert result["step"] == 10
    assert result["tokens_processed"] == 1000
    for p_original, p_loaded in zip(model.parameters(), new_model.parameters(), strict=True):
        assert torch.equal(p_original, p_loaded)


def test_save_and_load_roundtrip_restores_optimizer_and_scheduler(tmp_path):
    config = _tiny_model_config()
    model = DaraLMTransformer(config.architecture)
    optimizer = build_optimizer(model, config.training)
    scheduler = build_scheduler(optimizer, config.training)
    tokenizer_path = _make_tokenizer_file(tmp_path)

    # Take a few real steps so optimizer/scheduler have non-initial state.
    for _ in range(3):
        optimizer.zero_grad()
        input_ids = torch.randint(0, config.architecture.vocab_size, (2, 8))
        output = model(input_ids, labels=input_ids)
        output.loss.backward()
        optimizer.step()
        scheduler.step()
    lr_before_save = scheduler.get_last_lr()[0]

    checkpoint_dir = save_checkpoint(
        tmp_path / "checkpoints", "step-3", model, optimizer, scheduler, 3, 300, config,
        tokenizer_path,
    )

    new_model = DaraLMTransformer(config.architecture)
    new_optimizer = build_optimizer(new_model, config.training)
    new_scheduler = build_scheduler(new_optimizer, config.training)
    load_checkpoint(checkpoint_dir, new_model, new_optimizer, new_scheduler, tokenizer_path)

    assert new_scheduler.get_last_lr()[0] == lr_before_save
    assert new_scheduler.last_epoch == scheduler.last_epoch


def test_load_checkpoint_missing_file_raises(tmp_path):
    config = _tiny_model_config()
    model = DaraLMTransformer(config.architecture)

    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "nonexistent", model)


def test_load_checkpoint_tokenizer_mismatch_raises(tmp_path):
    config = _tiny_model_config()
    model = DaraLMTransformer(config.architecture)
    optimizer = build_optimizer(model, config.training)
    scheduler = build_scheduler(optimizer, config.training)
    tokenizer_path = _make_tokenizer_file(tmp_path)

    checkpoint_dir = save_checkpoint(
        tmp_path / "checkpoints", "step-1", model, optimizer, scheduler, 1, 100, config,
        tokenizer_path,
    )

    different_tokenizer = tmp_path / "different_tokenizer.model"
    different_tokenizer.write_bytes(b"totally-different-bytes")


    with pytest.raises(ValueError, match="Tokenizer mismatch"):
        load_checkpoint(checkpoint_dir, model, tokenizer_path=str(different_tokenizer))


def test_config_yaml_is_written_and_reloadable(tmp_path):
    config = _tiny_model_config()
    model = DaraLMTransformer(config.architecture)
    optimizer = build_optimizer(model, config.training)
    scheduler = build_scheduler(optimizer, config.training)
    tokenizer_path = _make_tokenizer_file(tmp_path)

    checkpoint_dir = save_checkpoint(
        tmp_path / "checkpoints", "step-1", model, optimizer, scheduler, 1, 100, config,
        tokenizer_path,
    )
    reloaded_config = ModelConfig.from_yaml(checkpoint_dir / "config.yaml")
    assert reloaded_config.model_name == config.model_name
    assert reloaded_config.architecture.hidden_size == config.architecture.hidden_size


def test_find_latest_checkpoint_picks_highest_step(tmp_path):
    config = _tiny_model_config()
    model = DaraLMTransformer(config.architecture)
    optimizer = build_optimizer(model, config.training)
    scheduler = build_scheduler(optimizer, config.training)
    tokenizer_path = _make_tokenizer_file(tmp_path)
    checkpoint_root = tmp_path / "checkpoints"

    for step in (5, 20, 10):
        save_checkpoint(
            checkpoint_root, f"step-{step}", model, optimizer, scheduler, step, step * 100,
            config, tokenizer_path,
        )
    save_checkpoint(
        checkpoint_root, "best", model, optimizer, scheduler, 999, 99900, config, tokenizer_path
    )

    latest = find_latest_checkpoint(checkpoint_root, config.model_name)
    assert latest is not None
    assert latest.name == "step-20"  # not "best", even though its step (999) is higher


def test_find_latest_checkpoint_returns_none_when_no_checkpoints(tmp_path):
    assert find_latest_checkpoint(tmp_path / "checkpoints", "nonexistent-model") is None
