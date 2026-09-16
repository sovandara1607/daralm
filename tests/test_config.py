"""Tests for daralm.model.config — the configuration schema and loader."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from daralm.model.config import ArchitectureConfig, ModelConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
TINY_CONFIG_PATH = REPO_ROOT / "configs" / "tiny.yaml"


def valid_architecture_dict() -> dict:
    return {
        "vocab_size": 8000,
        "hidden_size": 128,
        "num_layers": 2,
        "num_attention_heads": 4,
        "intermediate_size": 512,
        "max_position_embeddings": 256,
    }


def valid_training_dict() -> dict:
    return {
        "learning_rate": 0.0003,
        "batch_size": 8,
        "max_steps": 100,
        "warmup_steps": 10,
        "eval_interval": 20,
        "save_interval": 50,
    }


def valid_model_dict() -> dict:
    return {
        "model_name": "test-model",
        "architecture": valid_architecture_dict(),
        "training": valid_training_dict(),
    }


def test_valid_config_loads_from_yaml():
    config = ModelConfig.from_yaml(TINY_CONFIG_PATH)
    assert config.model_name == "daralm-tiny"
    assert config.architecture.hidden_size == 128
    assert config.architecture.num_layers == 2
    assert config.architecture.num_attention_heads == 4
    assert config.architecture.max_position_embeddings == 256
    assert config.training.optimizer == "adamw"


def test_hidden_size_not_divisible_by_heads_raises():
    bad = valid_architecture_dict()
    bad["hidden_size"] = 100
    bad["num_attention_heads"] = 7
    with pytest.raises(ValidationError):
        ArchitectureConfig(**bad)


def test_missing_required_field_raises():
    bad = valid_architecture_dict()
    del bad["vocab_size"]
    with pytest.raises(ValidationError):
        ArchitectureConfig(**bad)


def test_unknown_field_raises():
    bad = valid_model_dict()
    bad["not_a_real_field"] = "typo"
    with pytest.raises(ValidationError):
        ModelConfig(**bad)


def test_zero_hidden_size_raises():
    bad = valid_architecture_dict()
    bad["hidden_size"] = 0
    with pytest.raises(ValidationError):
        ArchitectureConfig(**bad)


def test_negative_learning_rate_raises():
    bad = valid_model_dict()
    bad["training"]["learning_rate"] = -0.1
    with pytest.raises(ValidationError):
        ModelConfig(**bad)


def test_head_dim_property():
    arch = ArchitectureConfig(**valid_architecture_dict())
    assert arch.head_dim == arch.hidden_size // arch.num_attention_heads
    assert arch.head_dim == 32


def test_from_yaml_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        ModelConfig.from_yaml(REPO_ROOT / "configs" / "does_not_exist.yaml")


def test_warmup_steps_beyond_max_steps_raises():
    bad = valid_model_dict()
    bad["training"]["warmup_steps"] = 100
    bad["training"]["max_steps"] = 100
    with pytest.raises(ValidationError):
        ModelConfig(**bad)


def test_10m_config_loads():
    config = ModelConfig.from_yaml(REPO_ROOT / "configs" / "10m.yaml")
    assert config.model_name == "daralm-10m"
    assert 16_000 <= config.architecture.vocab_size <= 32_000
    assert config.training.precision == "bf16"


def test_50m_config_loads():
    config = ModelConfig.from_yaml(REPO_ROOT / "configs" / "50m.yaml")
    assert config.model_name == "daralm-50m"
    assert 16_000 <= config.architecture.vocab_size <= 32_000
    assert config.training.precision == "bf16"


def test_gradient_checkpointing_defaults_to_false():
    config = ModelConfig(**valid_model_dict())
    assert config.training.gradient_checkpointing is False


def test_gradient_checkpointing_can_be_enabled():
    raw = valid_model_dict()
    raw["training"]["gradient_checkpointing"] = True
    config = ModelConfig(**raw)
    assert config.training.gradient_checkpointing is True


def test_compile_defaults_to_false():
    config = ModelConfig(**valid_model_dict())
    assert config.training.compile is False


def test_compile_can_be_enabled():
    raw = valid_model_dict()
    raw["training"]["compile"] = True
    config = ModelConfig(**raw)
    assert config.training.compile is True
