"""Tests for daralm.model.transformer — the full DaraLMTransformer."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest
import torch

from daralm.model.config import ArchitectureConfig
from daralm.model.transformer import DaraLMTransformer

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_inspect_model_config_module():
    """Import scripts/inspect_model_config.py as a module, by file path."""
    module_path = REPO_ROOT / "scripts" / "inspect_model_config.py"
    spec = importlib.util.spec_from_file_location("inspect_model_config", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["inspect_model_config"] = module
    spec.loader.exec_module(module)
    return module


def _tiny_architecture(**overrides) -> ArchitectureConfig:
    defaults = {
        "vocab_size": 100,
        "hidden_size": 32,
        "num_layers": 2,
        "num_attention_heads": 4,
        "intermediate_size": 64,
        "max_position_embeddings": 32,
        "dropout": 0.0,
    }
    defaults.update(overrides)
    return ArchitectureConfig(**defaults)


def test_forward_pass_output_shape():
    config = _tiny_architecture()
    model = DaraLMTransformer(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 10))
    output = model(input_ids)
    assert output.logits.shape == (2, 10, config.vocab_size)
    assert output.loss is None


def test_hidden_states_are_none_by_default():
    config = _tiny_architecture()
    model = DaraLMTransformer(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 10))
    output = model(input_ids)
    assert output.hidden_states is None


def test_hidden_states_returned_with_correct_shape_when_requested():
    config = _tiny_architecture()
    model = DaraLMTransformer(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 10))
    output = model(input_ids, return_hidden_states=True)
    assert output.hidden_states.shape == (2, 10, config.hidden_size)
    assert output.logits.shape == (2, 10, config.vocab_size)


def test_hidden_states_are_post_final_norm_pre_lm_head():
    config = _tiny_architecture()
    model = DaraLMTransformer(config)
    model.eval()
    input_ids = torch.randint(0, config.vocab_size, (2, 10))
    with torch.no_grad():
        output = model(input_ids, return_hidden_states=True)
        recomputed_logits = model.lm_head(output.hidden_states)
    assert torch.allclose(recomputed_logits, output.logits)


@pytest.mark.parametrize("batch_size", [1, 2, 8])
def test_various_batch_sizes(batch_size):
    config = _tiny_architecture()
    model = DaraLMTransformer(config)
    input_ids = torch.randint(0, config.vocab_size, (batch_size, 6))
    output = model(input_ids)
    assert output.logits.shape == (batch_size, 6, config.vocab_size)


@pytest.mark.parametrize("seq_len", [1, 5, 16, 32])
def test_various_sequence_lengths(seq_len):
    config = _tiny_architecture(max_position_embeddings=32)
    model = DaraLMTransformer(config)
    input_ids = torch.randint(0, config.vocab_size, (2, seq_len))
    output = model(input_ids)
    assert output.logits.shape == (2, seq_len, config.vocab_size)


def test_seq_len_beyond_max_position_embeddings_raises():
    config = _tiny_architecture(max_position_embeddings=8)
    model = DaraLMTransformer(config)
    input_ids = torch.randint(0, config.vocab_size, (1, 9))
    with pytest.raises(ValueError):
        model(input_ids)


def test_initial_loss_is_close_to_uniform_baseline():
    torch.manual_seed(0)
    config = _tiny_architecture(vocab_size=16000)
    model = DaraLMTransformer(config)
    input_ids = torch.randint(0, config.vocab_size, (4, 16))
    output = model(input_ids, labels=input_ids)

    expected = math.log(config.vocab_size)
    assert abs(output.loss.item() - expected) < 2.0


def test_loss_is_computed_when_labels_given():
    config = _tiny_architecture()
    model = DaraLMTransformer(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 10))
    output = model(input_ids, labels=input_ids)
    assert output.loss is not None
    assert output.loss.dim() == 0  # scalar
    assert output.loss.item() > 0


def test_loss_ignores_pad_tokens():
    config = _tiny_architecture(vocab_size=50)
    model = DaraLMTransformer(config, pad_token_id=0)
    input_ids = torch.randint(1, config.vocab_size, (1, 6))  # no pad tokens
    labels_with_padding = input_ids.clone()
    labels_with_padding[:, -2:] = 0  # pad out the last two target positions

    output_padded = model(input_ids, labels=labels_with_padding)
    assert torch.isfinite(output_padded.loss)


def test_loss_decreases_when_overfitting_one_batch():
    torch.manual_seed(0)
    config = _tiny_architecture(vocab_size=20, hidden_size=16, num_layers=2, num_attention_heads=2)
    model = DaraLMTransformer(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    input_ids = torch.randint(0, config.vocab_size, (4, 8))

    first_loss = None
    last_loss = None
    for step in range(50):
        optimizer.zero_grad()
        output = model(input_ids, labels=input_ids)
        output.loss.backward()
        optimizer.step()
        if step == 0:
            first_loss = output.loss.item()
        last_loss = output.loss.item()

    assert last_loss < first_loss


def test_lm_head_is_tied_to_token_embedding():
    config = _tiny_architecture()
    model = DaraLMTransformer(config)
    assert model.lm_head.weight is model.token_embedding.embedding.weight


def test_num_parameters_excludes_tied_weights_by_default():
    config = _tiny_architecture()
    model = DaraLMTransformer(config)
    tied_count = model.num_parameters(exclude_tied=True)
    untied_count = model.num_parameters(exclude_tied=False)
    assert untied_count > tied_count
    assert untied_count - tied_count == config.vocab_size * config.hidden_size


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"num_layers": 4, "hidden_size": 64, "num_attention_heads": 8, "intermediate_size": 128},
    ],
)
def test_param_count_matches_estimator(overrides):
    config = _tiny_architecture(**overrides)
    model = DaraLMTransformer(config)
    module = _load_inspect_model_config_module()
    estimated = module.estimate_parameters(config)
    actual = model.num_parameters(exclude_tied=True)
    assert actual == estimated


def test_real_configs_param_count_matches_estimator():
    from daralm.model.config import ModelConfig

    module = _load_inspect_model_config_module()
    for config_path in (REPO_ROOT / "configs" / "tiny.yaml", REPO_ROOT / "configs" / "50m.yaml"):
        config = ModelConfig.from_yaml(config_path)
        model = DaraLMTransformer(config.architecture)
        estimated = module.estimate_parameters(config.architecture)
        actual = model.num_parameters(exclude_tied=True)
        assert actual == estimated, f"{config_path}: estimated {estimated}, actual {actual}"


def test_full_model_has_no_access_to_future_tokens():
    torch.manual_seed(0)
    config = _tiny_architecture(dropout=0.0)
    model = DaraLMTransformer(config)
    model.eval()

    seq_len = 10
    cutoff = 5
    input_ids_a = torch.randint(0, config.vocab_size, (1, seq_len))
    input_ids_b = input_ids_a.clone()
    input_ids_b[:, cutoff:] = torch.randint(0, config.vocab_size, (1, seq_len - cutoff))

    with torch.no_grad():
        logits_a = model(input_ids_a).logits
        logits_b = model(input_ids_b).logits

    assert torch.allclose(logits_a[:, :cutoff, :], logits_b[:, :cutoff, :], atol=1e-5)


def test_cache_is_none_by_default():
    config = _tiny_architecture()
    model = DaraLMTransformer(config)
    input_ids = torch.randint(0, config.vocab_size, (1, 5))
    output = model(input_ids)
    assert output.past_key_values is None


def test_use_cache_returns_one_kv_pair_per_layer():
    config = _tiny_architecture(num_layers=3)
    model = DaraLMTransformer(config)
    input_ids = torch.randint(0, config.vocab_size, (1, 5))
    output = model(input_ids, use_cache=True)
    assert output.past_key_values is not None
    assert len(output.past_key_values) == 3
    for key, value in output.past_key_values:
        assert key.shape == (1, config.num_attention_heads, 5, config.head_dim)
        assert value.shape == (1, config.num_attention_heads, 5, config.head_dim)


def test_cached_incremental_generation_matches_full_recomputation():
    torch.manual_seed(0)
    config = _tiny_architecture(dropout=0.0)
    model = DaraLMTransformer(config)
    model.eval()

    seq_len = 6
    input_ids = torch.randint(0, config.vocab_size, (1, seq_len))

    with torch.no_grad():
        full_logits = model(input_ids).logits  # (1, seq_len, vocab_size), no cache

        past_key_values = None
        cached_logits = []
        for t in range(seq_len):
            step_output = model(
                input_ids[:, t : t + 1], past_key_values=past_key_values, use_cache=True
            )
            cached_logits.append(step_output.logits)
            past_key_values = step_output.past_key_values
        cached_logits = torch.cat(cached_logits, dim=1)

    assert torch.allclose(full_logits, cached_logits, atol=1e-5)


def test_cache_grows_by_one_position_per_step():
    config = _tiny_architecture()
    model = DaraLMTransformer(config)
    input_ids = torch.randint(0, config.vocab_size, (1, 3))

    output = model(input_ids, use_cache=True)
    assert output.past_key_values[0][0].size(2) == 3

    next_token = torch.randint(0, config.vocab_size, (1, 1))
    output2 = model(next_token, past_key_values=output.past_key_values, use_cache=True)
    assert output2.past_key_values[0][0].size(2) == 4


def test_cache_plus_new_tokens_beyond_max_position_raises():
    config = _tiny_architecture(max_position_embeddings=8)
    model = DaraLMTransformer(config)
    input_ids = torch.randint(0, config.vocab_size, (1, 6))
    output = model(input_ids, use_cache=True)  # cache length 6

    too_many_new = torch.randint(0, config.vocab_size, (1, 3))  # 6 + 3 > 8
    with pytest.raises(ValueError):
        model(too_many_new, past_key_values=output.past_key_values, use_cache=True)
