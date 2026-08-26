"""Tests for daralm.model.transformer — the full DaraLMTransformer.

Per spec section 16 ("Transformer" tests): forward pass, loss calculation,
different batch sizes, different sequence sizes. Also verifies weight tying
and cross-checks the real parameter count against the analytical estimator
in `scripts/inspect_model_config.py` — the promise made in that script's
docstring since Phase 0.
"""

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
    """Import scripts/inspect_model_config.py as a module, by file path.

    `scripts/` isn't a Python package (its files are standalone CLI entry
    points, not importable modules in the normal sense — see Phase 0's
    file-tree notes), so we load it directly rather than adding it to
    `sys.path` and risking name collisions with other tools.
    """
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


# --- forward pass / shapes --------------------------------------------------


def test_forward_pass_output_shape():
    config = _tiny_architecture()
    model = DaraLMTransformer(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 10))
    output = model(input_ids)
    assert output.logits.shape == (2, 10, config.vocab_size)
    assert output.loss is None


def test_hidden_states_are_none_by_default():
    # return_hidden_states defaults to False — every existing call site
    # (causal-LM training, generate()) is unaffected by this being added.
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
    # Still returns logits/loss as normal — additive, not a replacement.
    assert output.logits.shape == (2, 10, config.vocab_size)


def test_hidden_states_are_post_final_norm_pre_lm_head():
    # The returned hidden_states must be exactly what lm_head was applied
    # to, not some other intermediate tensor — verified by re-running
    # lm_head on the returned hidden_states and checking it reproduces
    # the real logits exactly.
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


# --- weight initialization -----------------------------------------------


def test_initial_loss_is_close_to_uniform_baseline():
    """A freshly-initialized, untrained model should predict roughly
    uniformly over the vocabulary, giving a loss near ln(vocab_size).

    This is a regression test for a real bug caught during development:
    without a proper (GPT-2-style) weight init, PyTorch's defaults left
    logits with far too much variance at init — observed losses of ~127
    (tiny) and ~510 (50m) instead of ~9.7 for vocab_size=16,000. See
    `DaraLMTransformer._init_weights`'s docstring for the full story.
    """
    torch.manual_seed(0)
    config = _tiny_architecture(vocab_size=16000)
    model = DaraLMTransformer(config)
    input_ids = torch.randint(0, config.vocab_size, (4, 16))
    output = model(input_ids, labels=input_ids)

    expected = math.log(config.vocab_size)
    # Generous tolerance — this checks "in the right ballpark", not an
    # exact value, since it depends on the random seed and model size.
    assert abs(output.loss.item() - expected) < 2.0


# --- loss --------------------------------------------------------------


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
    # Loss should be finite and computed only over non-pad targets.
    assert torch.isfinite(output_padded.loss)


def test_loss_decreases_when_overfitting_one_batch():
    """A minimal sanity check in the spirit of spec section 18/19: a tiny
    model should be able to drive down its loss on a single fixed batch.
    """
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


# --- weight tying --------------------------------------------------------


def test_lm_head_is_tied_to_token_embedding():
    config = _tiny_architecture()
    model = DaraLMTransformer(config)
    assert model.lm_head.weight is model.token_embedding.embedding.weight


def test_num_parameters_excludes_tied_weights_by_default():
    config = _tiny_architecture()
    model = DaraLMTransformer(config)
    tied_count = model.num_parameters(exclude_tied=True)
    untied_count = model.num_parameters(exclude_tied=False)
    # The tied embedding/LM-head matrix is counted twice when not excluded.
    assert untied_count > tied_count
    assert untied_count - tied_count == config.vocab_size * config.hidden_size


# --- parameter count matches the Phase 0 estimator --------------------------


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
    """End-to-end check against the actual shipped configs, not just synthetic ones."""
    from daralm.model.config import ModelConfig

    module = _load_inspect_model_config_module()
    for config_path in (REPO_ROOT / "configs" / "tiny.yaml", REPO_ROOT / "configs" / "50m.yaml"):
        config = ModelConfig.from_yaml(config_path)
        model = DaraLMTransformer(config.architecture)
        estimated = module.estimate_parameters(config.architecture)
        actual = model.num_parameters(exclude_tied=True)
        assert actual == estimated, f"{config_path}: estimated {estimated}, actual {actual}"


# --- end-to-end causal property --------------------------------------------


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
