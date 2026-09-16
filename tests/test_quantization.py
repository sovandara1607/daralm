"""Tests for daralm.inference.quantization — post-training dynamic int8 quantization."""

from __future__ import annotations

import io

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from daralm.evaluation.perplexity import compute_perplexity
from daralm.inference.quantization import quantize_dynamic_int8
from daralm.model.config import ArchitectureConfig
from daralm.model.transformer import DaraLMTransformer


@pytest.fixture
def tiny_model():
    config = ArchitectureConfig(
        vocab_size=64,
        hidden_size=32,
        num_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=32,
    )
    model = DaraLMTransformer(config, pad_token_id=0)
    model.eval()
    return model


@pytest.fixture
def tiny_dataloader():
    torch.manual_seed(0)
    tokens = torch.randint(low=1, high=64, size=(20, 16))  # avoid pad id 0 dominating loss
    return DataLoader(TensorDataset(tokens), batch_size=4)


def _unwrap(batch):
    """TensorDataset wraps each item in a 1-tuple."""
    (tokens,) = batch
    return tokens


def test_quantize_produces_valid_logits_shape(tiny_model):
    quantized = quantize_dynamic_int8(tiny_model)
    input_ids = torch.randint(0, 64, (2, 10))
    with torch.no_grad():
        output = quantized(input_ids)
    assert output.logits.shape == (2, 10, 64)


def test_quantize_requires_cpu_model(tiny_model):
    """Dynamic quantization has no MPS/CUDA backend — must fail loudly, not silently."""
    fake_device = torch.device("meta")
    tiny_model.to(fake_device)
    with pytest.raises(ValueError, match="CPU"):
        quantize_dynamic_int8(tiny_model)


def test_quantize_shrinks_serialized_size(tiny_model):
    original_buffer = io.BytesIO()
    torch.save(tiny_model.state_dict(), original_buffer)
    original_size = original_buffer.tell()

    quantized = quantize_dynamic_int8(tiny_model)
    quantized_buffer = io.BytesIO()
    torch.save(quantized.state_dict(), quantized_buffer)
    quantized_size = quantized_buffer.tell()

    assert quantized_size < original_size


def test_quantize_keeps_perplexity_within_tolerance(tiny_model, tiny_dataloader):
    device = torch.device("cpu")
    batches = [_unwrap(b) for b in tiny_dataloader]

    class _Loader:
        def __iter__(self):
            return iter(batches)

    _, original_ppl = compute_perplexity(tiny_model, _Loader(), device)

    quantized = quantize_dynamic_int8(tiny_model)
    _, quantized_ppl = compute_perplexity(quantized, _Loader(), device)

    # A randomly-initialized tiny model's perplexity is near the vocab-size chance
    # baseline either way; the real assertion is that quantization doesn't blow it
    # up disproportionately — 5% relative tolerance, matching Stage 1's plan.
    relative_delta = abs(quantized_ppl - original_ppl) / original_ppl
    assert relative_delta < 0.05, (
        f"quantized perplexity {quantized_ppl:.2f} vs original {original_ppl:.2f} "
        f"({relative_delta:.1%} relative delta, tolerance 5%)"
    )


def test_quantize_puts_model_in_eval_mode(tiny_model):
    tiny_model.train()
    quantize_dynamic_int8(tiny_model)
    assert tiny_model.training is False
