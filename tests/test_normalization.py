"""Tests for daralm.model.normalization — RMSNorm."""

from __future__ import annotations

import torch

from daralm.model.normalization import RMSNorm


def test_output_shape_matches_input():
    norm = RMSNorm(hidden_size=16)
    x = torch.randn(2, 5, 16)
    out = norm(x)
    assert out.shape == x.shape


def test_rms_norm_matches_manual_computation():
    torch.manual_seed(0)
    hidden_size = 8
    norm = RMSNorm(hidden_size, eps=1e-6)
    x = torch.randn(1, 1, hidden_size)

    expected_rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    expected = (x / expected_rms) * norm.weight

    out = norm(x)
    assert torch.allclose(out, expected, atol=1e-5)


def test_unit_weight_normalizes_to_unit_rms():
    norm = RMSNorm(hidden_size=32)
    x = torch.randn(4, 10, 32) * 100.0  # large arbitrary scale
    out = norm(x)
    rms = out.pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-3)


def test_zero_input_does_not_produce_nan():
    norm = RMSNorm(hidden_size=8)
    x = torch.zeros(1, 1, 8)
    out = norm(x)
    assert not torch.isnan(out).any()


def test_weight_is_learnable_parameter():
    norm = RMSNorm(hidden_size=8)
    assert isinstance(norm.weight, torch.nn.Parameter)
    assert norm.weight.requires_grad
