"""Tests for daralm.model.attention — CausalSelfAttention.

Per spec section 16: output dimensions, causal masking, and — the most
important property — that a position's output genuinely has no access to
future tokens.
"""

from __future__ import annotations

import pytest
import torch

from daralm.model.attention import CausalSelfAttention
from daralm.model.embeddings import RotaryEmbedding


def _make_attention(hidden_size=32, num_heads=4, max_pos=64, dropout=0.0):
    attn = CausalSelfAttention(
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        max_position_embeddings=max_pos,
        dropout=dropout,
    )
    rope = RotaryEmbedding(head_dim=hidden_size // num_heads, max_position_embeddings=max_pos)
    return attn, rope


def test_output_shape_matches_input():
    attn, rope = _make_attention(hidden_size=32, num_heads=4)
    x = torch.randn(2, 10, 32)
    cos, sin = rope(seq_len=10, device=x.device)
    out = attn(x, cos, sin)
    assert out.shape == x.shape


def test_rejects_hidden_size_not_divisible_by_heads():
    with pytest.raises(ValueError):
        CausalSelfAttention(
            hidden_size=30, num_attention_heads=4, max_position_embeddings=32
        )


@pytest.mark.parametrize("batch_size", [1, 2, 8])
def test_various_batch_sizes(batch_size):
    attn, rope = _make_attention()
    x = torch.randn(batch_size, 6, 32)
    cos, sin = rope(seq_len=6, device=x.device)
    out = attn(x, cos, sin)
    assert out.shape == (batch_size, 6, 32)


@pytest.mark.parametrize("seq_len", [1, 5, 32, 64])
def test_various_sequence_lengths(seq_len):
    attn, rope = _make_attention(max_pos=64)
    x = torch.randn(2, seq_len, 32)
    cos, sin = rope(seq_len=seq_len, device=x.device)
    out = attn(x, cos, sin)
    assert out.shape == (2, seq_len, 32)


def test_causal_mask_is_lower_triangular():
    attn, _ = _make_attention(max_pos=16)
    mask = attn.causal_mask
    assert mask.shape == (16, 16)
    assert torch.equal(mask, torch.tril(torch.ones(16, 16, dtype=torch.bool)))
    # Position 0 can only attend to itself; position 3 can attend to 0..3.
    assert mask[0].sum().item() == 1
    assert mask[3].sum().item() == 4


def test_no_access_to_future_tokens():
    """The defining causal property: changing a *future* token must not
    change an earlier position's output at all.

    We run the same attention module on two inputs that are identical up
    to position `t`, but differ afterward, and check outputs at position
    `t` (and everything before it) are exactly the same.
    """
    torch.manual_seed(0)
    attn, rope = _make_attention(hidden_size=32, num_heads=4, max_pos=16, dropout=0.0)
    attn.eval()  # disable dropout for a deterministic comparison

    seq_len = 8
    cutoff = 4  # positions 0..cutoff-1 are shared; positions cutoff.. differ

    x_a = torch.randn(1, seq_len, 32)
    x_b = x_a.clone()
    x_b[:, cutoff:, :] = torch.randn(1, seq_len - cutoff, 32)  # perturb only the future

    cos, sin = rope(seq_len=seq_len, device=x_a.device)
    with torch.no_grad():
        out_a = attn(x_a, cos, sin)
        out_b = attn(x_b, cos, sin)

    # Positions before the cutoff must be unaffected by the perturbation.
    assert torch.allclose(out_a[:, :cutoff, :], out_b[:, :cutoff, :], atol=1e-6)
    # Sanity check the test itself is meaningful: positions at/after the
    # cutoff (which now see different inputs at their own position) should
    # generally differ.
    assert not torch.allclose(out_a[:, cutoff:, :], out_b[:, cutoff:, :], atol=1e-6)


def test_dropout_zero_is_deterministic():
    torch.manual_seed(0)
    attn, rope = _make_attention(dropout=0.0)
    attn.eval()
    x = torch.randn(1, 5, 32)
    cos, sin = rope(seq_len=5, device=x.device)
    with torch.no_grad():
        out1 = attn(x, cos, sin)
        out2 = attn(x, cos, sin)
    assert torch.equal(out1, out2)


def test_no_bias_on_projections():
    attn, _ = _make_attention()
    assert attn.q_proj.bias is None
    assert attn.k_proj.bias is None
    assert attn.v_proj.bias is None
    assert attn.o_proj.bias is None
