"""Tests for daralm.model.attention — CausalSelfAttention."""

from __future__ import annotations

import pytest
import torch

from daralm.model.attention import CausalSelfAttention
from daralm.model.embeddings import RotaryEmbedding, apply_rotary_pos_emb


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
    out, _ = attn(x, cos, sin)
    assert out.shape == x.shape


def test_rejects_hidden_size_not_divisible_by_heads():
    with pytest.raises(ValueError):
        CausalSelfAttention(hidden_size=30, num_attention_heads=4, max_position_embeddings=32)


@pytest.mark.parametrize("batch_size", [1, 2, 8])
def test_various_batch_sizes(batch_size):
    attn, rope = _make_attention()
    x = torch.randn(batch_size, 6, 32)
    cos, sin = rope(seq_len=6, device=x.device)
    out, _ = attn(x, cos, sin)
    assert out.shape == (batch_size, 6, 32)


@pytest.mark.parametrize("seq_len", [1, 5, 32, 64])
def test_various_sequence_lengths(seq_len):
    attn, rope = _make_attention(max_pos=64)
    x = torch.randn(2, seq_len, 32)
    cos, sin = rope(seq_len=seq_len, device=x.device)
    out, _ = attn(x, cos, sin)
    assert out.shape == (2, seq_len, 32)


def test_causal_mask_is_lower_triangular():
    attn, _ = _make_attention(max_pos=16)
    mask = attn.causal_mask
    assert mask.shape == (16, 16)
    assert torch.equal(mask, torch.tril(torch.ones(16, 16, dtype=torch.bool)))
    assert mask[0].sum().item() == 1
    assert mask[3].sum().item() == 4


def test_no_access_to_future_tokens():
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
        out_a, _ = attn(x_a, cos, sin)
        out_b, _ = attn(x_b, cos, sin)

    assert torch.allclose(out_a[:, :cutoff, :], out_b[:, :cutoff, :], atol=1e-6)
    assert not torch.allclose(out_a[:, cutoff:, :], out_b[:, cutoff:, :], atol=1e-6)


def test_dropout_zero_is_deterministic():
    torch.manual_seed(0)
    attn, rope = _make_attention(dropout=0.0)
    attn.eval()
    x = torch.randn(1, 5, 32)
    cos, sin = rope(seq_len=5, device=x.device)
    with torch.no_grad():
        out1, _ = attn(x, cos, sin)
        out2, _ = attn(x, cos, sin)
    assert torch.equal(out1, out2)


def test_no_bias_on_projections():
    attn, _ = _make_attention()
    assert attn.q_proj.bias is None
    assert attn.k_proj.bias is None
    assert attn.v_proj.bias is None
    assert attn.o_proj.bias is None


def _manual_attention_forward(attn: CausalSelfAttention, x, rotary_cos, rotary_sin):
    """Reference implementation matching the pre-SDPA math, for equivalence checks."""
    import math

    batch_size, seq_len, _ = x.shape
    q = attn._split_heads(attn.q_proj(x), batch_size, seq_len)
    k = attn._split_heads(attn.k_proj(x), batch_size, seq_len)
    v = attn._split_heads(attn.v_proj(x), batch_size, seq_len)
    q, k = apply_rotary_pos_emb(q, k, rotary_cos, rotary_sin)

    scores = (q @ k.transpose(-2, -1)) / math.sqrt(attn.head_dim)
    mask = attn.causal_mask[:seq_len, :seq_len]
    scores = scores.masked_fill(~mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    out = probs @ v
    out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
    return attn.o_proj(out)


def test_sdpa_matches_manual_attention_forward():
    """SDPA-based forward must produce the same logits as the original manual math."""
    torch.manual_seed(0)
    attn, rope = _make_attention(hidden_size=32, num_heads=4, max_pos=16, dropout=0.0)
    attn.eval()  # no dropout, deterministic comparison

    x = torch.randn(2, 8, 32)
    cos, sin = rope(seq_len=8, device=x.device)

    out_sdpa, _ = attn(x, cos, sin)
    out_manual = _manual_attention_forward(attn, x, cos, sin)

    assert torch.allclose(out_sdpa, out_manual, atol=1e-5)


def test_sdpa_matches_manual_attention_gradients():
    """Backward pass through SDPA must match the manual implementation's gradients."""
    torch.manual_seed(0)
    attn, rope = _make_attention(hidden_size=32, num_heads=4, max_pos=16, dropout=0.0)
    attn.eval()

    x = torch.randn(2, 8, 32, requires_grad=True)
    cos, sin = rope(seq_len=8, device=x.device)

    out_sdpa, _ = attn(x, cos, sin)
    out_sdpa.sum().backward()
    grad_sdpa = x.grad.clone()
    x.grad = None

    out_manual = _manual_attention_forward(attn, x, cos, sin)
    out_manual.sum().backward()
    grad_manual = x.grad.clone()

    assert torch.allclose(grad_sdpa, grad_manual, atol=1e-4)


def test_decode_with_cache_matches_full_prefill():
    """Feeding tokens one at a time through the KV cache must match a full prefill."""
    torch.manual_seed(0)
    attn, rope = _make_attention(hidden_size=32, num_heads=4, max_pos=16, dropout=0.0)
    attn.eval()

    seq_len = 5
    x = torch.randn(1, seq_len, 32)
    cos_full, sin_full = rope(seq_len=seq_len, device=x.device)
    with torch.no_grad():
        out_full, _ = attn(x, cos_full, sin_full)

    # Now replay the same tokens one at a time through the cache.
    past_key_value = None
    outs = []
    for t in range(seq_len):
        x_t = x[:, t : t + 1, :]
        cos_t, sin_t = rope(seq_len=1, device=x.device, offset=t)
        with torch.no_grad():
            out_t, past_key_value = attn(
                x_t, cos_t, sin_t, past_key_value=past_key_value, use_cache=True
            )
        outs.append(out_t)
    out_cached = torch.cat(outs, dim=1)

    assert torch.allclose(out_full, out_cached, atol=1e-5)
