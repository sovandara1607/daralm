"""Tests for daralm.model.embeddings — TokenEmbedding and RoPE."""

from __future__ import annotations

import pytest
import torch

from daralm.model.embeddings import (
    RotaryEmbedding,
    TokenEmbedding,
    apply_rotary_pos_emb,
    rotate_half,
)

# --- TokenEmbedding ---------------------------------------------------------


def test_token_embedding_output_shape():
    emb = TokenEmbedding(vocab_size=100, hidden_size=16)
    input_ids = torch.randint(0, 100, (2, 5))
    out = emb(input_ids)
    assert out.shape == (2, 5, 16)


def test_token_embedding_is_scaled_by_sqrt_hidden_size():
    torch.manual_seed(0)
    hidden_size = 64
    emb = TokenEmbedding(vocab_size=10, hidden_size=hidden_size)
    input_ids = torch.tensor([[0]])
    scaled_out = emb(input_ids)
    raw_out = emb.embedding(input_ids)
    assert torch.allclose(scaled_out, raw_out * (hidden_size**0.5))


# --- RotaryEmbedding / rotate_half ------------------------------------------


def test_rotary_embedding_output_shapes():
    rope = RotaryEmbedding(head_dim=8, max_position_embeddings=32)
    cos, sin = rope(seq_len=10, device=torch.device("cpu"))
    assert cos.shape == (10, 8)
    assert sin.shape == (10, 8)


def test_rotary_embedding_rejects_odd_head_dim():
    with pytest.raises(ValueError):
        RotaryEmbedding(head_dim=7, max_position_embeddings=32)


def test_rotary_embedding_rejects_seq_len_beyond_max():
    rope = RotaryEmbedding(head_dim=8, max_position_embeddings=16)
    with pytest.raises(ValueError):
        rope(seq_len=17, device=torch.device("cpu"))


def test_rotate_half_shape_preserved():
    x = torch.randn(2, 3, 5, 8)
    assert rotate_half(x).shape == x.shape


def test_rotate_half_known_values():
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    # first half [1,2], second half [3,4] -> [-3,-4, 1, 2]
    expected = torch.tensor([[-3.0, -4.0, 1.0, 2.0]])
    assert torch.equal(rotate_half(x), expected)


def test_apply_rotary_pos_emb_preserves_shape():
    batch, heads, seq_len, head_dim = 2, 4, 6, 8
    q = torch.randn(batch, heads, seq_len, head_dim)
    k = torch.randn(batch, heads, seq_len, head_dim)
    rope = RotaryEmbedding(head_dim=head_dim, max_position_embeddings=32)
    cos, sin = rope(seq_len, device=q.device)
    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape


def test_rope_preserves_vector_norm():
    # Rotation should not change a vector's length — a basic correctness
    # check that this is really a rotation, not an arbitrary transform.
    batch, heads, seq_len, head_dim = 1, 1, 5, 8
    torch.manual_seed(0)
    q = torch.randn(batch, heads, seq_len, head_dim)
    rope = RotaryEmbedding(head_dim=head_dim, max_position_embeddings=32)
    cos, sin = rope(seq_len, device=q.device)
    q_rot, _ = apply_rotary_pos_emb(q, q.clone(), cos, sin)
    original_norms = q.norm(dim=-1)
    rotated_norms = q_rot.norm(dim=-1)
    assert torch.allclose(original_norms, rotated_norms, atol=1e-5)


def test_rope_dot_product_depends_only_on_relative_position():
    """The defining property of RoPE: after rotation, (rotated_q_i . rotated_k_j)
    depends only on (i - j), not on the absolute positions i, j themselves.

    We rotate the *same* q, k vectors at two different absolute position
    pairs that share the same offset (0, 2) and (5, 7) — both have
    relative offset 2 — and check the resulting dot products match.
    """
    head_dim = 8
    torch.manual_seed(0)
    q_vec = torch.randn(1, 1, 1, head_dim)
    k_vec = torch.randn(1, 1, 1, head_dim)

    rope = RotaryEmbedding(head_dim=head_dim, max_position_embeddings=32)
    cos_all, sin_all = rope(seq_len=10, device=q_vec.device)

    def rotated_dot(pos_q: int, pos_k: int) -> torch.Tensor:
        cos_q, sin_q = cos_all[pos_q : pos_q + 1], sin_all[pos_q : pos_q + 1]
        cos_k, sin_k = cos_all[pos_k : pos_k + 1], sin_all[pos_k : pos_k + 1]
        q_rot = q_vec * cos_q + rotate_half(q_vec) * sin_q
        k_rot = k_vec * cos_k + rotate_half(k_vec) * sin_k
        return (q_rot * k_rot).sum()

    dot_offset_2_at_0 = rotated_dot(pos_q=2, pos_k=0)
    dot_offset_2_at_5 = rotated_dot(pos_q=7, pos_k=5)

    assert torch.allclose(dot_offset_2_at_0, dot_offset_2_at_5, atol=1e-4)
