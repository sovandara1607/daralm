"""Token embeddings and Rotary Position Embeddings (RoPE).

The architecture pipeline (see project README / spec) is:

    Token IDs -> Token Embedding -> Position Representation -> Transformer Blocks

`TokenEmbedding` handles the first arrow. RoPE handles the second — but
unlike the classic "add a learned position vector to the token embedding"
approach (GPT-2 style), RoPE does not touch the embedding at all. Instead
it rotates the Query and Key vectors *inside attention*, by an angle that
depends on token position. That's why `apply_rotary_pos_emb` is called
from `daralm.model.attention`, not from anywhere in the embedding forward
pass — RoPE has zero learned parameters and only exists at attention time.

Why RoPE over a learned absolute position embedding table:

1. It's parameter-free — no `max_position_embeddings x hidden_size` weight
   matrix to learn (which is also why `scripts/inspect_model_config.py`'s
   parameter estimate no longer includes a positional-embedding term, as
   of Phase 3).
2. It encodes *relative* position directly in the attention dot product:
   rotating q at position i and k at position j makes their dot product a
   function of (i - j) alone, not of i and j separately (see
   `tests/test_embeddings.py` for a test that verifies exactly this
   property). A learned absolute table has to learn that relative
   relationship indirectly, if at all.
"""

from __future__ import annotations

import torch
from torch import nn


class TokenEmbedding(nn.Module):
    """Maps token IDs to vectors, via a standard learned lookup table.

    Scales the output by sqrt(hidden_size), a convention from the original
    "Attention Is All You Need" paper: it keeps the embedding's scale
    comparable to the positional/attention signal it will be summed or
    compared against downstream, which otherwise starts out tiny relative
    to it since embeddings are typically initialized with unit-ish variance
    while hidden_size can be in the hundreds.
    """

    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(input_ids) * (self.hidden_size**0.5)


class RotaryEmbedding(nn.Module):
    """Precomputes the cos/sin rotation tables used by `apply_rotary_pos_emb`.

    Standard formulation (as in RoFormer / LLaMA): for each pair of
    dimensions in a head, rotate by an angle `position * inv_freq`, where
    `inv_freq` decays geometrically across the head's dimensions so
    different dimension-pairs rotate at different frequencies — similar in
    spirit to the sinusoidal frequencies of the original Transformer's
    fixed position embedding, but applied as a rotation instead of an
    additive vector.
    """

    def __init__(self, head_dim: int, max_position_embeddings: int, theta: float = 10000.0) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"RoPE requires an even head_dim, got {head_dim}")
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings

        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (cos, sin), each shaped (seq_len, head_dim)."""
        if seq_len > self.max_position_embeddings:
            raise ValueError(
                f"seq_len ({seq_len}) exceeds max_position_embeddings "
                f"({self.max_position_embeddings})"
            )
        positions = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(positions, self.inv_freq)  # (seq_len, head_dim / 2)
        # Duplicate across the two halves of head_dim so `rotate_half` below
        # can pair dimension i with dimension i + head_dim/2 directly.
        angles = torch.cat([freqs, freqs], dim=-1)  # (seq_len, head_dim)
        return angles.cos(), angles.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Split the last dim in half and swap+negate: (x1, x2) -> (-x2, x1).

    This "rotate half" trick, combined with the duplicated cos/sin from
    `RotaryEmbedding`, implements a standard 2D rotation independently on
    each of head_dim/2 dimension-pairs — the textbook rotation matrix
    [[cos, -sin], [sin, cos]] applied to (x1, x2), just written in a
    vectorized form that avoids an explicit loop over pairs.
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate query and key tensors by position-dependent angles.

    Args:
        q, k: shape (batch, num_heads, seq_len, head_dim)
        cos, sin: shape (seq_len, head_dim), from `RotaryEmbedding`

    Returns:
        (q_rotated, k_rotated), same shapes as q, k.
    """
    cos = cos.unsqueeze(0).unsqueeze(0)  # -> (1, 1, seq_len, head_dim), broadcasts over batch/heads
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_rotated = (q * cos) + (rotate_half(q) * sin)
    k_rotated = (k * cos) + (rotate_half(k) * sin)
    return q_rotated, k_rotated
