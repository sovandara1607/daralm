"""Token embeddings and Rotary Position Embeddings (RoPE)."""

from __future__ import annotations

import torch
from torch import nn


class TokenEmbedding(nn.Module):
    """Maps token IDs to vectors, via a standard learned lookup table."""

    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(input_ids) * (self.hidden_size**0.5)


class RotaryEmbedding(nn.Module):
    """Precomputes the cos/sin rotation tables used by `apply_rotary_pos_emb`."""

    def __init__(self, head_dim: int, max_position_embeddings: int, theta: float = 10000.0) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"RoPE requires an even head_dim, got {head_dim}")
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings

        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self, seq_len: int, device: torch.device, offset: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (cos, sin), each shaped (seq_len, head_dim)."""
        if offset + seq_len > self.max_position_embeddings:
            raise ValueError(
                f"offset + seq_len ({offset + seq_len}) exceeds max_position_embeddings "
                f"({self.max_position_embeddings})"
            )
        positions = torch.arange(offset, offset + seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(positions, self.inv_freq)  # (seq_len, head_dim / 2)
        angles = torch.cat([freqs, freqs], dim=-1)  # (seq_len, head_dim)
        return angles.cos(), angles.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Split the last dim in half and swap+negate: (x1, x2) -> (-x2, x1)."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate query and key tensors by position-dependent angles."""
    cos = cos.unsqueeze(0).unsqueeze(0)  # -> (1, 1, seq_len, head_dim), broadcasts over batch/heads
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_rotated = (q * cos) + (rotate_half(q) * sin)
    k_rotated = (k * cos) + (rotate_half(k) * sin)
    return q_rotated, k_rotated
