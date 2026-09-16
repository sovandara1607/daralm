"""Causal multi-head self-attention."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from daralm.model.embeddings import apply_rotary_pos_emb


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with Rotary Position Embeddings."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        max_position_embeddings: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by "
                f"num_attention_heads ({num_attention_heads})"
            )
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.max_position_embeddings = max_position_embeddings

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        # Applied via scaled_dot_product_attention's dropout_p below, not as a
        # submodule — SDPA's fused kernels never materialize attention weights
        # for an nn.Dropout to act on.
        self.attn_dropout_p = dropout
        self.resid_dropout = nn.Dropout(dropout)

        # Still exposed for the decode-with-cache mask below and for tests
        # asserting the mask shape directly.
        causal_mask = torch.tril(
            torch.ones(max_position_embeddings, max_position_embeddings, dtype=torch.bool)
        )
        self.register_buffer("causal_mask", causal_mask, persistent=False)

    def _split_heads(self, x: torch.Tensor, batch_size: int, seq_len: int) -> torch.Tensor:
        x = x.view(batch_size, seq_len, self.num_attention_heads, self.head_dim)
        return x.transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        rotary_cos: torch.Tensor,
        rotary_sin: torch.Tensor,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        batch_size, seq_len, _ = x.shape

        q = self._split_heads(self.q_proj(x), batch_size, seq_len)
        k = self._split_heads(self.k_proj(x), batch_size, seq_len)
        v = self._split_heads(self.v_proj(x), batch_size, seq_len)

        q, k = apply_rotary_pos_emb(q, k, rotary_cos, rotary_sin)

        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        present_key_value = (k, v) if use_cache else None

        past_len = k.size(2) - seq_len  # 0 when there's no cache yet

        if past_len == 0:
            # Full prefill: SDPA's own causal mask is exactly right, and cheaper
            # than materializing a boolean mask ourselves.
            attn_mask = None
            is_causal = True
        else:
            # Continuing generation from a cache: each new query sees every
            # cached key plus earlier keys within this chunk, never a later one.
            attn_mask = self.causal_mask[past_len : past_len + seq_len, : past_len + seq_len]
            is_causal = False

        dropout_p = self.attn_dropout_p if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal
        )
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        out = self.o_proj(out)
        return self.resid_dropout(out), present_key_value
