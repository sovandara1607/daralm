from __future__ import annotations

import torch
from torch import nn

from daralm.model.attention import CausalSelfAttention
from daralm.model.feedforward import FeedForward
from daralm.model.normalization import RMSNorm


class TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        intermediate_size: int,
        max_position_embeddings: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(hidden_size)
        self.attention = CausalSelfAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            max_position_embeddings=max_position_embeddings,
            dropout=dropout,
        )
        self.ffn_norm = RMSNorm(hidden_size)
        self.feed_forward = FeedForward(hidden_size, intermediate_size, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        rotary_cos: torch.Tensor,
        rotary_sin: torch.Tensor,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        attn_out, present_key_value = self.attention(
            self.attention_norm(x), rotary_cos, rotary_sin, past_key_value, use_cache
        )
        x = x + attn_out
        x = x + self.feed_forward(self.ffn_norm(x))
        return x, present_key_value
