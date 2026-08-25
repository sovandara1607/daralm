"""One decoder Transformer block: pre-norm attention + pre-norm feed-forward,
each wrapped in a residual connection.

    x -> RMSNorm -> CausalSelfAttention -> (+x) -> RMSNorm -> FeedForward -> (+x)

Two design choices worth calling out:

Pre-normalization (norm *before* the sub-layer, not after): in the original
"Attention Is All You Need" Transformer, normalization came after the
residual add ("post-norm"). Post-norm Transformers are notoriously unstable
to train at depth without a careful learning-rate warmup, because gradients
have to flow back through the normalization at every layer. Pre-norm
(GPT-2 onward) instead normalizes only the input *to* each sub-layer, so
the residual stream itself is never renormalized — gradients have a clean,
unimpeded path all the way back through the residual connections, which is
substantially more stable and is now the standard choice.

Residual connections (the `+ x`): without them, stacking many layers makes
both the forward signal and backward gradient have to pass through every
single layer's transformation, and either can shrink or blow up
exponentially with depth. The `+x` gives the network a direct, always-on
path that bypasses each sub-layer, so a layer only has to learn a
*correction* to its input rather than needing to reproduce/preserve it
first — this is what actually makes deep Transformers trainable.
"""

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
    ) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x), rotary_cos, rotary_sin)
        x = x + self.feed_forward(self.ffn_norm(x))
        return x
