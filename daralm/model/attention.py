"""Causal multi-head self-attention.

Why attention uses separate Query/Key/Value projections: attention lets
every position gather information from earlier positions, weighted by how
"relevant" each earlier position is. Q ("what am I looking for"), K ("what
do I contain, for others to match against"), and V ("what do I actually
offer once matched") are three different *roles* the same token plays —
using one shared projection for all three would force a token's search
query and its offered content to be the same vector, which is a much more
restrictive model. Splitting them into three learned projections lets the
network use different subspaces for "searching" versus "being found"
versus "being read".

Why a causal mask exists: DaraLM is trained to predict the next token from
only the tokens before it (see spec section 10 — causal language modeling).
If position i's attention could see position i+1's key/value, the model
could "cheat" during training by copying the very answer it's supposed to
predict, and would then be useless at actual generation time, where future
tokens don't exist yet. The causal mask enforces, structurally, that
position i's output can only be a function of positions <= i.

Implementation note: attention here is written out explicitly (project Q/K/V
-> scaled dot product -> mask -> softmax -> weighted sum of V) rather than
delegating to a fused kernel like `torch.nn.functional.scaled_dot_product_attention`
or FlashAttention. That's deliberate for this phase — the whole point of
DaraLM is to understand what attention actually computes, not just call a
library function that computes it. Swapping in a fused/FlashAttention
kernel for speed is explicitly a later optimization phase (spec section 20,
"Phase 3: Flash Attention, torch.compile"), not a Phase-3-architecture
concern.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from daralm.model.embeddings import apply_rotary_pos_emb


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with Rotary Position Embeddings.

    No bias terms on any of the Q/K/V/O projections (matches the parameter
    estimate in `scripts/inspect_model_config.py`: `4 * hidden_size**2` per
    layer, four square projections, no bias vectors).
    """

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

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Precomputed once at max size; sliced down to the actual sequence
        # length at forward time. `persistent=False` keeps it out of the
        # checkpoint state_dict — it's derived, not learned.
        causal_mask = torch.tril(
            torch.ones(max_position_embeddings, max_position_embeddings, dtype=torch.bool)
        )
        self.register_buffer("causal_mask", causal_mask, persistent=False)

    def _split_heads(self, x: torch.Tensor, batch_size: int, seq_len: int) -> torch.Tensor:
        # (batch, seq_len, hidden_size) -> (batch, num_heads, seq_len, head_dim)
        x = x.view(batch_size, seq_len, self.num_attention_heads, self.head_dim)
        return x.transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        rotary_cos: torch.Tensor,
        rotary_sin: torch.Tensor,
    ) -> torch.Tensor:
        """Args:
            x: (batch, seq_len, hidden_size)
            rotary_cos, rotary_sin: (seq_len, head_dim), from `RotaryEmbedding`

        Returns:
            (batch, seq_len, hidden_size)
        """
        batch_size, seq_len, _ = x.shape

        q = self._split_heads(self.q_proj(x), batch_size, seq_len)
        k = self._split_heads(self.k_proj(x), batch_size, seq_len)
        v = self._split_heads(self.v_proj(x), batch_size, seq_len)

        q, k = apply_rotary_pos_emb(q, k, rotary_cos, rotary_sin)

        # Scaled dot-product attention scores: (batch, heads, seq_len, seq_len).
        # Scaling by 1/sqrt(head_dim) keeps the dot products (and therefore
        # the softmax) from growing too large in magnitude as head_dim
        # grows, which would otherwise push softmax into a near-one-hot,
        # near-zero-gradient regime.
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        mask = self.causal_mask[:seq_len, :seq_len]
        scores = scores.masked_fill(~mask, float("-inf"))

        probs = torch.softmax(scores, dim=-1)
        probs = self.attn_dropout(probs)

        out = probs @ v  # (batch, heads, seq_len, head_dim)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        out = self.o_proj(out)
        return self.resid_dropout(out)
