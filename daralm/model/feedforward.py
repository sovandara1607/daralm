"""Position-wise feed-forward network (the "FFN" half of each Transformer block).

Where attention lets tokens exchange information *across* positions, the
feed-forward network processes each position *independently*, projecting it
up to a wider intermediate dimension, applying a nonlinearity, then
projecting back down. Attention is where the model reasons about
relationships between tokens; the FFN is where most of a Transformer's raw
parameter capacity (and, empirically, much of its factual/pattern storage)
lives.

Two linear layers, no bias (matching the `2 * hidden_size * intermediate_size`
term in `scripts/inspect_model_config.py`'s parameter estimate), with GELU
as the nonlinearity in between. GELU (used in GPT-2/GPT-3/BERT) is a smooth
approximation of "pass the input through if positive, zero-ish it out if
very negative" — smoother than ReLU, which empirically trains better for
Transformers. SwiGLU (a 3-matrix gated variant used in LLaMA/Mistral) is a
reasonable further upgrade, but would change the parameter-count formula
(3 matrices instead of 2) already established since Phase 0 — noted here as
a deliberate scope decision for this first implementation, not an oversight.
"""

from __future__ import annotations

import torch
from torch import nn


class FeedForward(nn.Module):
    """Linear(hidden -> intermediate) -> GELU -> Linear(intermediate -> hidden) -> Dropout."""

    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.activation = nn.GELU()
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up_proj(x)
        x = self.activation(x)
        x = self.down_proj(x)
        return self.dropout(x)
