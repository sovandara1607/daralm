"""RMSNorm — the normalization layer used throughout DaraLM's Transformer blocks.

Why normalization at all: as a signal flows through many stacked layers,
its scale can drift (grow or shrink) layer over layer, which destabilizes
training (gradients explode or vanish). Normalizing the activations back to
a consistent scale before each sub-layer keeps training stable regardless
of depth.

Why RMSNorm instead of the original Transformer's LayerNorm: LayerNorm
re-centers *and* re-scales its input (subtract the mean, divide by the
standard deviation, then apply a learned scale and bias). RMSNorm drops the
re-centering step entirely — it only rescales by the root-mean-square of
the input, with a single learned per-channel scale and no bias. Empirically
(as used in LLaMA, Mistral, and most modern LLMs) this loses essentially
nothing in model quality while being cheaper to compute and having half as
many parameters (`hidden_size` instead of `2 * hidden_size` — no bias term,
no mean to subtract). This is also the assumption `scripts/inspect_model_config.py`
has used for its parameter-count estimate since Phase 0.
"""

from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    """Root-Mean-Square Layer Normalization.

    out = x / sqrt(mean(x^2, dim=-1) + eps) * weight
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute in float32 for numerical stability regardless of the
        # input's dtype (matters under fp16/bf16 mixed-precision training,
        # where the mean-of-squares can lose precision) then cast back.
        input_dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (x.to(input_dtype)) * self.weight
