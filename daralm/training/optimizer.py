"""Optimizer construction.

Uses AdamW (or plain SGD, if a config asks for it) with one deliberate
refinement over the naive "pass every parameter the same weight decay":
parameters are split into two groups, and weight decay is applied only to
2D+ weight matrices (attention/FFN projections, the tied embedding), not to
RMSNorm's 1D scale vectors.

Why this matters: weight decay pulls every decayed parameter toward zero
each step, as a regularizer against overfitting. That's a sensible prior
for a big weight matrix (no individual weight "matters" on its own), but
RMSNorm's per-channel scale is a small, structurally important vector —
decaying it toward zero fights against what normalization is trying to do.
This split (decay matrices, not norms/biases) is standard practice across
GPT-2/GPT-3/LLaMA-style training code.
"""

from __future__ import annotations

import torch
from torch import nn

from daralm.model.config import TrainingConfig


def build_optimizer(model: nn.Module, config: TrainingConfig) -> torch.optim.Optimizer:
    """Construct the optimizer specified by `config.optimizer`.

    Raises `ValueError` for an optimizer name the config schema doesn't
    already restrict to (defensive — `Literal["adamw", "sgd"]` in
    `TrainingConfig` should make this unreachable, but a function taking a
    string shouldn't silently ignore an unrecognized one).
    """
    decay_params = []
    no_decay_params = []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        if param.dim() >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": config.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    if config.optimizer == "adamw":
        return torch.optim.AdamW(param_groups, lr=config.learning_rate, betas=(0.9, 0.95))
    if config.optimizer == "sgd":
        return torch.optim.SGD(param_groups, lr=config.learning_rate, momentum=0.9)
    raise ValueError(f"Unknown optimizer: {config.optimizer!r}")
