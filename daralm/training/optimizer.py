"""Optimizer construction."""

from __future__ import annotations

import torch
from torch import nn

from daralm.model.config import TrainingConfig


def build_optimizer(model: nn.Module, config: TrainingConfig) -> torch.optim.Optimizer:
    """Construct the optimizer specified by `config.optimizer`."""
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
