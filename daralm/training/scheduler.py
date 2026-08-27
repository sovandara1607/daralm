"""Learning-rate schedule: linear warmup, then cosine decay."""

from __future__ import annotations

import math

import torch
from torch.optim.lr_scheduler import LambdaLR

from daralm.model.config import TrainingConfig


def build_scheduler(optimizer: torch.optim.Optimizer, config: TrainingConfig) -> LambdaLR:
    """Build a warmup + cosine-decay LR scheduler as a learning-rate multiplier."""
    warmup_steps = config.warmup_steps
    max_steps = config.max_steps
    min_lr_ratio = config.min_lr_ratio

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        progress = min(progress, 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda=lr_lambda)
