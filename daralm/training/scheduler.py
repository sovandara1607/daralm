"""Learning-rate schedule: linear warmup, then cosine decay.

Why warmup matters: at the very start of training, the model's weights are
freshly initialized and gradients can be large and noisy — taking full-size
optimizer steps immediately risks a bad, hard-to-recover-from update before
the model has "settled". Warmup ramps the learning rate up from ~0 to its
target value over the first `warmup_steps` steps, giving the optimizer a
chance to take small, cautious steps while the loss landscape near
initialization is least understood.

Why cosine decay after that: having reached the target (peak) learning
rate, we want it to shrink smoothly toward the end of training rather than
staying high (which would keep the model "bouncing around" a good solution
without settling into it) or dropping off a cliff (which under-uses the
budget for steady improvement). A cosine curve decays slowly at first, then
faster through the middle, then slowly again near the end — empirically a
good fit for most LLM training runs, and simpler to reason about than
step-decay or a bespoke schedule.
"""

from __future__ import annotations

import math

import torch
from torch.optim.lr_scheduler import LambdaLR

from daralm.model.config import TrainingConfig


def build_scheduler(optimizer: torch.optim.Optimizer, config: TrainingConfig) -> LambdaLR:
    """Build a warmup + cosine-decay LR scheduler as a learning-rate multiplier.

    Returns a `LambdaLR` whose multiplier goes:
      0 -> 1.0 linearly over `warmup_steps`, then
      1.0 -> `min_lr_ratio` following a cosine curve over the remaining steps.

    `TrainingConfig.check_warmup_within_max_steps` already guarantees
    `warmup_steps < max_steps`, so the cosine phase always has at least one
    step to operate over.
    """
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
