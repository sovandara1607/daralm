"""Reproducibility: seed every source of randomness from a single entry point.

Training runs are only reproducible if every RNG in the stack — Python's
own ``random``, NumPy, and PyTorch (CPU and CUDA) — is seeded consistently.
Forgetting even one of these means "rerun with the same seed" silently
stops being true, which makes debugging (and comparing experiments)
unreliable. Call ``set_seed`` once, early, before any data shuffling or
model initialization.
"""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch (CPU + CUDA) with the same value.

    Args:
        seed: The seed value, recorded alongside experiment metadata so a
            run can be reproduced later.
        deterministic: If True, also force PyTorch's CUDA backend into fully
            deterministic (but slower) algorithms via
            ``torch.backends.cudnn.deterministic = True`` and
            ``torch.backends.cudnn.benchmark = False``. Off by default —
            most of the time you want speed, and full bitwise determinism
            only matters when you're specifically debugging a
            reproducibility issue.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
