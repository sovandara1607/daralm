"""Device selection: auto-detect the best available compute device.

Never assume CUDA exists. Training and inference code should call
``get_device()`` once and thread the result through, rather than each
module independently guessing what hardware is available.

Priority order: CUDA (NVIDIA GPU) > MPS (Apple Silicon) > CPU. CPU is
always usable — worst case, everything just runs slower — so this
function never raises for lack of hardware, only for an unrecognized
``prefer`` override.
"""

from __future__ import annotations

import torch


def get_device(prefer: str | None = None) -> torch.device:
    """Return the best available ``torch.device``.

    Args:
        prefer: Force a specific device ("cuda", "mps", or "cpu") instead of
            auto-detecting — useful for debugging on CPU even when a GPU is
            present. ``None`` (default) auto-detects: CUDA, then MPS, then
            CPU.

    Returns:
        A ``torch.device`` ready to use with ``.to(device)``.

    Raises:
        ValueError: if ``prefer`` is not one of "cuda", "mps", "cpu".
    """
    valid_devices = {"cuda", "mps", "cpu"}
    if prefer is not None:
        if prefer not in valid_devices:
            raise ValueError(f"Unknown device '{prefer}'; expected one of {sorted(valid_devices)}")
        return torch.device(prefer)

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_device_name(device: torch.device) -> str:
    """Return a human-readable label for ``device``.

    Examples: "NVIDIA GeForce RTX 4090 (CUDA)", "Apple Silicon (MPS)", "CPU".
    """
    if device.type == "cuda":
        index = device.index if device.index is not None else 0
        return f"{torch.cuda.get_device_name(index)} (CUDA)"
    if device.type == "mps":
        return "Apple Silicon (MPS)"
    return "CPU"
