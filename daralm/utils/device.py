"""Device selection: auto-detect the best available compute device."""

from __future__ import annotations

import torch


def get_device(prefer: str | None = None) -> torch.device:
    """Return the best available ``torch.device``."""
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
    """Return a human-readable label for ``device``."""
    if device.type == "cuda":
        index = device.index if device.index is not None else 0
        return f"{torch.cuda.get_device_name(index)} (CUDA)"
    if device.type == "mps":
        return "Apple Silicon (MPS)"
    return "CPU"


def peak_memory_gb(device: torch.device) -> float | None:
    """Peak allocated device memory in GB, or None if not measurable (e.g. plain CPU)."""
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 1024**3
    if device.type == "mps":
        # MPS reports current allocation; CUDA also reports peak allocation.
        return torch.mps.current_allocated_memory() / 1024**3
    return None
