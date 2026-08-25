"""Tests for daralm.utils.device — device auto-detection."""

from __future__ import annotations

import pytest
import torch

from daralm.utils.device import get_device, get_device_name


def test_get_device_returns_torch_device():
    device = get_device()
    assert isinstance(device, torch.device)
    assert device.type in {"cuda", "mps", "cpu"}


def test_get_device_prefers_cpu_when_forced():
    device = get_device(prefer="cpu")
    assert device.type == "cpu"


def test_get_device_invalid_prefer_raises():
    with pytest.raises(ValueError):
        get_device(prefer="quantum")


def test_get_device_name_returns_nonempty_string():
    device = get_device(prefer="cpu")
    name = get_device_name(device)
    assert isinstance(name, str)
    assert len(name) > 0
    assert name == "CPU"
