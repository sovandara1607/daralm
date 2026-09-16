"""Post-training dynamic int8 quantization for CPU serving.

`torch.ao.quantization.quantize_dynamic` quantizes `nn.Linear` weights to int8 and
requantizes activations on the fly at inference time. It's CPU-only — PyTorch's
dynamic quantization backend has no MPS/CUDA kernels — which matches this
project's documented CPU-only Linux deployment target (see `requirements.txt`'s
header comment). No new dependency: this is part of core `torch`.
"""

from __future__ import annotations

import torch
from torch import nn

from daralm.model.transformer import DaraLMTransformer
from daralm.utils.logging import get_logger

logger = get_logger(__name__)

# Preference order when torch hasn't already selected an engine: x86/fbgemm for
# the Intel/AMD servers this project's Docker image targets, qnnpack for the
# ARM dev Mac. PyTorch defaults `torch.backends.quantized.engine` to "none" on a
# fresh process even when a usable engine is compiled in — quantize_dynamic
# fails outright ("Didn't find engine ... NoQEngine") until one is selected.
_ENGINE_PREFERENCE = ("x86", "fbgemm", "qnnpack")


def _ensure_quantization_engine() -> None:
    if torch.backends.quantized.engine != "none":
        return
    supported = torch.backends.quantized.supported_engines
    usable = [engine for engine in _ENGINE_PREFERENCE if engine in supported]
    if not usable:
        raise RuntimeError(
            "No usable PyTorch quantized-ops engine on this build "
            f"(supported: {supported}) — dynamic quantization is unavailable here."
        )
    torch.backends.quantized.engine = usable[0]
    logger.info("Selected torch quantization engine: %r", usable[0])


def quantize_dynamic_int8(model: DaraLMTransformer) -> nn.Module:
    """Return a dynamically int8-quantized copy of `model`'s `nn.Linear` layers.

    Raises if `model` isn't already on a CPU device — call `model.to("cpu")`
    first; there is no MPS/CUDA dynamic-quantization backend to fall back to.
    """
    device = next(model.parameters()).device
    if device.type != "cpu":
        raise ValueError(
            f"Dynamic int8 quantization requires a CPU model (got device={device.type!r}); "
            "PyTorch's dynamic quantization backend has no MPS/CUDA kernels. "
            "Call model.to('cpu') first."
        )
    model.eval()  # quantization assumes inference-mode behavior (e.g. no dropout)
    _ensure_quantization_engine()
    num_linear = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
    quantized = torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    logger.info("Quantized %d nn.Linear layers to int8 (dynamic, CPU)", num_linear)
    return quantized
