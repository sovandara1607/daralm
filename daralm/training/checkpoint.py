"""Checkpointing: save everything needed to resume training exactly, or to
know precisely what produced a given checkpoint.

Per spec sections 12-13, a checkpoint is more than model weights — without
the optimizer/scheduler state, RNG state, step count, and the exact config
that produced it, "resume training" silently becomes "start a *slightly
different* training run that happens to reuse some weights", and a
checkpoint with no recorded config is a checkpoint nobody can safely reuse
or compare against later.

Layout on disk (per spec section 12):

    checkpoints/<model_name>/
      step-1000/
        checkpoint.pt   # model + optimizer + scheduler + step + RNG state
        config.yaml     # human-readable copy of the config that produced this
      step-5000/
        ...
      best/
        ...             # same contents, copied whenever val loss improves
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.optim.lr_scheduler import LRScheduler

from daralm.model.config import ModelConfig
from daralm.model.transformer import DaraLMTransformer
from daralm.utils.logging import get_logger

logger = get_logger(__name__)


def _file_sha256(path: str | Path) -> str:
    """Fingerprint a file's contents — used to detect tokenizer mismatches on resume."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_checkpoint(
    checkpoint_root: str | Path,
    tag: str,
    model: DaraLMTransformer,
    optimizer: torch.optim.Optimizer,
    scheduler: LRScheduler,
    step: int,
    tokens_processed: int,
    config: ModelConfig,
    tokenizer_path: str | Path,
) -> Path:
    """Save a full checkpoint under `checkpoint_root/<config.model_name>/<tag>/`.

    Returns the checkpoint directory path.
    """
    checkpoint_dir = Path(checkpoint_root) / config.model_name / tag
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    state: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "step": step,
        "tokens_processed": tokens_processed,
        "pad_token_id": model.pad_token_id,
        # Random seed states — restoring these (not just the model weights)
        # is what makes "resume" reproduce the *exact* continuation of a
        # run, not just a plausible-looking one.
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_state": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
        "tokenizer_path": str(tokenizer_path),
        "tokenizer_sha256": _file_sha256(tokenizer_path),
        "config": config.model_dump(),
    }
    torch.save(state, checkpoint_dir / "checkpoint.pt")

    with (checkpoint_dir / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(config.model_dump(), f, default_flow_style=False, sort_keys=False)

    logger.info("Saved checkpoint to %s (step=%d)", checkpoint_dir, step)
    return checkpoint_dir


def load_checkpoint(
    checkpoint_dir: str | Path,
    model: DaraLMTransformer,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: LRScheduler | None = None,
    tokenizer_path: str | Path | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load a checkpoint saved by `save_checkpoint`, restoring model/optimizer/
    scheduler/RNG state in place.

    Args:
        tokenizer_path: if given, verified against the checkpoint's recorded
            tokenizer fingerprint — raises `ValueError` on mismatch rather
            than silently resuming training against a different vocabulary
            than the checkpoint was produced with.

    Returns:
        A dict with `step`, `tokens_processed`, and `config` (as a plain
        dict) for the caller to resume the training loop from.

    Raises:
        FileNotFoundError: if the checkpoint file doesn't exist.
        ValueError: if `tokenizer_path` is given and its hash doesn't match
            the tokenizer this checkpoint was trained with.
    """
    checkpoint_path = Path(checkpoint_dir) / "checkpoint.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state = torch.load(checkpoint_path, map_location=map_location, weights_only=False)

    if tokenizer_path is not None:
        current_hash = _file_sha256(tokenizer_path)
        saved_hash = state["tokenizer_sha256"]
        if current_hash != saved_hash:
            raise ValueError(
                f"Tokenizer mismatch: checkpoint was trained with tokenizer "
                f"{state['tokenizer_path']} (sha256={saved_hash[:12]}...), but "
                f"{tokenizer_path} has a different fingerprint "
                f"(sha256={current_hash[:12]}...). Resuming with a different "
                "tokenizer would silently corrupt training."
            )

    model.load_state_dict(state["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    if scheduler is not None:
        scheduler.load_state_dict(state["scheduler_state_dict"])

    random.setstate(state["python_rng_state"])
    np.random.set_state(state["numpy_rng_state"])
    torch.set_rng_state(state["torch_rng_state"])
    if state["torch_cuda_rng_state"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda_rng_state"])

    logger.info("Loaded checkpoint from %s (step=%d)", checkpoint_path, state["step"])
    return {
        "step": state["step"],
        "tokens_processed": state["tokens_processed"],
        "config": state["config"],
    }


def write_meta_json(checkpoint_dir: str | Path, **fields: Any) -> None:
    """Write a small `meta.json` alongside a checkpoint for quick human/CLI inspection
    (e.g. loss/perplexity at save time) without having to `torch.load` the full checkpoint.
    """
    path = Path(checkpoint_dir) / "meta.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(fields, f, indent=2)


def find_latest_checkpoint(checkpoint_root: str | Path, model_name: str) -> Path | None:
    """Find the highest-step `step-N` checkpoint dir for `model_name`, for `--resume`.

    Returns None if no `step-*` checkpoints exist yet (a fresh run, not an
    error). Deliberately ignores `best/` — resuming training should
    continue from the most recent state, not silently rewind to whichever
    step happened to have the best validation loss.
    """
    model_dir = Path(checkpoint_root) / model_name
    if not model_dir.exists():
        return None

    step_dirs = []
    for entry in model_dir.iterdir():
        if entry.is_dir() and entry.name.startswith("step-"):
            try:
                step_num = int(entry.name.removeprefix("step-"))
            except ValueError:
                continue
            step_dirs.append((step_num, entry))

    if not step_dirs:
        return None
    return max(step_dirs, key=lambda pair: pair[0])[1]
