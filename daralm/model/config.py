"""Configuration schema for DaraLM model architecture and training.

This module is the single source of truth for what constitutes a *valid*
model configuration. Every model size (tiny, 10m, 50m, 150m, 500m, ...) is
expressed as a YAML file validated against these schemas — no architecture
parameters are ever hard-coded into model code. That is what lets us swap
`configs/tiny.yaml` for `configs/50m.yaml` without touching a line of
Python once the Transformer itself is implemented (Phase 3).

We use Pydantic (not plain dataclasses) specifically for two things a
project like this needs from day one:

1. Fail loudly on bad configs. `extra="forbid"` means a typo'd key like
   `hiddne_size` raises a clear ``ValidationError`` instead of being
   silently ignored — you find out immediately, not three hours into a
   training run.
2. Cross-field validation. E.g. `hidden_size` must be evenly divisible by
   `num_attention_heads` (each head gets an equal slice of the hidden
   dimension) — a `model_validator` catches this before it becomes a
   confusing shape-mismatch error deep inside attention.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class ArchitectureConfig(BaseModel):
    """Transformer architecture hyperparameters.

    These fully determine the model's shape and parameter count. Nothing
    here is specific to any one model size — `configs/tiny.yaml` and
    `configs/50m.yaml` populate the exact same fields with different
    numbers.
    """

    model_config = {"extra": "forbid"}

    type: Literal["decoder_only_transformer"] = "decoder_only_transformer"
    vocab_size: int = Field(gt=0, description="Tokenizer vocabulary size")
    hidden_size: int = Field(gt=0, description="Model/embedding dimension (d_model)")
    num_layers: int = Field(gt=0, description="Number of transformer decoder blocks")
    num_attention_heads: int = Field(gt=0, description="Number of attention heads")
    intermediate_size: int = Field(gt=0, description="Feed-forward hidden dimension")
    max_position_embeddings: int = Field(gt=0, description="Maximum context length")
    dropout: float = Field(ge=0.0, le=1.0, default=0.0, description="Dropout probability")

    @model_validator(mode="after")
    def check_head_divisibility(self) -> ArchitectureConfig:
        """hidden_size must split evenly across attention heads.

        Multi-head attention slices the hidden dimension into
        `num_attention_heads` equal chunks (`head_dim = hidden_size //
        num_attention_heads`). If it doesn't divide evenly, that slicing is
        undefined — better to reject the config now than to fail with a
        cryptic tensor-shape error inside attention later.
        """
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_attention_heads ({self.num_attention_heads}); "
                f"got remainder {self.hidden_size % self.num_attention_heads}"
            )
        return self

    @model_validator(mode="after")
    def check_head_dim_even(self) -> ArchitectureConfig:
        """head_dim must be even — required by Rotary Position Embeddings (Phase 3).

        RoPE rotates each head's vector by splitting it into two equal
        halves (`daralm.model.embeddings.rotate_half`); an odd head_dim
        can't be split evenly. Discovered while implementing Phase 3 and
        added here so a bad config is rejected at load time, not deep
        inside the attention forward pass.
        """
        if self.head_dim % 2 != 0:
            raise ValueError(
                f"head_dim (hidden_size / num_attention_heads = {self.head_dim}) must be "
                "even — Rotary Position Embeddings split each head's vector in half."
            )
        return self

    @property
    def head_dim(self) -> int:
        """Dimension of each attention head."""
        return self.hidden_size // self.num_attention_heads


class TrainingConfig(BaseModel):
    """Training-time hyperparameters and run orchestration.

    One YAML drives the whole run (`python scripts/train.py --config
    configs/tiny.yaml`) rather than splitting model hyperparameters and
    run-orchestration settings (batch size, step counts, eval cadence)
    across two separately-merged config files — a deliberate simplicity
    choice: a second config layer would add a merge/precedence question
    ("which file wins if both set `learning_rate`?") for no real benefit
    at this project's scale.

    `batch_size`, `max_steps`, `warmup_steps`, `eval_interval`, and
    `save_interval` have no defaults — they're specific enough per-run
    choices that a silently-inherited default (e.g. training running
    "forever" because `max_steps` was forgotten) is worse than being forced
    to state them explicitly in every config.
    """

    model_config = {"extra": "forbid"}

    precision: Literal["fp32", "fp16", "bf16"] = "fp32"
    optimizer: Literal["adamw", "sgd"] = "adamw"
    learning_rate: float = Field(gt=0.0)
    weight_decay: float = Field(ge=0.0, default=0.1)
    gradient_clip: float = Field(gt=0.0, default=1.0)

    batch_size: int = Field(
        gt=0, description="Sequences per forward/backward micro-batch (before accumulation)"
    )
    gradient_accumulation_steps: int = Field(
        ge=1,
        default=1,
        description=(
            "Micro-batches accumulated before each optimizer step; "
            "effective batch size = batch_size * gradient_accumulation_steps"
        ),
    )
    max_steps: int = Field(gt=0, description="Total optimizer steps for this run")
    warmup_steps: int = Field(ge=0, description="Linear LR warmup steps before cosine decay")
    eval_interval: int = Field(gt=0, description="Steps between validation passes")
    save_interval: int = Field(gt=0, description="Steps between checkpoint saves")
    log_interval: int = Field(gt=0, default=10, description="Steps between progress log lines")
    min_lr_ratio: float = Field(
        ge=0.0,
        le=1.0,
        default=0.1,
        description="Cosine decay floor, as a fraction of the peak learning_rate",
    )
    seed: int = Field(default=42, description="Seed for reproducibility; recorded per checkpoint")

    @model_validator(mode="after")
    def check_warmup_within_max_steps(self) -> TrainingConfig:
        """Warmup that never finishes (>= max_steps) means the run never
        reaches its peak learning rate — almost certainly a config mistake
        rather than an intended setup, so reject it rather than silently
        training at a permanently-reduced LR.
        """
        if self.warmup_steps >= self.max_steps:
            raise ValueError(
                f"warmup_steps ({self.warmup_steps}) must be less than "
                f"max_steps ({self.max_steps})"
            )
        return self


class ModelConfig(BaseModel):
    """Top-level DaraLM configuration — the full contents of a config YAML.

    Load and validate with :meth:`from_yaml`, e.g.::

        config = ModelConfig.from_yaml("configs/50m.yaml")
    """

    model_config = {"extra": "forbid"}

    model_name: str = Field(min_length=1)
    tags: list[str] = Field(default_factory=list)
    architecture: ArchitectureConfig
    training: TrainingConfig

    @classmethod
    def from_yaml(cls, path: str | Path) -> ModelConfig:
        """Load and validate a config from a YAML file.

        Raises ``FileNotFoundError`` if the path doesn't exist, ``ValueError``
        if the file is empty, and ``pydantic.ValidationError`` if the
        contents don't match the schema. No error is ever swallowed —
        configuration mistakes should surface immediately, not silently
        produce a model with the wrong shape.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        if raw is None:
            raise ValueError(f"Config file is empty: {path}")
        return cls.model_validate(raw)
