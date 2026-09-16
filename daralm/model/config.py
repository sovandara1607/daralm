"""Configuration schema for DaraLM model architecture and training."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class ArchitectureConfig(BaseModel):
    """Transformer architecture hyperparameters."""

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
        """hidden_size must split evenly across attention heads."""
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_attention_heads ({self.num_attention_heads}); "
                f"got remainder {self.hidden_size % self.num_attention_heads}"
            )
        return self

    @model_validator(mode="after")
    def check_head_dim_even(self) -> ArchitectureConfig:
        """head_dim must be even — required by Rotary Position Embeddings (Phase 3)."""
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
    """Training-time hyperparameters and run orchestration."""

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
    gradient_checkpointing: bool = Field(
        default=False,
        description=(
            "Trade recompute for memory by not storing each TransformerBlock's "
            "activations for backward (torch.utils.checkpoint). Only worth enabling "
            "if it lets batch_size increase within the same memory budget — see "
            "Model Optimization Stage 3."
        ),
    )
    compile: bool = Field(
        default=False,
        description=(
            "torch.compile the training step in place (nn.Module.compile — keeps "
            "state_dict keys and the model's class unchanged, unlike wrapping with "
            "torch.compile(model), which would break checkpoint compatibility). "
            "Measured ~1.2-1.4x step-time speedup on this project's CPU/MPS targets "
            "at 50m scale — see Model Optimization Stage 3."
        ),
    )

    @model_validator(mode="after")
    def check_warmup_within_max_steps(self) -> TrainingConfig:
        if self.warmup_steps >= self.max_steps:
            raise ValueError(
                f"warmup_steps ({self.warmup_steps}) must be less than max_steps ({self.max_steps})"
            )
        return self


class ModelConfig(BaseModel):
    """Top-level DaraLM configuration — the full contents of a config YAML."""

    model_config = {"extra": "forbid"}

    model_name: str = Field(min_length=1)
    tags: list[str] = Field(default_factory=list)
    architecture: ArchitectureConfig
    training: TrainingConfig

    @classmethod
    def from_yaml(cls, path: str | Path) -> ModelConfig:
        """Load and validate a config from a YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        if raw is None:
            raise ValueError(f"Config file is empty: {path}")
        return cls.model_validate(raw)
