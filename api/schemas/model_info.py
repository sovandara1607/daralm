"""Response schemas for GET /health and GET /v1/model."""

from __future__ import annotations

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str  # "ok" — kept a plain string rather than an enum: this is
    device: str


class ModelInfoResponse(BaseModel):
    model_name: str
    tags: list[str]
    vocab_size: int
    hidden_size: int
    num_layers: int
    num_attention_heads: int
    max_position_embeddings: int
    parameters: int
    checkpoint_step: int | None
    device: str
