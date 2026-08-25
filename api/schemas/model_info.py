"""Response schemas for GET /health and GET /v1/model — spec section 26."""

from __future__ import annotations

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str  # "ok" — kept a plain string rather than an enum: this is
    # the one field a load balancer/uptime check greps for, not a place to
    # add ceremony.
    device: str


class ModelInfoResponse(BaseModel):
    """Everything a client needs to know about the currently-loaded model
    without generating anything — architecture shape, size, and the
    checkpoint/training step it was loaded from, all read straight from
    the config and checkpoint metadata already on disk (nothing here is
    hand-typed or duplicated from MODEL_CARD.md).
    """

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
