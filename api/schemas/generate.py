"""Request/response schema for POST /v1/generate — spec section 26's example endpoint."""

from __future__ import annotations

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    """Mirrors the spec's literal example request body, plus the two extra
    sampling knobs `daralm.inference.sampling` already supports
    (`top_k`, `repetition_penalty`) — exposed as optional so the spec's
    exact example (`prompt`, `max_new_tokens`, `temperature`, `top_p`)
    still works unchanged with every other field defaulting to a no-op.
    """

    model_config = {"extra": "forbid"}

    prompt: str = Field(min_length=1, description="Seed text to continue")
    max_new_tokens: int = Field(
        default=100, gt=0, le=1024, description="Hard cap on generated tokens"
    )
    temperature: float = Field(default=1.0, ge=0.0, description="0.0 = greedy/deterministic")
    top_p: float = Field(default=1.0, gt=0.0, le=1.0, description="Nucleus sampling threshold")
    top_k: int = Field(default=0, ge=0, description="0 disables top-k filtering")
    repetition_penalty: float = Field(default=1.0, ge=1.0, description="1.0 = no penalty")
    stop_on_eos: bool = Field(default=True, description="Stop as soon as <eos> is generated")


class GenerateResponse(BaseModel):
    """Mirrors the spec's literal example response body, plus `model` for
    basic traceability (which checkpoint actually produced this text) —
    the one addition beyond the spec's exact two fields.
    """

    generated_text: str
    tokens_generated: int
    model: str
