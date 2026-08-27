"""Request/response schema for POST /v1/generate."""

from __future__ import annotations

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
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
    generated_text: str
    tokens_generated: int
    model: str
