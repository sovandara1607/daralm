"""Request/response schema for POST /v1/tokenize."""

from __future__ import annotations

from pydantic import BaseModel, Field


class TokenizeRequest(BaseModel):
    model_config = {"extra": "forbid"}

    text: str = Field(min_length=0, description="Text to tokenize")
    add_bos: bool = Field(default=False)
    add_eos: bool = Field(default=False)


class TokenizeResponse(BaseModel):
    token_ids: list[int]
    tokens: list[str]  # human-readable subword pieces, e.g. "▁hello"
    token_count: int
