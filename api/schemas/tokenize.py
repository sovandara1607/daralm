"""Request/response schema for POST /v1/tokenize — spec section 26.

Useful standalone, not just as a generate() implementation detail: Phase 9
found a real tokenizer coverage gap (`<`/`>` characters falling back to
`<unk>`) by inspecting `encode()`/`decode()` output directly. This endpoint
exposes that same inspection over HTTP instead of requiring a Python
console — e.g. for checking how a given piece of Khmer or English text (or
a chat-template-wrapped prompt) actually gets tokenized before spending a
generate() call on it.
"""

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
