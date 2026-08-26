"""Request/response schema for POST /v1/chat.

Not in spec section 26's literal four endpoints — deliberately deferred
back in Phase 10 (see api/main.py's docstring at the time) to avoid an
unlisted endpoint the spec didn't ask for. Built now as the first piece of
the "production architecture" direction (spec section 27): `/v1/generate`
is a raw-completion endpoint (no chat template applied); this one wraps
`daralm.inference.generator.generate_chat`, which applies the
`<user>/<assistant>` chat template `InstructionDataset` trains on.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    model_config = {"extra": "forbid"}

    instruction: str = Field(min_length=1, description="The user's message")
    max_new_tokens: int = Field(
        default=150, gt=0, le=1024, description="Hard cap on generated tokens"
    )
    temperature: float = Field(default=1.0, ge=0.0, description="0.0 = greedy/deterministic")
    top_p: float = Field(default=1.0, gt=0.0, le=1.0, description="Nucleus sampling threshold")
    top_k: int = Field(default=0, ge=0, description="0 disables top-k filtering")
    repetition_penalty: float = Field(default=1.0, ge=1.0, description="1.0 = no penalty")


class ChatResponse(BaseModel):
    response: str
    tokens_generated: int
    model: str
