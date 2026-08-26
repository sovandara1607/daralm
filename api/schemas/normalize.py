"""Request/response schema for POST /v1/normalize.

The first capability from the NLP-platform roadmap (`ROADMAP_NLP_PLATFORM.md`)
exposed over HTTP — matches `/v1/tokenize`'s pattern (a standalone utility
endpoint, not tied to generation) since normalization is exactly the kind
of thing worth checking before spending a `/v1/generate` or `/v1/chat`
call on text, the same reasoning `/v1/tokenize` was built on.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class NormalizeRequest(BaseModel):
    model_config = {"extra": "forbid"}

    text: str = Field(min_length=0, description="Text to normalize")
    digits: Literal["keep", "arabic", "khmer"] = Field(
        default="keep",
        description="'keep' leaves digits as-is; 'arabic' converts Khmer "
        "digits to 0-9; 'khmer' converts 0-9 to Khmer digit glyphs",
    )


class NormalizeResponse(BaseModel):
    normalized_text: str
    changed: bool  # true if normalization actually altered the input
