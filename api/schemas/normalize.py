"""Request/response schema for POST /v1/normalize."""

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
