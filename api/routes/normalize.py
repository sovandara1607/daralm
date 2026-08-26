"""POST /v1/normalize — Khmer text normalization.

Unlike every other route in this package, this one has **no model
dependency** — `daralm.data.khmer_normalize` is pure text processing, no
tokenizer or loaded checkpoint involved. So this route doesn't use
`get_model_service` at all, and works even before the model finishes
loading (same reasoning `/metrics` already applies for a different reason).
"""

from __future__ import annotations

from fastapi import APIRouter

from api.schemas.normalize import NormalizeRequest, NormalizeResponse
from daralm.data.khmer_normalize import normalize_khmer_text

router = APIRouter(prefix="/v1", tags=["normalize"])


@router.post("/normalize", response_model=NormalizeResponse)
def normalize(body: NormalizeRequest) -> NormalizeResponse:
    result = normalize_khmer_text(body.text, digits=body.digits)
    return NormalizeResponse(normalized_text=result, changed=result != body.text)
