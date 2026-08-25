"""POST /v1/tokenize — spec section 26."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api.dependencies import get_model_service
from api.schemas.tokenize import TokenizeRequest, TokenizeResponse
from api.services.model_service import ModelService

router = APIRouter(prefix="/v1", tags=["tokenize"])


@router.post("/tokenize", response_model=TokenizeResponse)
def tokenize(
    body: TokenizeRequest, service: ModelService = Depends(get_model_service)
) -> TokenizeResponse:
    return TokenizeResponse(**service.tokenize(body.text, body.add_bos, body.add_eos))
