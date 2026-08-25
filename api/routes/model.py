"""GET /v1/model — spec section 26."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api.dependencies import get_model_service
from api.schemas.model_info import ModelInfoResponse
from api.services.model_service import ModelService

router = APIRouter(prefix="/v1", tags=["model"])


@router.get("/model", response_model=ModelInfoResponse)
def model_info(service: ModelService = Depends(get_model_service)) -> ModelInfoResponse:
    """Report exactly what's loaded — architecture shape, parameter count,
    and the checkpoint step it was restored from. A client should never
    have to guess which of Base/Instruct, or which training step, is
    answering its requests.
    """
    return ModelInfoResponse(**service.model_info())
