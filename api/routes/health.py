"""GET /health."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api.dependencies import get_model_service
from api.schemas.model_info import HealthResponse
from api.services.model_service import ModelService

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(service: ModelService = Depends(get_model_service)) -> HealthResponse:
    """Liveness/readiness check."""
    return HealthResponse(status="ok", device=str(service.device))
