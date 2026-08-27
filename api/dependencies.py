"""FastAPI dependency for retrieving the process-wide `ModelService`."""

from __future__ import annotations

from fastapi import HTTPException, Request

from api.services.model_service import ModelService


def get_model_service(request: Request) -> ModelService:
    """Fetch the `ModelService` `api.main`'s startup attached to `app.state`."""
    service = getattr(request.app.state, "model_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet")
    return service
