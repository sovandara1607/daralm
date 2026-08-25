"""FastAPI dependency for retrieving the process-wide `ModelService`.

Routes depend on `get_model_service`, not on a module-level global — that's
what lets `tests/test_api.py` swap in a tiny fixture model via FastAPI's
`app.dependency_overrides` instead of loading a real checkpoint per test.
"""

from __future__ import annotations

from fastapi import HTTPException, Request

from api.services.model_service import ModelService


def get_model_service(request: Request) -> ModelService:
    """Fetch the `ModelService` `api.main`'s startup attached to `app.state`.

    Raising a clean 503 here (rather than an `AttributeError`) covers the
    narrow window where a request somehow arrives before startup finished,
    or in a misconfigured deployment that never ran it at all.
    """
    service = getattr(request.app.state, "model_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet")
    return service
