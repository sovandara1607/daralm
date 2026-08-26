"""GET /metrics — Prometheus text exposition format, spec section 27."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

router = APIRouter(tags=["observability"])


@router.get("/metrics")
def metrics() -> Response:
    """No auth, no model dependency — deliberately reachable even before
    the model finishes loading, so a scraper can tell the process is alive
    during the (few-second) startup window `/health`'s 503 covers.
    """
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
