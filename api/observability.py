from __future__ import annotations

import time

from prometheus_client import Counter, Histogram
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from daralm.utils.logging import get_logger

logger = get_logger(__name__)

REQUEST_COUNT = Counter(
    "daralm_requests_total",
    "Total HTTP requests handled",
    labelnames=("method", "path", "status_code"),
)
REQUEST_DURATION = Histogram(
    "daralm_request_duration_seconds",
    "HTTP request duration in seconds",
    labelnames=("method", "path"),
)
TOKENS_GENERATED = Counter(
    "daralm_tokens_generated_total",
    "Total tokens generated, by endpoint",
    labelnames=("endpoint",),
)


class ObservabilityMiddleware(BaseHTTPMiddleware):
    """Logs every request and records it into the Prometheus counters above."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        start = time.monotonic()
        response = await call_next(request)
        duration = time.monotonic() - start

        method = request.method
        path = request.url.path
        REQUEST_COUNT.labels(method=method, path=path, status_code=response.status_code).inc()
        REQUEST_DURATION.labels(method=method, path=path).observe(duration)
        logger.info("%s %s -> %d (%.3fs)", method, path, response.status_code, duration)
        return response


def record_tokens_generated(endpoint: str, count: int) -> None:
    TOKENS_GENERATED.labels(endpoint=endpoint).inc(count)
