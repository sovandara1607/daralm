"""Observability — the first layer of spec section 27's "production
architecture": request metrics + structured request logging.

Deliberately NOT a Prometheus/Grafana deployment (spec section 30: "do not
add [infrastructure] before it is needed") — just the standard first step
any of that would need anyway: a `/metrics` endpoint in Prometheus's own
text exposition format, produced by the standard `prometheus_client`
library rather than hand-rolled, so a real Prometheus server could scrape
this process the moment one exists, with zero code changes here.

Metrics are process-global (module-level, not per-`ModelService`) because
Prometheus's client library is designed that way — a `CollectorRegistry`
lives for the process's lifetime, same as the metrics it holds, regardless
of which model happens to be loaded into `app.state.model_service`.
"""

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
    """Logs every request and records it into the Prometheus counters above.

    Uses `request.url.path` (not `request.scope['route'].path`) for the
    label value — simpler, and fine at this project's scale (a handful of
    fixed routes, no high-cardinality path parameters like `/users/{id}`
    that would explode label cardinality in a real production system).
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        start = time.monotonic()
        response = await call_next(request)
        duration = time.monotonic() - start

        method = request.method
        path = request.url.path
        REQUEST_COUNT.labels(method=method, path=path, status_code=response.status_code).inc()
        REQUEST_DURATION.labels(method=method, path=path).observe(duration)
        logger.info(
            "%s %s -> %d (%.3fs)", method, path, response.status_code, duration
        )
        return response


def record_tokens_generated(endpoint: str, count: int) -> None:
    """Called by routes after a successful generate/chat call — kept as a
    plain function (not a method on `ModelService`) so `ModelService`
    doesn't need to know Prometheus exists; it stays HTTP/metrics-agnostic,
    same reasoning as why routes, not the service layer, own this.
    """
    TOKENS_GENERATED.labels(endpoint=endpoint).inc(count)
