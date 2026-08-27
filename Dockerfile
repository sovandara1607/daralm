# Multi-stage, CPU-only inference image; model artifacts are mounted at runtime.

FROM python:3.11-slim AS builder

RUN pip install --no-cache-dir uv

WORKDIR /app

# Install dependencies before source to preserve the cache layer.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY daralm/ ./daralm/
COPY api/ ./api/
COPY web/ ./web/
RUN uv sync --frozen --no-dev

FROM python:3.11-slim AS runtime

# curl supports the container health check.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 daralm
WORKDIR /app

# Set ownership during COPY to avoid duplicating layers.
COPY --from=builder --chown=daralm:daralm /app/.venv /app/.venv
COPY --from=builder --chown=daralm:daralm /app/daralm /app/daralm
COPY --from=builder --chown=daralm:daralm /app/api /app/api
COPY --from=builder --chown=daralm:daralm /app/web /app/web

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

# Prepare bind-mount locations for runtime artifacts.
RUN mkdir -p /app/configs /app/checkpoints /app/data \
    && chown daralm:daralm /app/configs /app/checkpoints /app/data
USER daralm

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
