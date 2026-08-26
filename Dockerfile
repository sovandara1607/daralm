# DaraLM inference API — spec section 26/27.
#
# Two stages: `builder` resolves and installs dependencies into a venv
# (this is the slow, cacheable step — it only reruns when pyproject.toml/
# uv.lock change, not on every source edit); `runtime` copies just that
# venv plus the application source into a clean, smaller image, and runs
# as a non-root user.
#
# What this image does NOT contain: checkpoints/, configs/, or data/.
# Those are mounted in at `docker run` time (see the Makefile's
# `docker-run` target) rather than baked in — api/main.py already decides
# which model to serve via DARALM_CONFIG/DARALM_CHECKPOINT/DARALM_TOKENIZER
# env vars, so the image itself stays reusable across every checkpoint
# this project trains, instead of needing a rebuild per model.
#
# Note on hardware: pyproject.toml's [tool.uv.sources] resolves torch
# against PyTorch's CPU-only wheel index specifically on Linux (this
# image's platform), instead of the default PyPI index's CUDA-bundled
# build — local dev on Apple Silicon (MPS) is untouched by that marker and
# keeps resolving its normal wheel. This container therefore always runs
# inference on CPU, `--gpus` or not. `daralm.utils.device.get_device()`
# already handles that fallback with no code changes — see its docstring
# — but generation will be slower than the MPS numbers in the README/
# MODEL_CARD files, which is expected, not a regression.

# --- builder -----------------------------------------------------------

FROM python:3.11-slim AS builder

# Matches this project's own pinned dev interpreter (.python-version) —
# same reasoning as that file's own comment: avoid being an early adopter
# on a CPython version PyTorch wheel support hasn't caught up to yet.

RUN pip install --no-cache-dir uv

WORKDIR /app

# Dependency files copied (and installed) before the rest of the source,
# so editing daralm/ or api/ later doesn't invalidate this layer — only a
# pyproject.toml/uv.lock change does.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY daralm/ ./daralm/
COPY api/ ./api/
COPY web/ ./web/
RUN uv sync --frozen --no-dev

# --- runtime -------------------------------------------------------------

FROM python:3.11-slim AS runtime

# curl: used only by the HEALTHCHECK below to hit GET /health the same way
# a real client would, over the network — not by the app itself.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 daralm
WORKDIR /app

# --chown here (not a separate `RUN chown -R` after the fact) matters a
# lot more than it looks: chown-ing an already-copied multi-GB directory
# touches every file's metadata, and on a layered filesystem that means
# the *entire* directory gets written again into a new layer just to
# change ownership — this doubled image size in the first version of this
# Dockerfile (two ~5GB layers: the COPY, then the chown). --chown sets
# ownership as part of the copy itself, so it's one write, not two.
COPY --from=builder --chown=daralm:daralm /app/.venv /app/.venv
COPY --from=builder --chown=daralm:daralm /app/daralm /app/daralm
COPY --from=builder --chown=daralm:daralm /app/api /app/api
COPY --from=builder --chown=daralm:daralm /app/web /app/web

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

# configs/, checkpoints/, and data/ are expected to be bind-mounted here at
# `docker run` time (see Makefile) — creating the mount points up front so
# they exist (and are owned by the non-root user) even if a caller forgets
# to mount one of them, rather than the container failing to even start a
# directory listing. Scoped to just these three (empty) directories, not
# `-R` over everything already copied above — same reasoning as the
# --chown note.
RUN mkdir -p /app/configs /app/checkpoints /app/data \
    && chown daralm:daralm /app/configs /app/checkpoints /app/data
USER daralm

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
