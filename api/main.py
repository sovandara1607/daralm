"""FastAPI app entry point."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from api.observability import ObservabilityMiddleware
from api.routes import chat, frontend, generate, health, metrics, model, normalize, tokenize
from api.services.model_service import ModelService
from daralm.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_CONFIG = "configs/50m.yaml"


def _first_existing(*paths: str) -> str:
    """Choose the first locally available serving artifact."""
    for path in paths:
        if Path(path).exists():
            return path
    return paths[0]


DEFAULT_CHECKPOINT = _first_existing(
    "checkpoints/daralm-50m/best",
    "experiments/hf_release/daralm-50m",
)
DEFAULT_TOKENIZER = _first_existing(
    "checkpoints/tokenizer/unigram.model",
    "experiments/hf_release/daralm-50m/tokenizer.model",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model exactly once, at process startup — not per request."""
    config_path = os.environ.get("DARALM_CONFIG", DEFAULT_CONFIG)
    checkpoint_dir = os.environ.get("DARALM_CHECKPOINT", DEFAULT_CHECKPOINT)
    tokenizer_path = os.environ.get("DARALM_TOKENIZER", DEFAULT_TOKENIZER)
    logger.info(
        "Starting DaraLM API: config=%s checkpoint=%s tokenizer=%s",
        config_path,
        checkpoint_dir,
        tokenizer_path,
    )
    app.state.model_service = ModelService.from_checkpoint(
        config_path, checkpoint_dir, tokenizer_path
    )
    yield
    app.state.model_service = None


app = FastAPI(
    title="DaraLM Inference API",
    description="Serving layer for DaraLM — a from-scratch decoder-only Transformer "
    "for Khmer + English. Educational/portfolio project; see the repo README and "
    "MODEL_CARD.md for what this model can and can't do.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(ObservabilityMiddleware)

app.include_router(health.router)
app.include_router(model.router)
app.include_router(normalize.router)
app.include_router(tokenize.router)
app.include_router(generate.router)
app.include_router(chat.router)
app.include_router(metrics.router)
app.include_router(frontend.router)
