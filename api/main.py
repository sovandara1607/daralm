"""FastAPI app entry point — spec section 26, plus the section 27
"production architecture" pieces built on top: `/v1/chat`, `/metrics`
(request counts/latency/tokens via `api.observability`, Prometheus text
format), structured request logging (same middleware), and `GET /`
serving a static frontend (`web/index.html`) so the API has something to
actually click through instead of only curl.

Run it:

    uv run uvicorn api.main:app --reload

Which model gets served is config-driven, never hard-coded (spec section
30's "do not hard-code model architecture" applies just as much to serving
as to training) — set via three environment variables, each with a default
that works out of the box against this project's own trained checkpoints:

    DARALM_CONFIG      configs/<model>.yaml           (default: configs/50m.yaml)
    DARALM_CHECKPOINT  checkpoints/<model_name>/<tag>  (default: checkpoints/daralm-50m/best)
    DARALM_TOKENIZER   the trained tokenizer .model    (default:
                       checkpoints/tokenizer/unigram.model)

Defaults point at DaraLM-50M **Base**, not Instruct — `/v1/generate` is a
raw-completion endpoint (spec section 26's exact example), which is what
Base was trained for. `/v1/chat` (spec section 27's "production
architecture" direction, not one of section 26's original four) wraps
`daralm.inference.generator.generate_chat`, applying the
`<user>/<assistant>` chat template `InstructionDataset` trains on — it's
only meaningful against an instruction-tuned checkpoint. Point the three
env vars at `configs/50m-instruct.yaml` / `checkpoints/daralm-50m-instruct/best`
to serve Instruct instead. Calling `/v1/chat` against Base doesn't error
(the model just runs whatever text it's given), but the output won't
look like a chat response — Base was never trained on the chat template.
This is a deployer choice, not something this file decides for you: no
endpoint is restricted to one checkpoint type.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.observability import ObservabilityMiddleware
from api.routes import chat, frontend, generate, health, metrics, model, normalize, tokenize
from api.services.model_service import ModelService
from daralm.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_CONFIG = "configs/50m.yaml"
DEFAULT_CHECKPOINT = "checkpoints/daralm-50m/best"
DEFAULT_TOKENIZER = "checkpoints/tokenizer/unigram.model"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model exactly once, at process startup — not per request.

    A `ValidationError`/`FileNotFoundError`/`ValueError` here (bad config,
    missing checkpoint, tokenizer/vocab mismatch) crashes startup with a
    clear message instead of the service coming up "successfully" and then
    500ing on the first real request — the same "fail loudly" principle
    `ModelConfig.from_yaml` already applies to training configs.
    """
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
