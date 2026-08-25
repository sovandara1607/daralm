"""FastAPI app entry point — spec section 26.

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

Defaults point at DaraLM-50M **Base**, not Instruct: `/v1/generate` is a
raw-completion endpoint (no chat template applied — spec section 26 lists
exactly `/health`, `/v1/model`, `/v1/tokenize`, `/v1/generate`, nothing
chat-specific), which is what Base was trained for. To serve Instruct
instead, point the three env vars at `configs/50m-instruct.yaml` /
`checkpoints/daralm-50m-instruct/best` — `/v1/generate` will still work
(the underlying `generate()` call doesn't care which checkpoint it's
given), it just won't wrap the prompt in the `<user>/<assistant>` chat
template the way `daralm.inference.generator.generate_chat` does; that
wiring is future work, not built in this phase, to avoid an unlisted
endpoint the spec didn't ask for.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.routes import generate, health, model, tokenize
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

app.include_router(health.router)
app.include_router(model.router)
app.include_router(tokenize.router)
app.include_router(generate.router)
