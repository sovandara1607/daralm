"""POST /v1/generate."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api.dependencies import get_model_service
from api.observability import record_tokens_generated
from api.schemas.generate import GenerateRequest, GenerateResponse
from api.services.model_service import ModelService

router = APIRouter(prefix="/v1", tags=["generate"])


@router.post("/generate", response_model=GenerateResponse)
async def generate_text(
    body: GenerateRequest, service: ModelService = Depends(get_model_service)
) -> GenerateResponse:
    result = await service.generate(
        prompt=body.prompt,
        max_new_tokens=body.max_new_tokens,
        temperature=body.temperature,
        top_p=body.top_p,
        top_k=body.top_k,
        repetition_penalty=body.repetition_penalty,
        stop_on_eos=body.stop_on_eos,
    )
    record_tokens_generated("generate", result["tokens_generated"])
    return GenerateResponse(**result, model=service.config.model_name)
