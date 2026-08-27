from __future__ import annotations

from fastapi import APIRouter, Depends

from api.dependencies import get_model_service
from api.observability import record_tokens_generated
from api.schemas.chat import ChatRequest, ChatResponse
from api.services.model_service import ModelService

router = APIRouter(prefix="/v1", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(
    body: ChatRequest, service: ModelService = Depends(get_model_service)
) -> ChatResponse:
    result = await service.chat(
        instruction=body.instruction,
        max_new_tokens=body.max_new_tokens,
        temperature=body.temperature,
        top_p=body.top_p,
        top_k=body.top_k,
        repetition_penalty=body.repetition_penalty,
    )
    record_tokens_generated("chat", result["tokens_generated"])
    return ChatResponse(**result, model=service.config.model_name)
