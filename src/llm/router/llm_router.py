from fastapi import APIRouter
from ..service import llm_service

# Request payload
from ..payload.request.llm_chat_request_dto import LLMChatRequestDto

router = APIRouter(
    prefix="/api/llm",
)

@router.post("")
async def llm_chat(llm_chat_request_dto: LLMChatRequestDto):
    response = llm_service.generate_llm_response(llm_chat_request_dto)
    return response
