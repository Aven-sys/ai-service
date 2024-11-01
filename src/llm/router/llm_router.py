from fastapi import APIRouter
from common.util import pydantic_util

# Request payload
from ..payload.request.llm_chat_request_dto import LLMChatRequestDto

router = APIRouter(
    prefix="/api/llm",
)

@router.post("")
async def llm_chat(llm_chat_request_dto: LLMChatRequestDto):
    if llm_chat_request_dto.structure_output_config:
        pydantic_config = llm_chat_request_dto.structure_output_config
        model = pydantic_util.create_pydantic_model_from_config(pydantic_config)
        for field_name, field_info in model.model_fields.items():
            print(f"  Field: {field_name}")
            print(f"  Default: {field_info.default}")
            print(f"  Description: {field_info.description}")
            print()

    return {
        "message": "Hello World"
    }
