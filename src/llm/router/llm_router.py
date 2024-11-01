from fastapi import APIRouter
from common.util import pydantic_util
from ..util.langchain_pydantic_model_generator import create_pydantic_model_from_config, print_model_fields
import json
from ..service import llm_service

# Request payload
from ..payload.request.llm_chat_request_dto import LLMChatRequestDto

router = APIRouter(
    prefix="/api/llm",
)

@router.post("")
async def llm_chat(llm_chat_request_dto: LLMChatRequestDto):
    # if llm_chat_request_dto.structure_output_config:
    #     pydantic_config = llm_chat_request_dto.structure_output_config
    #     DynamicModel1 = create_pydantic_model_from_config(json.dumps(pydantic_config))
    #      # Print the model schema to view structure and fields
    #     print_model_fields(DynamicModel1)

    response = llm_service.generate_llm_response(llm_chat_request_dto)
    return response
