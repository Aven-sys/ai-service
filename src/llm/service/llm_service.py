from ..payload.request.llm_chat_request_dto import LLMChatRequestDto
from ..util.llm_util import LLM

def generate_llm_response(llm_request_dto: LLMChatRequestDto):
    llm = LLM(
        input=llm_request_dto.input,
        model_name=llm_request_dto.model,
        system_Prompt=llm_request_dto.system_prompt,
        max_error_allowed=llm_request_dto.max_error_allowed,
        is_structured_output=llm_request_dto.is_structured_output,
        structure_output_config=llm_request_dto.structure_output_config
    )
    response =  llm.run(llm_request_dto.input)
    return response
