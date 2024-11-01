from pydantic import BaseModel
from typing import Dict, List, Any, Optional
from common.payload.pydantic_config import PydanticConfig

class LLMChatRequestDto(BaseModel):
    input : Dict[str,str] = {}
    model: Optional[str] = "gpt-4o-mini"
    max_error_allowed: Optional[int] = 3
    system_prompt: Optional[str] = None
    is_structured_output: Optional[bool] = False
    structure_output_config: Optional[List[PydanticConfig]] = None