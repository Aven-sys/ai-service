from pydantic import BaseModel
from typing import Dict, List, Any, Optional, Union
from common.payload.pydantic_config import PydanticConfig

class LLMChatRequestDto(BaseModel):
    input : Dict[str,str] = {}
    model: Optional[str] = "gpt-4o-mini"
    max_error_allowed: Optional[int] = 3
    system_prompt: Optional[str] = None
    is_structured_output: Optional[bool] = True
    structure_output_config: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None
