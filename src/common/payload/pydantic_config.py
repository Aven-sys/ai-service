from pydantic import BaseModel
from typing import Dict, List, Any

# Define the structure for each object in the array
class PydanticConfig(BaseModel):
    name: str
    type: str
    default: Any
    description: str