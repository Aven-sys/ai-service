from pydantic import BaseModel
from typing import List

class EmbeddingResponseDto(BaseModel):
    embedding: List[float]
    