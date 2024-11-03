from typing import List
from pydantic import BaseModel

class RerankSingleResult(BaseModel):
    text: str
    key_id: int
    score: float
    normalized_score: float

class ListSingleRerankResponseDto(BaseModel):
    result: List[RerankSingleResult]