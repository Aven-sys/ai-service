from pydantic import BaseModel
from typing import List

# Model for each sentence pair
class RerankResult(BaseModel):
    text1: str
    text2: str
    score: float
    normalized_score: float

# Wrapper model for the list of sentence pairs
class ListRerankResponseDto(BaseModel):
    result: List[RerankResult]