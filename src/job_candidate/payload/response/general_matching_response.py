from typing import List, Any
from pydantic import BaseModel

class MatchResult(BaseModel):
    source_item: str
    target_item: str
    similarity_score: float

class GeneralMatchingResponse(BaseModel):
    matches: List[MatchResult]
    overall_score: float